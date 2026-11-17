"""Merge completed preparation runs locally without running restoration or OCR."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import uuid
import os

from .image_dataset import (dataset_open, dataset_read, dataset_path, dataset_write,
                            dataset_json_bytes, dataset_hash, dataset_file_hash, dataset_lock)
from .restoration_registry import VARIANT_SPECS
from .restoration_storage import PreparationStorageLimit


DATASET_MERGE_VERSION = 1


class DatasetMergeConflict(ValueError):
    """Inputs disagree about a source region or a method's output."""


def merge_copy_asset(source, target, *, reserve_bytes=2*1024**3, verify_hash=None):
    """Hardlink immutable files on the same volume; otherwise copy atomically."""
    source,target=Path(source),Path(target)
    if not source.is_file():raise FileNotFoundError(source)
    if verify_hash and dataset_file_hash(source)!=verify_hash:
        raise ValueError('Input image checksum mismatch: '+str(source))
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_file():
        if verify_hash and dataset_file_hash(target)!=verify_hash:
            raise ValueError('Existing merged image checksum mismatch: '+str(target))
        if target.stat().st_size!=source.stat().st_size:
            raise DatasetMergeConflict('Existing image has a different size: '+str(target))
        return 'reused'
    temporary=target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        try:
            os.link(source,temporary)
            action='hardlinked'
        except OSError:
            if shutil.disk_usage(target.parent).free < source.stat().st_size+reserve_bytes:
                raise PreparationStorageLimit('Not enough local disk space to copy: '+str(source))
            shutil.copyfile(source,temporary);action='copied'
        temporary.replace(target)
        return action
    finally:
        if temporary.exists():temporary.unlink()


def _merge_semantics(value):
    if isinstance(value,dict):
        return {k:_merge_semantics(v) for k,v in value.items() if k not in {'path','input_key'}}
    if isinstance(value,list):return [_merge_semantics(v) for v in value]
    return value


def _merge_config_map(target, incoming, label):
    for key,value in incoming.items():
        if key not in target:target[key]=copy.deepcopy(value)
        elif isinstance(value,dict) and isinstance(target[key],dict):
            _merge_config_map(target[key],value,label+'.'+str(key))
        elif target[key]!=value:
            if key in ('approved_variants','excluded_variants') and isinstance(value,list):
                target[key]=list(dict.fromkeys([*target[key],*value]))
            else:raise DatasetMergeConflict('Conflicting configuration: '+label+'.'+str(key))


def _merge_rows(root, reference):
    digest=hashlib.sha256()
    with dataset_path(root,reference['path']).open('rb') as stream:
        for line in stream:
            digest.update(line)
            if line.strip():yield json.loads(line)
    if digest.hexdigest()!=reference['sha256']:
        raise ValueError('Input manifest checksum mismatch: '+str(reference['path']))


def merge_prepared_datasets(inputs, output_dir, *, required_methods=None,
                            verify_image_hashes=False, reserve_bytes=2*1024**3, progress=None):
    """Publish a portable merged snapshot only after every method/region is present.

    Inputs must be completed Notebook 1 datasets. Images keep their original
    bytes. Small metadata files are rewritten to references inside this output.
    """
    if not str(output_dir).strip():raise ValueError('Configure OUTPUT_DIR for the local merged result')
    if type(reserve_bytes) is not int or reserve_bytes<0:raise ValueError('reserve_bytes must be nonnegative')
    output=Path(output_dir).expanduser().resolve();sources=[];source_ids=set()
    for value in inputs:
        if not str(value).strip():continue
        path=Path(value).expanduser()
        if path.is_dir() and not (path/'dataset.json').is_file() and (path/'_dataset/dataset.json').is_file():path=path/'_dataset'
        root,descriptor=dataset_open(path)
        root=root.resolve()
        if output==root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError('Merged output must be separate from every input _dataset')
        extension=descriptor.get('restoration')
        if not extension:raise ValueError('Input has no completed restoration report: '+str(root))
        report=json.loads(dataset_read(root,extension['report']))
        if (extension.get('format')!='vtm-image-restoration' or extension.get('version')!=1 or
            report.get('format')!='vtm-image-restoration' or report.get('version')!=1):
            raise ValueError('Unsupported input restoration format/version: '+str(root))
        canonical={k:v for k,v in descriptor.items() if k not in ('snapshot_id','restoration')}
        if dataset_hash(canonical)!=report.get('canonical_snapshot'):
            raise ValueError('Input canonical snapshot identity mismatch: '+str(root))
        if extension.get('partial') or report.get('partial'):
            raise ValueError('Finish the selected methods before merging: '+str(root))
        if descriptor['snapshot_id'] in source_ids:continue
        source_ids.add(descriptor['snapshot_id'])
        sources.append(dict(root=root,descriptor=descriptor,report=report,id=descriptor['snapshot_id']))
    if not sources:raise ValueError('Configure at least one completed input _dataset')
    selected=list(dict.fromkeys(['original',*[n for s in sources for n in s['report']['selected_variants']]]))
    if set(selected)-set(VARIANT_SPECS)-{'tadisr_text_mask'}:raise ValueError('Unsupported input method names')
    required=list(required_methods or [])
    missing=set(required)-set(selected)
    if missing:raise ValueError('Missing method datasets: '+', '.join(sorted(missing)))
    output.mkdir(parents=True,exist_ok=True)
    if (output/'dataset.json').is_file():dataset_open(output)
    stats=Counter();assets={};metadata_cache={};physical_pages={};aliases={};configurations={};crop_types=[]
    configuration={k:{} for k in ('model_options','variant_options','reviewed_regions','applicability')}
    preflight={}

    def remap_ref(source,reference,kind='asset'):
        cache_key=(source['id'],reference['path'],reference['sha256'],kind)
        if cache_key in metadata_cache:return {**copy.deepcopy(reference),**metadata_cache[cache_key]}
        filename=dataset_path(source['root'],reference['path'])
        if not filename.is_file():raise FileNotFoundError(filename)
        suffix=filename.suffix.lower()
        if suffix=='.json':
            value=json.loads(dataset_read(source['root'],reference))
            if kind=='layout':
                for key in ('preview','analysis_mask'):
                    if key in value:value[key]=remap_ref(source,value[key])
            payload=dataset_json_bytes(value)
            ref=dataset_write(output,'metadata/'+hashlib.sha256(payload).hexdigest()+'.json',payload)
            result={**reference,**ref}
        else:
            relative=('objects/'+reference['sha256'][:2]+'/'+reference['sha256']+'.png') if suffix=='.png' else (
                'assets/'+reference['sha256']+suffix)
            assets.setdefault(relative,(filename,reference['sha256']))
            result={**reference,'path':relative}
        metadata_cache[cache_key]={key:result[key] for key in ('path','sha256')}
        return copy.deepcopy(result)

    with dataset_lock(output),tempfile.TemporaryDirectory(prefix='vtm_merge_') as temporary:
        connection=sqlite3.connect(str(Path(temporary)/'rows.sqlite3'))
        try:
            connection.execute('CREATE TABLE rows (book TEXT, source TEXT, page INTEGER, region TEXT, variant TEXT, payload TEXT, semantic TEXT, PRIMARY KEY(book,source,page,region,variant))')
            for source in sources:
                descriptor,report=source['descriptor'],source['report']
                for key in configuration:
                    incoming=report.get('configuration',{}).get(key,{})
                    if key in ('model_options','variant_options','applicability'):
                        incoming={name:value for name,value in incoming.items() if name in report['selected_variants']}
                    _merge_config_map(configuration[key],incoming,key)
                for name,info in report.get('preflight',{}).items():
                    if name in report['selected_variants']:preflight.setdefault(name,info)
                for kind in report.get('crop_types',[]):
                    if kind not in crop_types:crop_types.append(kind)
                for name,cfg in descriptor.get('configurations',{}).items():
                    if name in configurations and configurations[name]!=cfg:
                        raise DatasetMergeConflict('Different preparation settings for book: '+name)
                    configurations[name]=cfg
                source['pages']={};source_expected=0
                for page in descriptor['pages']:
                    record=json.loads(dataset_read(source['root'],page['record']))
                    physical=(page['source_sha256'],page['page'])
                    alias=(page['book'],*physical)
                    source['pages'][alias]=record
                    source_expected+=len(record['regions'])*len(report['selected_variants'])
                    if physical in physical_pages:
                        original=physical_pages[physical]['original']
                        if _merge_semantics(original)!=_merge_semantics(record):
                            raise DatasetMergeConflict('Different original pixels/layout for '+str(alias))
                    else:
                        updated=copy.deepcopy(record)
                        updated['source']=remap_ref(source,record['source'])
                        updated['acquisition']=remap_ref(source,record['acquisition'])
                        updated['layout']=remap_ref(source,record['layout'],'layout')
                        for region in updated['regions']:
                            region['images']={name:remap_ref(source,ref) for name,ref in region['images'].items()}
                        payload=dataset_json_bytes(updated)
                        record_ref=dataset_write(output,'pages/'+hashlib.sha256(payload).hexdigest()+'.json',payload)
                        physical_pages[physical]=dict(original=record,record=updated,reference=record_ref)
                    aliases.setdefault(alias,{**page,'record':physical_pages[physical]['reference']})
                if progress:progress('Merging '+str(source['root'])+' : '+', '.join(report['selected_variants']))
                seen=set();states=Counter()
                for row in _merge_rows(source['root'],report['manifest']):
                    alias=(row['book'],row['source_sha256'],row['page_index'])
                    key=(*alias,row['region_id'],row['variant'])
                    if key in seen:raise DatasetMergeConflict('Duplicate input row: '+str(key))
                    seen.add(key);states[row['state']]+=1
                    if alias not in source['pages'] or row['variant'] not in report['selected_variants']:
                        raise ValueError('Input manifest contains an unexpected page/method')
                    original_record=source['pages'][alias]
                    regions={r['id']:r for r in original_record['regions']}
                    region=regions.get(row['region_id'])
                    if region is None or row['page']!=row['page_index']+1:
                        raise ValueError('Input manifest contains an invalid region/page')
                    if (row.get('source_image')!=original_record['source'] or
                        row.get('source_image_hash')!=original_record['source']['sha256'] or
                        row.get('canonical_image')!=region['images']['original'] or
                        any(row.get(k)!=region[k] for k in ('source_box','crop_type','reading_order'))):
                        raise ValueError('Input restoration row differs from its canonical source')
                    expected_provenance='neural_restoration' if row['variant']=='tadisr_text_mask' else VARIANT_SPECS[row['variant']][1]
                    if row.get('provenance')!=expected_provenance:raise ValueError('Invalid source provenance in restoration row')
                    if row['state'] not in ('ok','not_applicable','skipped_by_configuration'):
                        raise ValueError('Input has unfinished methods: '+str(key))
                    updated=copy.deepcopy(row)
                    canonical=physical_pages[(row['source_sha256'],row['page_index'])]['record']
                    new_region=next(r for r in canonical['regions'] if r['id']==row['region_id'])
                    updated['source_image']=canonical['source'];updated['source_image_hash']=canonical['source']['sha256']
                    updated['canonical_image']=new_region['images']['original']
                    if 'image' in updated:
                        updated['image']=new_region['images']['original'] if row['variant']=='original' else remap_ref(source,row['image'])
                    updated['extras']={name:remap_ref(source,ref) for name,ref in row.get('extras',{}).items()}
                    origin=dict(snapshot_id=source['id'],cache_key=row.get('key'),seconds=row.get('seconds'))
                    updated['merge_sources']=[origin]
                    semantic=dataset_json_bytes(_merge_semantics({k:updated.get(k) for k in (
                        'state','reason','image','extras','source_box','crop_type','reading_order','provenance',
                        'operation','parameters','coordinate_mapping_evidence','warnings')})).decode('utf8')
                    existing=connection.execute('SELECT payload,semantic FROM rows WHERE book=? AND source=? AND page=? AND region=? AND variant=?',key).fetchone()
                    if existing:
                        if existing[1]!=semantic:raise DatasetMergeConflict('Different outputs for the same method/region: '+str(key))
                        previous=json.loads(existing[0]);previous['merge_sources'].append(origin)
                        connection.execute('UPDATE rows SET payload=? WHERE book=? AND source=? AND page=? AND region=? AND variant=?',
                            (dataset_json_bytes(previous).decode('utf8'),*key))
                        stats['duplicate_rows']+=1
                    else:
                        connection.execute('INSERT INTO rows VALUES (?,?,?,?,?,?,?)',(*key,dataset_json_bytes(updated).decode('utf8'),semantic))
                if len(seen)!=source_expected or dict(states)!=report['states']:
                    raise ValueError('Input restoration manifest is incomplete or its counts disagree')
            expected=sum(len(physical_pages[(sha,page)]['record']['regions'])*len(selected) for book,sha,page in aliases)
            actual=connection.execute('SELECT COUNT(*) FROM rows').fetchone()[0]
            if actual!=expected:
                raise ValueError(f'Method/page coverage is incomplete: {actual}/{expected} rows. Use matching books/pages/crops for each method or supply the missing runs.')
            connection.commit()
            for index,(relative,(source,digest)) in enumerate(sorted(assets.items())):
                action=merge_copy_asset(source,output/relative,reserve_bytes=reserve_bytes,
                    verify_hash=digest if verify_image_hashes else None)
                stats[action]+=1
                if progress and index%5000==0:progress(f'Prepared {index+1}/{len(assets)} unique image/assets')
            pages=[aliases[key] for key in sorted(aliases)]
            canonical_rows=[]
            for page in pages:
                record=physical_pages[(page['source_sha256'],page['page'])]['record']
                for region in record['regions']:
                    canonical_rows.append(dict(book=page['book'],source_pdf=page['source_pdf'],source_sha256=page['source_sha256'],
                        page=page['page']+1,page_index=page['page'],source_image=record['source'],source_size=record['source']['size'],
                        region_id=region['id'],crop_type=region['crop_type'],source_box=region['source_box'],
                        reading_order=region['reading_order'],variant='original',**region['images']['original'],
                        layout=record['layout'],acquisition=record['acquisition'],processing_version=1))
            payload=b''.join(dataset_json_bytes(row) for row in canonical_rows)
            manifest=dataset_write(output,'manifests/'+hashlib.sha256(payload).hexdigest()+'.jsonl',payload)
            canonical=dict(format='vtm-image-dataset',version=1,pages=pages,
                expected_pages=[[p['book'],p['source_sha256'],p['page']] for p in pages],manifest=manifest,
                configurations=configurations,environment=sources[0]['descriptor'].get('environment',{}),
                code_sha256=dataset_hash({'merge_version':DATASET_MERGE_VERSION,'sources':[s['id'] for s in sources]}),
                geometry='top_left_source_pixels; half_open_boxes',ocr_text_coverage='unverified',
                merge_sources=[s['id'] for s in sources])
            canonical_id=dataset_hash(canonical)
            canonical_snapshot={**canonical,'snapshot_id':canonical_id}
            dataset_write(output,f'snapshots/{canonical_id}.json',dataset_json_bytes(canonical_snapshot))
            temporary_manifest=output/('.restoration_'+uuid.uuid4().hex+'.tmp');digest=hashlib.sha256();counts=Counter()
            try:
                with temporary_manifest.open('wb') as stream:
                    for (payload,) in connection.execute('SELECT payload FROM rows ORDER BY book,source,page,region,variant'):
                        data=payload.encode('utf8');stream.write(data);digest.update(data);counts[json.loads(payload)['state']]+=1
                relative='restoration_manifests/'+digest.hexdigest()+'.jsonl'
                target=output/relative;target.parent.mkdir(parents=True,exist_ok=True);temporary_manifest.replace(target)
            finally:
                if temporary_manifest.exists():temporary_manifest.unlink()
            configuration.update(variants=selected,crop_types=crop_types)
            report=dict(format='vtm-image-restoration',version=1,canonical_snapshot=canonical_id,
                manifest={'path':relative,'sha256':digest.hexdigest()},selected_variants=selected,
                crop_types=crop_types,states=dict(counts),partial=False,configuration=configuration,
                code_sha256=canonical['code_sha256'],preflight=preflight,
                merge_sources=[dict(snapshot_id=s['id'],selected_variants=s['report']['selected_variants'],
                    code_sha256=s['report'].get('code_sha256'),configuration=s['report'].get('configuration',{})) for s in sources])
            report_ref=dataset_write(output,'restoration_manifests/'+dataset_hash(report)+'.json',dataset_json_bytes(report))
            final={**canonical,'restoration':{'format':'vtm-image-restoration','version':1,'report':report_ref,'partial':False}}
            final['snapshot_id']=dataset_hash(final)
            dataset_write(output,f"snapshots/{final['snapshot_id']}.json",dataset_json_bytes(final))
            dataset_write(output,'dataset.json',dataset_json_bytes(final))
            result=dict(completed=True,dataset=str(output/'dataset.json'),snapshot_id=final['snapshot_id'],
                input_datasets=len(sources),methods=selected,pages=len(pages),manifest_rows=actual,states=dict(counts),
                unique_assets=len(assets),files=dict(stats),glyph_fidelity='unverified')
            dataset_write(output,'merge_summary.json',dataset_json_bytes(result))
            return result
        finally:connection.close()
