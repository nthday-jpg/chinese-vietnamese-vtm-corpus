"""Import a previous Kaggle run without writing to its mounted Input files."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import uuid
import zipfile

from .ocr_records import read_checkpoint_campaign
from .ocr_worker import load_results


def checkpoint_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    temporary.replace(path)


def find_previous_campaign(directory):
    """Prefer the outer campaign; exported book manifests are a different format."""
    directory = Path(directory)
    for name in ('multi_crop_manifest.json','multi_variant_manifest.json','manifest.json'):
        path = directory/name
        if path.is_file():
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            if data.get('pairs') and data.get('models'):
                return path
    candidates = []
    for child in sorted(directory.iterdir()):
        if child.is_dir() and child.name not in {'runs','variant_runs','crop_runs','render_cache','evidence'}:
            for name in ('multi_crop_manifest.json','multi_variant_manifest.json','manifest.json'):
                path=child/name
                if path.is_file():
                    data=json.loads(path.read_text(encoding='utf-8-sig'))
                    if data.get('pairs') and data.get('models'):
                        candidates.append(path);break
    if len(candidates)>1:
        raise ValueError('PREVIOUS_OCR_DIR contains several campaigns; select the intended campaign directory')
    return candidates[0] if candidates else None


def extract_ocr_archive(source, destination):
    """Only checkpoint metadata/raw text are needed; leave image caches in the ZIP."""
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        selected=[]
        for entry in archive.infolist():
            name=entry.filename.replace('\\','/')
            if (':' in name or not (destination/name).resolve().is_relative_to(destination)
                    or (entry.external_attr>>16)&0o170000==0o120000):
                raise ValueError('Unsafe path in previous OCR ZIP')
            if name.lower().endswith(('.json','.jsonl')):selected.append(entry)
        archive.extractall(destination,members=selected)


def previous_book_logs(directory):
    pending=[Path(directory)];found=[]
    for _ in range(3):
        children=[]
        for folder in pending:
            manifest=folder/'manifest.json'
            if manifest.is_file() and (folder/'raw').is_dir():
                data=json.loads(manifest.read_text(encoding='utf-8-sig'))
                if data.get('format')=='vtm-ocr-book-v1':found.append((folder,data));continue
            children.extend(p for p in folder.iterdir() if p.is_dir() and p.name not in
                            {'raw','pages','crops','evidence','render_cache','runs'})
        pending=children
    return found


def restore_previous_ocr(source, work_dir, *, input_dir=None):
    """Accept a complete checkpoint directory/ZIP, or the exported raw book logs."""
    source=Path(source).resolve();work_dir=Path(work_dir).resolve()
    if not source.exists():raise FileNotFoundError(f'PREVIOUS_OCR_DIR does not exist: {source}')
    if source==work_dir or work_dir.is_relative_to(source) or source.is_relative_to(work_dir):
        raise ValueError('PREVIOUS_OCR_DIR must be separate from the writable WORK_DIR')
    root=source.parent if source.is_file() and source.suffix.lower()=='.json' else source
    hint=find_previous_campaign(source) if source.is_dir() else source
    stamp=(hint or source).stat()
    archives=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(source.glob('*.zip'))] if source.is_dir() else []
    fingerprint=hashlib.sha256(json.dumps([str(source),stamp.st_size,stamp.st_mtime_ns,archives]).encode()).hexdigest()[:16]
    destination=work_dir/'previous_ocr'/fingerprint
    marker=destination/'restored.json'
    if marker.is_file():
        restored=json.loads(marker.read_text(encoding='utf8'))
        if Path(restored['manifest']).is_file():return Path(restored['manifest'])
    if source.is_file() and source.suffix.lower()=='.zip':
        root=destination/'unpacked'
        extract_ocr_archive(source,root)
    elif not root.is_dir():
        raise ValueError('PREVIOUS_OCR_DIR must be a checkpoint directory, manifest.json, or ZIP')
    campaign=source if source.is_file() and source.suffix.lower()=='.json' else find_previous_campaign(root)
    if campaign is None:
        # Per-book *_results.zip exports contain raw JSONL and can be imported too.
        from .imported_logs import import_ocr_logs
        found=previous_book_logs(root)
        if not found:
            for archive in sorted(root.glob('*.zip')):
                unpacked=destination/'book_archives'/hashlib.sha256(archive.name.encode()).hexdigest()[:16]
                extract_ocr_archive(archive,unpacked)
                found.extend(previous_book_logs(unpacked))
        logs=destination/'book_logs';seen_books=set()
        for folder,info in found:
            book=str(info['book'])
            if book in seen_books:raise ValueError('Several exported OCR versions for book '+book+'; select one crop/campaign')
            seen_books.add(book)
            target=logs/hashlib.sha256(book.encode()).hexdigest()[:16]
            target.mkdir(parents=True,exist_ok=True)
            shutil.copy2(folder/'manifest.json',target/'manifest.json')
            shutil.copytree(folder/'raw',target/'raw',dirs_exist_ok=True)
        logs.mkdir(parents=True,exist_ok=True)
        # Restore all history; the new OCR plan controls book/page selection.
        result=import_ocr_logs(logs,destination,input_dir=input_dir,books=None,max_books=None,pages='',limit=None)
    else:
        allowed=root.resolve()
        copied={};copied_runs={}
        def copy_campaign(path):
            path=Path(path).resolve()
            if path in copied:return copied[path]
            if not path.is_relative_to(allowed):raise ValueError('Previous campaign references files outside PREVIOUS_OCR_DIR')
            data=read_checkpoint_campaign(path)
            target=destination/(hashlib.sha256(str(path).encode()).hexdigest()[:16]+'_'+path.name)
            copied[path]=target
            for field in ('variant_manifests','crop_manifests'):
                if data.get(field):data[field]={n:str(copy_campaign(p)) for n,p in data[field].items()}
            for pair in data.get('pairs',[]):
                for engine in data['models']:
                    old=Path(pair[engine]).resolve()
                    if not old.is_relative_to(allowed) or not (old/'run_manifest.json').is_file():
                        raise FileNotFoundError(f'Previous OCR run is missing; preserve the complete runs tree: {old}')
                    if old in copied_runs:
                        pair[engine]=str(copied_runs[old]);continue
                    new=destination/'runs'/hashlib.sha256(str(old).encode()).hexdigest()[:20]
                    copied_runs[old]=new
                    new.mkdir(parents=True,exist_ok=True)
                    for filename in ('run_manifest.json','tasks.jsonl'):
                        if (old/filename).is_file():shutil.copy2(old/filename,new/filename)
                    if (old/'ocr_raw').is_dir():shutil.copytree(old/'ocr_raw',new/'ocr_raw',dirs_exist_ok=True)
                    pair[engine]=str(new)
            checkpoint_json(target,data)
            return target
        copied_root=copy_campaign(campaign)
        result=destination/'manifest.json'
        shutil.copy2(copied_root,result)
    checkpoint_json(marker,{'source':str(source),'manifest':str(result)})
    print('Previous OCR restored to:',result,flush=True)
    return Path(result)


@lru_cache(maxsize=16)
def checkpoint_run_candidates(manifest_path, stamp):
    seen, runs = set(), []
    def visit(path):
        path=Path(path).resolve()
        if path in seen:return
        seen.add(path);data=read_checkpoint_campaign(path)
        for field in ('crop_manifests','variant_manifests'):
            for child in data.get(field,{}).values():visit(child)
        for pair in data.get('pairs',[]):
            for engine in data['models']:
                directory=Path(pair[engine])
                key=(pair['book'],engine,str(directory))
                if key not in seen:
                    seen.add(key)
                    manifest=json.loads((directory/'run_manifest.json').read_text(encoding='utf8'))
                    runs.append((pair['book'],engine,str(directory),manifest))
    visit(manifest_path)
    return tuple(runs)


def reuse_previous_ocr(resume_manifest, cfg, directory, book, source_sha, pages, config_key):
    if not resume_manifest:return 0
    path=Path(resume_manifest)
    candidates=checkpoint_run_candidates(str(path.resolve()),path.stat().st_mtime_ns)
    name=cfg.run_name or cfg.engine
    wanted=set(pages)-load_results(directory/'ocr_raw',book,name).keys()
    copied=0
    for old_book,engine,old_directory,manifest in candidates:
        if not wanted:break
        source=manifest.get('books',{}).get(book,{})
        if (old_book!=book or engine!=cfg.engine or not source_sha or source.get('sha256')!=source_sha
                or source.get('legacy_source_unverified') or config_key(manifest['config'])!=config_key(cfg)):
            continue
        old_name=manifest['config'].get('run_name') or engine
        records=load_results(Path(old_directory)/'ocr_raw',book,old_name)
        selected=sorted(wanted & records.keys())
        if selected:
            target=directory/'ocr_raw'/book/(name+'_shardprevious_'+hashlib.sha256(old_directory.encode()).hexdigest()[:16]+'.jsonl')
            target.parent.mkdir(parents=True,exist_ok=True)
            # A completed import shard may already contain pages from an earlier selection.
            with target.open('a',encoding='utf8') as stream:
                for page in selected:stream.write(json.dumps(records[page],ensure_ascii=False)+'\n')
            wanted.difference_update(selected);copied+=len(selected)
    if copied:print(f'[resume] {book} / {cfg.engine} / {cfg.image_crop_type} / {cfg.image_variant}: reused {copied} pages',flush=True)
    return copied
