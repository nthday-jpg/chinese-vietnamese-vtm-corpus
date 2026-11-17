"""Recover display exports from completed receipts without publishing an OCR dataset."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DOCUMENT_VARIANTS = {  # Archival task identities only, no adapters imported.
    'docentr_binarization': ('docentr','binarization_base_2018_patch8','model_8_2018_base.pt'),
    'degan_binarization': ('degan','binarization','degan_weights.bundle'),
    'degan_deblurring': ('degan','deblurring','degan_weights.bundle'),
    'degan_watermark_removal': ('degan','watermark_removal','degan_weights.bundle'),
    'docres_appearance': ('docres','appearance','docres.pkl'),
    'docres_deblurring': ('docres','deblurring','docres.pkl'),
    'docres_deshadowing': ('docres','deshadowing','docres.pkl'),
    'docres_binarization': ('docres','binarization','docres.pkl'),
    'docdiff_deblurring': ('docdiff','deblurring','init.pth'),
    'restormer_motion_deblurring': ('restormer','motion','motion_deblurring.pth'),
    'restormer_defocus_deblurring': ('restormer','defocus','single_image_defocus_deblurring.pth'),
    'restormer_real_denoising': ('restormer','real','real_denoising.pth'),
    'restormer_gaussian_denoising': ('restormer','gaussian_color_blind','gaussian_color_denoising_blind.pth'),
}
from .image_dataset import dataset_hash, dataset_open, dataset_read


def prepared_checkpoint_export(directory, code_signatures, opencv_versions, *, progress=None):
    root=Path(directory).resolve()
    attempt_path=root/'last_attempt.json'
    attempt=json.loads(attempt_path.read_text(encoding='utf8')) if attempt_path.is_file() else {}
    plan=json.loads((root/'last_plan.json').read_text(encoding='utf8'))
    selected=list(plan['by_variant'])
    snapshot_id=attempt.get('canonical_snapshot')
    candidates=[root/'snapshots'/f'{snapshot_id}.json'] if snapshot_id else [
        p for p in (root/'snapshots').glob('*.json')
        if not json.loads(p.read_text(encoding='utf8')).get('restoration')]
    if len(candidates)!=1:
        raise ValueError('Cannot identify the canonical snapshot for this interrupted run; keep last_attempt.json and its canonical_snapshot')
    _,base=dataset_open(candidates[0])
    codes=list(dict.fromkeys(code_signatures));versions=list(dict.fromkeys(opencv_versions))
    if not codes or not versions:raise ValueError('Checkpoint export requires the saved code identity and OpenCV version')
    cache_files={p.stem:p for p in (root/'restoration_cache').glob('*.json')}
    empty_identity=dict(model=None,checkpoint=None,architecture=None,vendor=None,runtime=None,document_task=None)
    identities={name:{} for name in selected}
    for name,item in plan['by_variant'].items():
        if item['method']['provenance'] in ('source_observation','deterministic_transformation'):
            identities[name][dataset_hash(empty_identity)]=empty_identity
    architectures={'RRDBNet(3,3,64,23,32), scale=4':'realesrgan_x4',
        'CogView4-RealCE with upstream joint segmentation decoder':'tadisr_x4',
        'UniHIR BAGEL/MLLM':'unihir_restoration',
        'DiffTSR released IDM/TDM/MoM + TransOCR prior':'difftsr'}
    documents={tuple(value):name for name,value in DOCUMENT_VARIANTS.items()}
    invalid_receipts=0;results={}
    def read_receipt(item):
        key,path=item
        try:
            result=json.loads(path.read_text(encoding='utf8'))
            if result.get('key')!=key or result.get('state')!='ok':return key,None
            return key,result
        except (OSError,ValueError,TypeError):return key,None
    items=list(cache_files.items())
    # Small JSON files are read once, concurrently and in bounded batches.
    # Do not decode/hash source PNGs while reconstructing the export plan.
    with ThreadPoolExecutor(max_workers=min(16,max(1,len(items)))) as pool:
        for start in range(0,len(items),512):
            for key,result in pool.map(read_receipt,items[start:start+512]):
                if result is None:invalid_receipts+=1;continue
                identity=result.get('model_identity',{})
                name=documents.get(tuple(identity.get('document_task') or ())) or architectures.get(identity.get('architecture'))
                if name in identities:identities[name][dataset_hash(identity)]=identity
                results[key]={'image':result.get('image'),'mask':result.get('extras',{}).get('tadisr_text_mask')}
            if progress and (start//512)%10==0:progress(f'Read {min(start+512,len(items))}/{len(items)} completed-cache records')
    rows=[];matched=set();counts={name:0 for name in selected}
    ambiguous=[name for name,values in identities.items() if len(values)>1]
    def add(page,region,name,image):
        rows.append(dict(book=page['book'],source_sha256=page['source_sha256'],page=page['page']+1,
            crop_type=region['crop_type'],reading_order=region['reading_order'],variant=name,state='ok',image=image))
        counts[name]=counts.get(name,0)+1
    for page_number,page in enumerate(base['pages']):
        record=json.loads(dataset_read(root,page['record']))
        for region in record['regions']:
            original=region['images']['original']
            add(page,region,'original',original)
            for name in selected:
                if name in ('original','tadisr_text_mask') or len(identities[name])!=1:continue
                identity=next(iter(identities[name].values()))
                result=None
                for code in codes:
                    for version in versions:
                        key=dataset_hash(dict(source=original['sha256'],box=region['source_box'],name=name,
                            options=plan['by_variant'][name]['parameters'],model=identity,code=code,opencv=version))
                        if key in results and results[key]['image']:
                            result=results[key];matched.add(key);break
                    if result is not None:break
                if result is not None:
                    add(page,region,name,result['image'])
                    mask=result.get('mask')
                    if name=='tadisr_x4' and mask:add(page,region,'tadisr_text_mask',mask)
        if progress and page_number%100==0:progress(f'Matched {page_number+1}/{len(base["pages"])} pages to completed cache')
    recovery=dict(export_only=True,partial=True,matched_receipts=len(matched),cache_receipts=len(cache_files),
        invalid_receipts=invalid_receipts,ambiguous_methods=ambiguous,methods=counts,
        code_signatures=codes,opencv_versions=versions)
    report=dict(partial=True,selected_variants=selected,states={'ok':len(rows)},export_rows=rows,recovery=recovery)
    return root,base,report
