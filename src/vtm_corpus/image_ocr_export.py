"""Portable prepared PNGs and their exact per-region OCR checkpoint records."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path
import re
import shutil
import uuid

from .image_dataset import dataset_open, dataset_page, dataset_file_hash, dataset_path
from .prepared_variants import restoration_index, OCR_IMAGE_VARIANTS


def image_ocr_name(value):
    text = re.sub(r'[^\w.-]+', '_', str(value), flags=re.UNICODE).strip('._')[:72] or 'book'
    if text.upper().split('.')[0] in {'CON', 'PRN', 'AUX', 'NUL', *[f'COM{i}' for i in range(1,10)], *[f'LPT{i}' for i in range(1,10)]}:
        text = '_' + text
    return text


def image_ocr_copy(root, entry, directory, relative, generated, verified):
    source = dataset_path(root, entry['path'])
    identity = (str(source), entry['sha256'])
    if identity not in verified:
        if dataset_file_hash(source) != entry['sha256']:
            raise ValueError(f'Prepared image checksum mismatch: {source}')
        verified.add(identity)
    target = dataset_path(directory, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or dataset_file_hash(target) != entry['sha256']:
        temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    generated.add(relative)


def image_region_ocr(run, book, page, region_id, method, image):
    if run is None:
        return {'status': 'not_run', 'text': None}
    record = run['records'].get((book, page-1))
    if record is None:
        return {'status': 'missing_output', 'text': None}
    regions = record['raw'].get('image_preparation', {}).get('regions', [])
    matched = [region for region in regions if region.get('canonical_region_id') == region_id]
    if len(matched) != 1:
        return {'status': 'unmapped_region' if not matched else 'ambiguous_region', 'text': None,
                'note': 'Page output is retained in raw/ or variants/; no region mapping was invented.'}
    region = matched[0]
    matched_image = (region.get('variant_sha256') == image['sha256'] and region.get('variant') == method)
    return {'status': ('image_identity_mismatch' if not matched_image else
                       'ok' if isinstance(region.get('text'), str) else 'missing_text'),
            'text': region.get('text'), 'region_record': region,
            'match_basis': 'prepared_image_sha256_and_canonical_region_id',
            'prepared_image_matches': matched_image,
            'quality_issues': record.get('quality_issues', []),
            'inference_input_size': region.get('input_size'), 'processor_size': region.get('processor_size')}


def image_ocr_html(items, book, models):
    labels = {'ok':'Đã có kết quả', 'original_retained':'Giữ nguyên ảnh gốc',
              'not_applicable':'Không áp dụng', 'skipped_by_configuration':'Không được chọn',
              'missing_output':'Thiếu OCR', 'not_run':'Chưa chạy OCR', 'image_unavailable':'Chưa có ảnh',
              'image_identity_mismatch':'Ảnh không khớp checkpoint OCR', 'missing_text':'Thiếu văn bản OCR',
              'unmapped_region':'Chưa ghép được OCR với vùng ảnh', 'ambiguous_region':'Ghép vùng chưa xác định'}
    body = ('<!doctype html><html lang="vi"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Ảnh và OCR · ' + escape(book) + '</title><link rel="stylesheet" href="style.css">'
            '<body><main><nav><a href="index.html">Mục lục</a> · '
            '<a href="image_ocr.jsonl">Metadata ảnh và OCR</a></nav><header><h1>Ảnh và OCR · ' + escape(book) + '</h1>'
            '<p>Ảnh đã chuẩn bị và văn bản từng vùng của bốn model. Không chọn thay bạn một bản OCR.</p>'
            '<p class="muted">PNG giữ nguyên byte; model có thể thu/phóng ảnh khi inference. '
            'Kích thước đầu vào và raw từng vùng nằm trong JSONL. Mask là ảnh chẩn đoán, không đưa vào OCR.</p></header>')
    for item in items:
        body += '<article class="card"><h2>' + escape(f"Trang {item['page']} · {item['method']} · {item['region_id']}") + '</h2>'
        body += '<p>' + escape(f"Trạng thái ảnh: {labels.get(item['image_state'], item['image_state'])} · provenance: {item.get('provenance') or 'chưa có ảnh'}") + '</p>'
        if item.get('reason'):
            body += '<p class="muted">' + escape(item['reason']) + '</p>'
        if item.get('image'):
            path = escape(item['image'], quote=True)
            body += f'<figure><a href="{path}"><img loading="lazy" src="{path}" alt="Ảnh đã chuẩn bị"></a></figure>'
        if item.get('diagnostic_only'):
            body += '<p>Ảnh chẩn đoán; không có kết quả OCR trên mask.</p>'
        else:
            body += '<div class="predictions">'
            for model in models:
                prediction = item['ocr'][model]
                body += '<section class="prediction"><h3>' + escape(model) + '</h3><p>' + escape(labels.get(prediction['status'],prediction['status'])) + '</p>'
                if prediction.get('text_file'):
                    body += '<a href="' + escape(prediction['text_file'], quote=True) + '">TXT</a>'
                text = prediction.get('text')
                body += '<pre>' + escape(text if text is not None else 'Chưa có OCR khớp với ảnh/vùng này.') + '</pre></section>'
            body += '</div>'
        body += '</article>'
    return body + '</main></body></html>'


def export_image_ocr(result, book, directory, rows, runs_by_method, generated, write_text, *, prepared_dataset=None):
    """Copy existing outputs and read checkpoints only; never render or run OCR."""
    if not prepared_dataset:
        configs = result.get('manifests', {}).get(book, {})
        prepared_dataset = next((value['config'].get('image_dataset') for value in configs.values()
                                 if value['config'].get('image_dataset')), None)
    if not prepared_dataset:
        first = next(iter(next(iter(runs_by_method.values())).values()))
        prepared_dataset = first['manifest']['config'].get('image_dataset')
    if not prepared_dataset:
        return {'status': 'not_applicable', 'images': 0, 'reason': 'No prepared image dataset in this checkpoint.'}
    root, descriptor = dataset_open(prepared_dataset)
    index, report = restoration_index(root, descriptor) if descriptor.get('restoration') else ({}, {})
    methods = list(dict.fromkeys(['original', *report.get('selected_variants', []), *runs_by_method]))
    if set(methods) - (set(OCR_IMAGE_VARIANTS) | {'tadisr_text_mask'}):
        raise ValueError('Unsupported image export method')
    models = list(result['models'])
    first = next(iter(next(iter(runs_by_method.values())).values()))
    crop_type = first['manifest']['config'].get('image_crop_type', 'group')
    crop_type = 'page' if crop_type == 'native' else crop_type
    expected_sha = first['manifest'].get('books', {}).get(book, {}).get('sha256')
    pages = {row['page'] for row in rows}
    candidates = [page for page in descriptor['pages'] if page['page']+1 in pages and
                  str(page['book']).removesuffix('.pdf') == str(book).removesuffix('.pdf') and
                  (not expected_sha or page['source_sha256'] == expected_sha)]
    if {page['page']+1 for page in candidates} != pages:
        raise ValueError(f'Prepared image dataset does not cover the exported source/pages: {book}')
    items, verified, text_rows = [], set(), {(method, model): [] for method in methods for model in models}
    partial, copied, seen, names = False, set(), set(), {}
    for page in sorted(candidates, key=lambda value: value['page']):
        record = dataset_page(root, descriptor, page['source_sha256'], page['page'])
        regions = [region for region in record['regions'] if region['crop_type'] == crop_type]
        if not regions:
            raise ValueError(f'Prepared crop type missing: {book}/p{page["page"]+1}/{crop_type}')
        restored_methods = {method for method in methods if any(index.get((page['source_sha256'], page['page'], region['id'], method), {}).get('state') == 'ok' for region in regions)}
        for region in regions:
            for method in methods:
                identity = (page['source_sha256'], page['page'], region['id'], method)
                if identity in seen:
                    continue
                seen.add(identity)
                saved = index.get(identity, {})
                entry = region['images'].get(method)
                state = 'ok' if entry else saved.get('state', 'not_run')
                provenance = 'source_observation' if method == 'original' else saved.get('provenance')
                if state == 'ok' and not entry:
                    entry = saved['image']
                retained = (method in restored_methods and state == 'not_applicable' and
                            str(saved.get('reason', '')).startswith(('selective_unihir:', 'selective_restoration:')))
                if retained:
                    entry = region['images']['original'];state = 'original_retained';provenance = 'source_observation'
                basename = f"{image_ocr_name(book)}-p{page['page']+1:05d}-{crop_type}{region['reading_order']+1:03d}"
                owner = names.setdefault((method, basename), identity)
                if owner != identity:
                    raise ValueError(f'Duplicate image/OCR filename for different regions: {basename}')
                relative = f"images/{'original' if retained else method}/{basename}.png" if entry else None
                item = dict(book=book, page=page['page']+1, source_sha256=page['source_sha256'],
                    method=method, crop_type=crop_type, region_id=region['id'], source_box=region['source_box'],
                    image_state=state, image=relative, image_sha256=entry['sha256'] if entry else None,
                    image_size=entry['size'] if entry else None, provenance=provenance,
                    diagnostic_only=method == 'tadisr_text_mask', reason=saved.get('reason'), ocr={})
                if state in {'failed','missing_assets','runtime_incompatible','inference_failed','timeout','resource_exceeded'}:
                    partial = True
                if entry and relative not in copied:
                    image_ocr_copy(root, entry, directory, relative, generated, verified)
                    copied.add(relative)
                if not item['diagnostic_only']:
                    for model in models:
                        if not re.fullmatch(r'[A-Za-z0-9_-]+', model):
                            raise ValueError('Unsafe model name in OCR export')
                        run = runs_by_method.get(method, {}).get(model)
                        prediction = (image_region_ocr(run, book, item['page'], region['id'], method, entry)
                                      if entry else {'status': 'image_unavailable', 'text': None})
                        if entry and prediction['status'] != 'ok':
                            partial = True
                        if prediction['status'] == 'ok' and prediction['text'] is not None:
                            text_path = f'ocr/{method}/{model}/{basename}.txt'
                            write_text(directory, text_path, prediction['text'], generated)
                            prediction['text_file'] = text_path
                        item['ocr'][model] = prediction
                        text_rows[method, model].append({**{k:v for k,v in item.items() if k!='ocr'}, **prediction})
                items.append(item)
    for (method, model), records in text_rows.items():
        if records:
            write_text(directory, f'ocr/{method}/{model}/regions.jsonl',
                       ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in records), generated)
    write_text(directory, 'image_ocr.jsonl', ''.join(json.dumps(item, ensure_ascii=False)+'\n' for item in items), generated)
    write_text(directory, 'image_ocr.html', image_ocr_html(items, book, models), generated)
    return {'status': 'partial' if partial else 'complete', 'images': len(copied),
            'regions': len(items), 'snapshot_id': descriptor['snapshot_id'],
            'index_html': 'image_ocr.html', 'metadata': 'image_ocr.jsonl',
            'methods': methods, 'match_basis': 'prepared_image_sha256_and_canonical_region_id'}
