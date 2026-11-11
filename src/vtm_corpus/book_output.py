"""Portable UTF-8 OCR books: static HTML, page JSONL, crop evidence and final ZIP."""
from __future__ import annotations

import hashlib
from copy import deepcopy
from html import escape as report_escape
import json
from pathlib import Path
import re
import time
import zipfile

from .character_regions import character_crops
from .corpus_ocr import now, write_json
from .ocr_records import load_run, read_jsonl
from .ocr_worker import render
from .page_consensus import page_consensus_summary, page_structure_overview
from .evidence_report import portable_evidence, book_spatial_coverage, book_variant_stability, EVIDENCE_JS


BOOK_OUTPUT_FORMAT = 'vtm-ocr-book-v1'
BOOK_CSS = '''
:root{color-scheme:light;--ink:#172b36;--muted:#526471;--line:#d5dfe3;--accent:#075e69}
*{box-sizing:border-box}body{margin:0;background:#f3f6f5;color:var(--ink);font:16px/1.65 system-ui,sans-serif}
main{max-width:1150px;margin:auto;padding:24px}header,section,.card{background:white;border:1px solid var(--line);border-radius:12px;padding:20px;margin-bottom:18px}
h1,h2,h3{line-height:1.3}a{color:var(--accent)}nav{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:16px}
.muted{color:var(--muted)}.notice{border-left:4px solid #b66c00;padding:12px;background:#fff6df}
table{width:100%;border-collapse:collapse}th,td{padding:9px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
.scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:18px/1.9 'Noto Serif CJK SC','Microsoft YaHei',serif}
.tabs{display:flex;gap:8px;flex-wrap:wrap}button,input{font:inherit;padding:9px 14px;border:1px solid var(--line);border-radius:6px;background:white}
button{cursor:pointer}button[aria-selected=true]{background:var(--accent);color:white}.panel[hidden]{display:none}
.predictions{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}.prediction{background:#f2f6f7;padding:10px;overflow-wrap:anywhere}
.images{display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap}.images figure{margin:0;max-width:100%}img{max-width:100%;height:auto;border:1px solid var(--line)}figcaption{color:var(--muted);font-size:14px}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;background:#e4f4ed}.bad{background:#fff0d9}.status{font-weight:600}
@media(max-width:600px){main{padding:10px}header,section,.card{padding:14px}th,td{padding:6px}}
'''
BOOK_JS = '''
document.querySelectorAll('[data-model]').forEach(button=>button.addEventListener('click',()=>{
 document.querySelectorAll('[data-model]').forEach(b=>b.setAttribute('aria-selected',String(b===button)));
 document.querySelectorAll('.panel').forEach(p=>p.hidden=p.id!==button.dataset.model);
}));
const filter=document.querySelector('#page-filter');
if(filter) filter.addEventListener('input',()=>document.querySelectorAll('tbody tr[data-page]').forEach(row=>{
 row.hidden=!row.textContent.toLowerCase().includes(filter.value.toLowerCase());
}));
'''


def book_output_slug(book):
    slug = re.sub(r'[^\w.-]+', '_', str(book), flags=re.UNICODE).strip('._')[:100] or 'book'
    if slug.upper().split('.')[0] in {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]}:
        slug = '_' + slug
    return slug


def book_file(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError(f'Output path escapes book folder: {relative}')
    return path


def book_write_text(root, relative, text, generated):
    path = book_file(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8', newline='\n')
    temporary.replace(path)
    generated.add(relative)


def book_html(title, body, *, page=False):
    return ('<!doctype html><html lang="vi"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{report_escape(title)}</title><link rel="stylesheet" href="' +
            ('../' if page else '') + 'style.css">'
            f'<body><main>{body}</main><script src="' +
            ('../' if page else '') + 'reader.js"></script></body></html>')


def book_prediction(value):
    return 'chưa có output' if value is None else value or '∅'


def book_page_consensus(diagnostic):
    """Portable text/coordinate evidence; missing geometry stays unresolved."""
    if not diagnostic:
        return ''
    html = '<section><h2>Đồng thuận cấp trang</h2><p>' + report_escape(diagnostic['status']) + '</p>'
    html += '<p class="muted">Dấu hiệu đối chiếu giữa model; chưa chứng minh cột thật bị thiếu, lặp hoặc văn bản là bịa. Đa số có thể cùng sai.</p>'
    if diagnostic.get('overview'):
        html += '<figure><img loading="lazy" src="../'+report_escape(diagnostic['overview'],quote=True)+'" alt="Ngữ cảnh các vùng có dấu hiệu bất đồng cấp trang"><figcaption>Khung đỏ là vùng ngữ cảnh do model ước lượng; số khung trùng số vùng bên dưới. Vùng chỉ có mốc chữ không được vẽ khung.</figcaption></figure>'
    findings = diagnostic['findings']
    if findings:
        html += '<div class="scroll"><table><thead><tr><th>Model</th><th>Dấu hiệu</th><th>Vùng</th><th>Đối chứng</th><th>Luồng văn bản</th></tr></thead><tbody>'
        for item in findings:
            stream = ('Output gốc, đã chuẩn hóa trước so ký tự' if item.get('corrected_before_character_consensus')
                      else 'Văn bản đưa vào đối chiếu')
            html += '<tr>' + ''.join('<td>'+report_escape(value)+'</td>' for value in (
                item['model'],item['label'],', '.join(map(str,item['region_ids'])),', '.join(item['supporters']),stream)) + '</tr>'
        html += '</tbody></table></div>'
    else:
        html += '<p>Chưa phát hiện bất đồng cấu trúc với bằng chứng hiện có; không có nghĩa OCR đã đầy đủ hoặc đúng.</p>'
    selected = {number for item in findings for number in item['region_ids']}
    for region in diagnostic['regions']:
        if region['id'] not in selected:
            continue
        html += f'<details class="card"><summary>Vùng {region["id"]} · mốc từ {report_escape(region["reference_model"])}</summary>'
        html += '<pre>'+report_escape(region['text'])+'</pre>'
        if region['location']:
            html += '<p class="muted">Khung ngữ cảnh do model ước lượng (tọa độ chuẩn hóa 0–1): ' + report_escape(str(region['location']['bbox'])) + '. Khung không xác nhận cùng vị trí ảnh giữa các model.</p>'
        else:
            html += '<p class="muted">Chưa định vị trên ảnh; đây là vùng theo mốc văn bản.</p>'
        html += '<ul>'
        for name,match in region['models'].items():
            spans = [m['span'] for m in match['matches']]
            html += '<li>'+report_escape(f'{name}: {match["status"]}; số lần khớp={match.get("copies", "chưa rõ")}; offsets={spans}')+'</li>'
        html += '</ul><p class="muted">Offsets đếm từ 0 trong chuỗi bỏ khoảng trắng; không phải tọa độ ảnh.</p></details>'
    return html+'</section>'


def book_save_crop(root, relative, image, generated):
    path = book_file(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.png.tmp')
    image.save(temporary, format='PNG', compress_level=1)
    temporary.replace(path)
    generated.add(relative)


def zip_ocr_book(directory, archive_path):
    """ZIP lives beside the book; PNGs are stored, text is lightly compressed."""
    directory, archive_path = Path(directory).resolve(), Path(archive_path).resolve()
    if archive_path.is_relative_to(directory):
        raise ValueError('ZIP must be outside the book directory')
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1,
                         allowZip64=True, strict_timestamps=False) as archive:
        for path in sorted(directory.rglob('*')):
            if path.is_symlink() or not path.resolve().is_relative_to(directory):
                raise ValueError(f'Cannot archive external/symlink path: {path}')
            if path.is_file():
                compression = zipfile.ZIP_STORED if path.suffix.lower() in {'.png','.jpg','.jpeg','.webp'} else zipfile.ZIP_DEFLATED
                archive.write(path, arcname=f'{directory.name}/{path.relative_to(directory).as_posix()}',
                              compress_type=compression, compresslevel=1)
    with zipfile.ZipFile(temporary) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f'ZIP CRC failed: {bad}')
    temporary.replace(archive_path)
    return archive_path


def export_ocr_book(result, book, directory, *, include_crops=True, create_zip=True, dpi=250, renderer=None):
    """Export selected pages; missing outputs remain explicit and never become votes."""
    started = time.monotonic()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    previous_path = book_file(directory, 'manifest.json')
    previous = json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.is_file() else {}
    if previous and previous.get('format') != BOOK_OUTPUT_FORMAT:
        raise ValueError(f'Folder contains another output format: {directory}')
    rows = sorted((row for row in result['rows'] if row['book'] == book), key=lambda row: row['page'])
    if not rows:
        raise ValueError(f'Book has no selected pages: {book}')
    campaign_path = Path(result['corpus_manifest'])
    campaign = json.loads(campaign_path.read_text(encoding='utf-8'))
    pair = next(pair for pair in campaign['pairs'] if pair['book'] == book)
    models = result['models']
    warnings, runs, errors, generated = [], {}, [], set()
    stage_path = campaign_path.parent / 'stage_errors.json'
    if stage_path.is_file():
        for model, error in json.loads(stage_path.read_text(encoding='utf-8')).items():
            errors.append({'model': model, 'book': book, 'page': None, 'stage_error': True, **error})
            warnings.append(f'{model}: {error["error"]}: {error["message"]}')
    for model in models:
        runs[model] = load_run(Path(pair[model]), model, warnings,
                               reading_order=result.get('settings', {}).get('reading_order', 'auto'))
        cfg = runs[model]['manifest']['config']
        error_path = Path(pair[model]) / 'ocr_raw' / book / f"{cfg.get('run_name') or model}_errors.jsonl"
        if error_path.is_file():
            for error in read_jsonl(error_path, allow_incomplete_tail=True):
                errors.append({'model': model, 'book': book, **error, 'page': error['page'] + 1})
        lines = []
        for row in rows:
            record = runs[model]['records'].get((book, row['page'] - 1))
            quality = row.get('quality_issues', {}).get(model, [])
            geometry = record.get('geometry', record['raw']) if record else {}
            lines.append(json.dumps({'page': row['page'],
                'status': 'quality_review' if quality else 'ok' if record else 'missing',
                'quality_issues': quality, 'reading_order': record.get('reading_order') if record else None,
                'text': record['text'] if record else None, 'raw': record['raw'] if record else None,
                'comparison_geometry': {key: geometry[key] for key in ('character_boxes', 'text_lines', 'coordinate_frame', 'w', 'h') if key in geometry}}, ensure_ascii=False))
        book_write_text(directory, f'raw/{model}.jsonl', '\n'.join(lines) + '\n', generated)
        log = campaign_path.parent / f'{model}_notebook.log'
        if log.is_file():
            book_write_text(directory, f'logs/{model}.log', log.read_text(encoding='utf-8', errors='replace'), generated)
    book_write_text(directory, 'errors.jsonl', ''.join(json.dumps(error, ensure_ascii=False) + '\n' for error in errors), generated)
    book_write_text(directory, 'style.css', BOOK_CSS, generated)
    book_write_text(directory, 'reader.js', BOOK_JS + (EVIDENCE_JS if result.get('evidence_schema') else ''), generated)
    for variant, variant_result in result.get('variant_results', {}).items():
        for model in models:
            vr = load_run(Path(next(p[model] for p in read_jsonl_campaign(variant_result['corpus_manifest']) if p['book'] == book)), model, warnings)
            selected = [{'page': r['page'], 'status': 'ok' if (book, r['page']-1) in vr['records'] else 'missing_output',
                         'record': vr['records'].get((book, r['page']-1))} for r in rows]
            book_write_text(directory, f'variants/{variant}/{model}.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected), generated)
    comparisons, index_rows = [], []
    render_count = crop_count = structural_overviews = 0
    for position, row in enumerate(rows):
        page, consensus = row['page'], row['consensus']
        records = {model: runs[model]['records'].get((book, page - 1)) for model in models}
        nav = ['<a href="../index.html">Mục lục</a>']
        if position:
            nav.append(f'<a href="p{rows[position-1]["page"]:05d}.html">← Trang trước</a>')
        if position + 1 < len(rows):
            nav.append(f'<a href="p{rows[position+1]["page"]:05d}.html">Trang sau →</a>')
        body = f'<nav>{"".join(nav)}</nav><header><h1>Trang {page} · {report_escape(book)}</h1>'
        body += '<p class="muted">Đối chiếu các dự đoán OCR; đa số không phải đáp án đã kiểm chứng.</p></header>'
        if row['reasons']:
            body += '<p class="notice">Trạng thái: ' + report_escape(', '.join(row['reasons'])) + '</p>'
        tabs, panels = [], []
        for index, model in enumerate(models):
            safe_model = report_escape(model, quote=True)
            record = records[model]
            text = record['text'] if record else 'Chưa có output cho trang này. Xem errors.jsonl hoặc log model.'
            tabs.append(f'<button type="button" role="tab" data-model="model-{safe_model}" aria-selected="{str(index==0).lower()}">{safe_model}</button>')
            quality = consensus.get('excluded_models', {}).get(model, [])
            notice = ('<p class="notice">Không bỏ phiếu; cần kiểm tra output: ' + report_escape(', '.join(quality)) + '</p>') if quality else ''
            panels.append(f'<article class="panel" id="model-{safe_model}"{ " hidden" if index else ""}><h3>{safe_model}</h3>{notice}<pre>{report_escape(text)}</pre></article>')
        body += '<section><h2>Văn bản OCR</h2><div class="tabs" role="tablist">' + ''.join(tabs) + '</div>' + ''.join(panels) + '</section>'
        diagnostic = deepcopy(row.get('page_consensus'))
        image = None
        if diagnostic and include_crops and row['source_verified'] and row['source'] and Path(row['source']).is_file():
            numbers = {n for item in diagnostic['findings'] for n in item['region_ids']}
            if any(r['id'] in numbers and r['location'] for r in diagnostic['regions']):
                image = (renderer or render)(row['source'],page-1,dpi,100_000)
                render_count += 1
                overview = page_structure_overview(image,diagnostic)
                filename = f'crops/p{page:05d}/page_structure.png'
                book_save_crop(directory,filename,overview,generated)
                overview.close()
                diagnostic['overview'] = filename
                structural_overviews += 1
        body += '<div id="page-consensus">'+book_page_consensus(diagnostic)+'</div>'
        spatial = portable_evidence(row.get('spatial_coverage'), directory, page, generated)
        stability = portable_evidence(row.get('variant_stability'), directory, page, generated)
        body += book_spatial_coverage(spatial) + book_variant_stability(stability)
        body += '<section id="character-consensus"><h2>Vùng bất đồng</h2>'
        regions = []
        for region in consensus.get('conflict_regions', []):
            region = dict(region)
            number = region['number']
            body += f'<article class="card"><h3>Vùng {number}</h3><div class="predictions">'
            for model, value in region['predictions'].items():
                prediction = 'không bỏ phiếu (output cần kiểm tra)' if model in consensus.get('excluded_models', {}) else book_prediction(value)
                body += f'<div class="prediction"><strong>{report_escape(model)}</strong><br>{report_escape(prediction)}</div>'
            body += '</div>'
            location = region['localization']
            if location and include_crops and row['source_verified'] and row['source'] and Path(row['source']).is_file():
                if image is None:
                    image = (renderer or render)(row['source'], page - 1, dpi, 100_000)
                    render_count += 1
                crop, context, rectangle = character_crops(image, location)
                crop_name = f'crops/p{page:05d}/region_{number:04d}.png'
                context_name = f'crops/p{page:05d}/region_{number:04d}_context.png'
                book_save_crop(directory, crop_name, crop, generated)
                book_save_crop(directory, context_name, context, generated)
                region.update(crop=crop_name, context_crop=context_name, source_pixel_rectangle=list(rectangle))
                caption = ('Ngữ cảnh dòng; chưa định vị được chữ bất đồng.' if location.get('granularity') == 'line_context'
                           else 'Khung chữ/cụm chữ do PP-OCR ước lượng; các model khác được ghép bằng căn chỉnh chuỗi.')
                body += '<p class="muted">' + caption + '</p><div class="images">'
                for filename, picture, label in ((crop_name, crop, 'Ảnh cắt'), (context_name, context, 'Ngữ cảnh · khung đỏ')):
                    body += f'<figure><img loading="lazy" src="../{filename}" width="{picture.width}" height="{picture.height}" alt="{label}"><figcaption>{label}</figcaption></figure>'
                body += '</div>'
                crop_count += 1
                crop.close()
                context.close()
            else:
                label = ('Xuất ảnh đã tắt.' if not include_crops else
                         'Chưa xác minh/truy cập được PDF nguồn.' if not row['source_verified'] or not row['source'] or not Path(row['source']).is_file()
                         else 'Chưa đủ tọa độ hoặc căn chỉnh để cắt ảnh vùng này.')
                body += f'<p class="muted">{label}</p>'
            body += '</article>'
            regions.append(region)
        if not consensus['conflicts']:
            body += '<p>Không ghi nhận cột bất đồng trong các output tham gia so sánh. Kiểm tra trạng thái model thiếu, không bỏ phiếu hoặc chưa căn chỉnh được phía trên.</p>'
        if image is not None:
            image.close()
        body += '</section>'
        book_write_text(directory, f'pages/p{page:05d}.html', book_html(f'Trang {page} · {book}', body, page=True), generated)
        comparisons.append(json.dumps({'page': page, 'status': consensus['status'], 'source_verified': row['source_verified'],
            'reasons': row['reasons'], 'missing_models': consensus['missing_models'], 'reference': consensus['reference'],
            'excluded_models': consensus.get('excluded_models', {}), 'reading_order': row.get('reading_order', {}),
            'page_consensus': diagnostic,
            **({'evidence_schema': result['evidence_schema'], 'spatial_coverage': spatial,
                'variant_stability': stability, 'variant_consensus': row.get('variant_consensus')} if result.get('evidence_schema') else {}),
            'alignment': consensus.get('alignment'), 'line_alignment': consensus.get('line_alignment', {}),
            'pairwise_distances': consensus['pairwise_distances'], 'conflicts': consensus['conflicts'],
            'regions': regions}, ensure_ascii=False))
        cells = ''.join(f'<td><span class="badge{ " bad" if not records[model] or model in consensus.get("excluded_models", {}) else ""}">{"Không bỏ phiếu" if model in consensus.get("excluded_models", {}) else "OK" if records[model] else "Thiếu"}</span></td>' for model in models)
        structural = page_consensus_summary(row.get('page_consensus') or {})
        index_rows.append(f'<tr data-page="{page}"><td><a href="pages/p{page:05d}.html">Trang {page}</a></td>{cells}<td>{len(consensus["conflicts"])}</td><td>{report_escape(structural)}</td></tr>')
    book_write_text(directory, 'comparison.jsonl', '\n'.join(comparisons) + '\n', generated)
    counts = {model: sum((book, row['page'] - 1) in runs[model]['records'] for row in rows) for model in models}
    complete = all(count == len(rows) for count in counts.values())
    quality_pages = sum(bool(row['consensus'].get('excluded_models')) for row in rows)
    body = f'<header><h1>{report_escape(book)}</h1><p class="status">{ "Đủ output các trang đã chọn" if complete else "Kết quả còn thiếu output"} · {len(rows)} trang</p>'
    body += '<p>Chọn trang để đọc bốn dự đoán và xem ảnh bất đồng. Số trang theo PDF, bắt đầu từ 1.</p>'
    body += '<p class="muted">Số cột bất đồng là kết quả căn chỉnh chuỗi, không phải số chữ OCR sai đã được xác nhận.</p>'
    if quality_pages:
        body += f'<p class="notice">{quality_pages} trang có output cần kiểm tra và không tham gia bỏ phiếu; xem từng trang để biết model và lý do.</p>'
    body += '<p class="muted">Giữ nguyên thư mục sau khi giải nén. Mở được offline, không cần server.</p></header>'
    all_warnings = sorted(set(result.get('warnings', []) + warnings))
    if all_warnings:
        body += '<section class="notice"><h2>Lưu ý dữ liệu</h2><ul>' + ''.join('<li>' + report_escape(value) + '</li>' for value in all_warnings) + '</ul></section>'
    body += '<section><h2>Các trang</h2><label>Lọc trang / trạng thái <input id="page-filter" type="search" placeholder="Ví dụ: Trang 12 hoặc Thiếu"></label>'
    body += '<div class="scroll"><table><thead><tr><th>Trang</th>' + ''.join(f'<th>{report_escape(model)}</th>' for model in models) + '<th>Bất đồng</th><th>Dấu hiệu cấp trang</th></tr></thead><tbody>'
    body += ''.join(index_rows) + '</tbody></table></div></section>'
    body += '<section><h2>Dữ liệu và nhật ký</h2><ul>' + ''.join(f'<li><a href="raw/{model}.jsonl">{model}.jsonl</a> · {counts[model]}/{len(rows)} trang</li>' for model in models)
    body += '<li><a href="comparison.jsonl">Bất đồng và tọa độ</a></li><li><a href="errors.jsonl">Lịch sử lỗi</a></li><li><a href="manifest.json">Thông tin nguồn và cấu hình</a></li></ul></section>'
    book_write_text(directory, 'index.html', book_html(book, body), generated)
    generated.add('manifest.json')
    manifest = {'format': BOOK_OUTPUT_FORMAT, 'book': book, 'created_at': now(),
        'source': rows[0]['source'], 'source_verified': all(row['source_verified'] for row in rows),
        'page_numbers': [row['page'] for row in rows], 'status': 'complete' if complete else 'partial',
        'page_number_base': 1, 'raw_page_index_base': 0, 'alignment_column_index_base': 0,
        'models': {model: {'done': counts[model], 'selected': len(rows), 'config': runs[model]['manifest']['config'],
                    'source': runs[model]['manifest'].get('books', {}).get(book)} for model in models},
        'warnings': all_warnings, 'crop_regions': crop_count, 'pages_rendered_for_crops': render_count,
        'page_structure_overviews': structural_overviews,
        'quality_review_pages': quality_pages, 'comparison_settings': result.get('settings', {}),
        **({'evidence_schema': result['evidence_schema'], 'selected_variants': result.get('selected_variants'),
            'spatial_summary': result.get('spatial_summary'), 'stability_summary': result.get('stability_summary'),
            'variant_manifests': {v: r['manifests'][book] for v, r in result.get('variant_results', {}).items()}}
           if result.get('evidence_schema') else {}),
        'crop_dpi': dpi, 'generated_files': sorted(generated),
        'export_seconds': round(time.monotonic() - started, 3)}
    # Remove only stale files listed by this exporter, after writing replacements.
    for relative in set(previous.get('generated_files', [])) - generated:
        stale = book_file(directory, relative)
        if stale.is_file():
            stale.unlink()
    write_json(previous_path, manifest)
    archive = zip_ocr_book(directory, directory.parent / f'{directory.name}_results.zip') if create_zip else None
    return {'book': book, 'directory': str(directory), 'index_html': str(directory / 'index.html'),
            'zip_path': str(archive) if archive else None, 'pages': len(rows), 'status': manifest['status'],
            'crop_regions': crop_count, 'pages_rendered_for_crops': render_count,
            'export_seconds': round(time.monotonic() - started, 3)}


def export_corpus_results(result, output_dir, *, include_crops=True, create_zip=True, dpi=250, renderer=None):
    books = list(dict.fromkeys(row['book'] for row in result['rows']))
    names = [book_output_slug(book) for book in books]
    if len({name.casefold() for name in names}) != len(names):
        names = [f'{name}_{hashlib.sha256(book.encode()).hexdigest()[:8]}' for name, book in zip(names, books)]
    return [export_ocr_book(result, book, Path(output_dir) / name, include_crops=include_crops,
                           create_zip=create_zip, dpi=dpi, renderer=renderer) for book, name in zip(books, names)]


def read_jsonl_campaign(path):
    return json.loads(Path(path).read_text(encoding='utf8'))['pairs']
