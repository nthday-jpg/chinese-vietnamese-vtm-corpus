"""Offline HTML presentation and portable image references for optional evidence."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from html import escape
import json
from pathlib import Path


EVIDENCE_JS = '''
document.querySelectorAll('[data-evidence-filter]').forEach(input=>input.addEventListener('input',()=>{
 const filters=[...document.querySelectorAll('[data-evidence-filter]')];
 document.querySelectorAll('[data-evidence-row]').forEach(row=>{
  row.hidden=filters.some(f=>f.value&&!String(row.dataset[f.dataset.evidenceFilter]||'').toLowerCase().includes(f.value.toLowerCase()));
 });
}));
'''


def portable_evidence(value, root, page, generated):
    value = deepcopy(value)
    root = Path(root).resolve()
    visited = set()
    def visit(item):
        if isinstance(item, (dict, list)):
            if id(item) in visited:
                return
            visited.add(id(item))
        if isinstance(item, dict):
            if {'path', 'sha256', 'size'} <= item.keys():
                source = Path(item['path'])
                data = source.read_bytes()
                if hashlib.sha256(data).hexdigest() != item['sha256']:
                    raise ValueError('Evidence image changed since analysis')
                relative = f'evidence/p{page:05d}/{item["sha256"]}.png'
                target = root/relative
                if not target.resolve().is_relative_to(root):
                    raise ValueError('Evidence output escapes book directory')
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix('.tmp')
                temporary.write_bytes(data); temporary.replace(target)
                generated.add(relative)
                item['path'] = relative
            else:
                for v in item.values(): visit(v)
        elif isinstance(item, list):
            for v in item: visit(v)
    visit(value)
    return value


def evidence_picture(ref, caption):
    return ('<figure><img loading="lazy" src="../'+escape(ref['path'], quote=True)+'" alt="'+escape(caption, quote=True)+
            '"><figcaption>'+escape(caption)+'</figcaption></figure>')


def book_spatial_coverage(diagnostic):
    if not diagnostic:
        return ''
    html = '<section id="spatial-coverage"><h2>Độ phủ không gian độc lập</h2>'
    html += '<p>Ứng viên xanh dương: phân tích ảnh gốc; OCR xanh ngọc: khung ước lượng; đỏ: nghi chưa có bằng chứng định vị; cam: chưa chắc chắn.</p>'
    html += '<p>Overlap đo phần mực ứng viên nằm trong khung OCR. Không phải tỷ lệ chữ đọc đúng hay đọc đủ; khung dòng/khối chỉ là ngữ cảnh.</p>'
    html += '<p>'+escape(diagnostic['status']+': '+diagnostic.get('reason', ''))+'</p>'
    html += '<pre>'+escape(json.dumps(diagnostic.get('summary', {}), ensure_ascii=False, indent=2))+'</pre>'
    html += '<p><a href="#page-consensus">Đồng thuận cấp trang</a> · <a href="#character-consensus">Bất đồng ký tự</a></p><div class="images">'
    for name, ref in diagnostic.get('artifacts', {}).items():
        html += evidence_picture(ref, name)
    html += '</div><div class="scroll"><table><thead><tr><th>Vùng nguồn</th><th>Phân loại</th><th>Overlap</th><th>Lý do</th></tr></thead><tbody>'
    for c in diagnostic.get('findings', []):
        html += '<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in (c['id']+' '+str(c['bbox']), c['kind'],
            c['candidate_ink_overlap_ratio'], '; '.join(c['uncertainty'])))+'</tr>'
    html += '</tbody></table></div>'
    for c in diagnostic.get('findings', []):
        if c.get('detail_crop'):
            html += '<details><summary>'+escape(c['id']+' · '+c['kind'])+'</summary>'
            html += evidence_picture(c['detail_crop'], 'Pixel nguồn; chưa xác nhận đây là chữ bị bỏ sót')+'</details>'
    return html+'</section>'


def book_variant_stability(diagnostic):
    if not diagnostic:
        return ''
    html = '<section id="variant-stability"><h2>Độ ổn định qua biến thể ảnh</h2>'
    html += '<p>So sánh trong từng engine; mỗi biến thể có đồng thuận bốn engine riêng. Biến thể không tăng số phiếu độc lập. Chữ gốc và mọi phương án được giữ nguyên.</p>'
    html += '<pre>'+escape(json.dumps(diagnostic['summary'], ensure_ascii=False, indent=2))+'</pre>'
    html += '<div class="scroll"><table><thead><tr><th>Biến thể</th>'
    engines = list(next(iter(diagnostic['matrix'].values())))
    html += ''.join('<th>'+escape(e)+'</th>' for e in engines)+'</tr></thead><tbody>'
    for variant, records in diagnostic['matrix'].items():
        html += '<tr><th>'+escape(variant)+'</th>'
        for e, rec in records.items():
            dims = [(r.get('variant_size'), r.get('input_size'), r.get('processor_size')) for r in rec['regions']]
            html += '<td><details><summary>'+escape(rec['status'])+'</summary><pre>'+escape(rec['text'] or '')+'</pre><p>'+escape(str(dims))+'</p><p>'+escape(str(rec.get('input_status') or rec['processor_observation']))+'</p></details></td>'
        html += '</tr>'
    html += '</tbody></table></div><p>Kích thước: prepared → worker → processor; null nghĩa là chưa quan sát được. Khác ngân sách pixel là yếu tố gây nhiễu phép so.</p>'
    html += '<p>Lọc engine <input data-evidence-filter="engine"> biến thể <input data-evidence-filter="variant"> vùng <input data-evidence-filter="region"> loại <input data-evidence-filter="kind"></p>'
    for pair in diagnostic['comparisons']:
        for c in [pair, *pair['regions']]:
            attrs = ' '.join('data-'+k+'="'+escape(str(v), quote=True)+'"' for k, v in (
                ('engine', pair['engine']), ('variant', pair['variant']), ('region', c['region_id']),
                ('kind', ' '.join([c['status'], *[change['kind'] for change in c['changes']]]))))
            html += '<details data-evidence-row '+attrs+'><summary>'+escape(
                f'{pair["engine"]} · {pair["variant"]} · {c["region_id"]} · {c["status"]}')+'</summary>'
            html += '<p>Tham chiếu: '+escape(pair['reference_variant'])+'</p><div class="predictions"><pre>'+escape(c['reference_text'] or '')+'</pre><pre>'+escape(c['variant_text'] or '')+'</pre></div>'
            html += '<pre>'+escape(json.dumps({k: c.get(k) for k in ('normalized_edit_distance', 'substitutions', 'insertions', 'deletions', 'changes', 'uncertainty')}, ensure_ascii=False, indent=2))+'</pre>'
            if c.get('image_comparison'):
                images = c['image_comparison']
                html += '<p>Ngữ cảnh crop canonical; không suy ra vị trí glyph từ offset chuỗi.</p><div class="images">'
                html += evidence_picture(images['original'], 'Ảnh nguồn')+evidence_picture(images['variant'], pair['variant'])+'</div>'
            html += '</details>'
    html += '<details><summary>Các dấu hiệu cần rà soát</summary><pre>'+escape(json.dumps(diagnostic['risks'], ensure_ascii=False, indent=2))+'</pre></details>'
    return html+'</section>'
