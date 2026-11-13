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
document.querySelectorAll('[data-restoration-review]').forEach(section=>{
 const selectors=[...section.querySelectorAll('[data-review-case]')];
 section.querySelector('[data-review-save]').onclick=()=>{
  const labels=selectors.filter(s=>s.value).map(s=>({case_id:s.dataset.reviewCase,label:s.value,
   note:section.querySelector('[data-review-note="'+s.dataset.reviewCase+'"]').value}));
  const payload={schema:'vtm-restoration-review-v1',labels,
   sampling:JSON.parse(section.dataset.sampling),exported_cases:selectors.length};
  const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));
  const a=document.createElement('a');a.href=url;a.download='restoration_labels_p'+section.dataset.page+'.json';a.click();URL.revokeObjectURL(url);
 };
 section.querySelector('[data-review-load]').onchange=async event=>{
  try {
   const payload=JSON.parse(await event.target.files[0].text());
   if(payload.schema!=='vtm-restoration-review-v1'||!Array.isArray(payload.labels))throw Error('Sai định dạng');
   const seen=new Set();
   for(const row of payload.labels){
    const s=selectors.find(s=>s.dataset.reviewCase===row.case_id);
    if(!s||seen.has(row.case_id)||!['rescued','regressed','unchanged','uncertain'].includes(row.label))throw Error('Nhãn hoặc evidence identity không khớp trang');
    seen.add(row.case_id);
   }
   for(const row of payload.labels){
    selectors.find(s=>s.dataset.reviewCase===row.case_id).value=row.label;
    section.querySelector('[data-review-note="'+row.case_id+'"]').value=row.note||'';
   }
   section.querySelector('[data-review-message]').textContent='Đã đọc nhãn. Hãy tải JSON để lưu ra đĩa.';
  }catch(error){section.querySelector('[data-review-message]').textContent=error.message;}
 };
});
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
    if diagnostic.get('groups'):
        html += '<p>Tỷ lệ tách theo cơ chế và từng engine. Nhóm sensitivity có thể đổi glyph; không là độ bất biến bảo toàn nội dung.</p>'
        html += '<pre>'+escape(json.dumps(diagnostic['groups'], ensure_ascii=False, indent=2))+'</pre>'
        html += '<details><summary>Thống kê cùng vùng canonical</summary><pre>'+escape(json.dumps(
            diagnostic.get('region_groups', {}), ensure_ascii=False, indent=2))+'</pre></details>'
        html += '<details><summary>Hai trục đồng thuận và ổn định; chưa biết đúng/sai</summary><pre>'+escape(json.dumps(
            diagnostic.get('consensus_stability_axes', []), ensure_ascii=False, indent=2))+'</pre></details>'
        html += '<details><summary>Nhiều engine cùng đổi cách đọc</summary><pre>'+escape(json.dumps(
            diagnostic.get('joint_changes', []), ensure_ascii=False, indent=2))+'</pre></details>'
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


def book_restoration_evaluation(diagnostic, page):
    if not diagnostic:
        return ''
    sampling = diagnostic.get('sampling', {})
    blind = diagnostic.get('parameters', {}).get('blind_review', False)
    html = '<section id="restoration-evaluation" data-restoration-review data-page="'+str(page)+'" data-sampling="'+escape(json.dumps(sampling), quote=True)+'">'
    html += '<h2>Restoration Ablation và kiểm tra thị giác</h2><p>'+escape(diagnostic['status'])+'</p>'
    html += '<p>Cùng crop và tọa độ nguồn. Ảnh được hiển thị cùng chiều rộng; kích thước pixel thực được ghi riêng. Cảnh báo nét và agreement không xác nhận chữ đúng/sai. Nhãn do người đọc nhập, không tự lưu khi đóng trang.</p>'
    html += '<pre>'+escape(json.dumps(sampling, ensure_ascii=False, indent=2))+'</pre>'
    if diagnostic.get('reason'):
        html += '<p>'+escape(diagnostic['reason'])+'</p>'
    variants = sorted({v for r in diagnostic['regions'] for v in r['variants'] if v != 'original'},
                      key=lambda v: hashlib.sha256(v.encode()).hexdigest())
    names = {v: 'Phương án '+str(i+1) for i, v in enumerate(variants)} if blind else {v: v for v in variants}
    names['original'] = 'Ảnh nguồn'
    if blind:
        html += '<p>Đang giấu tên phương pháp trong mục H10. Các mục stability/raw vẫn công khai; chỉ xem mục này khi đánh giá mù.</p>'
    cases = {c['case_id']: c for c in diagnostic.get('review_cases', [])}
    for r in diagnostic['regions']:
        html += '<article><h3>'+escape(r['region_id']+' · '+str(r['source_box']))+'</h3><div class="images">'
        for variant, info in r['variants'].items():
            if info.get('image'):
                html += evidence_picture(info['image'], names[variant]+' · '+str(info['image']['size']))
        html += '</div>'
        for variant, info in r['variants'].items():
            if variant == 'original':
                continue
            html += '<details><summary>'+escape(names[variant]+' · '+info['status'])+'</summary>'
            html += '<p>'+escape(info.get('bicubic_status', ''))+'</p><pre>'+escape(json.dumps({
                'agreement': r['agreement'][variant], 'agreement_change': info.get('agreement_change'),
                'visual_fidelity': info.get('visual_fidelity'), 'restoration_seconds': info.get('restoration_seconds'),
                'ocr_inputs': info['readings']}, ensure_ascii=False, indent=2))+'</pre></details>'
        for pair in r['comparisons']:
            html += '<details><summary>'+escape(pair['engine']+' · '+names[pair['reference_variant']]+' → '+names[pair['variant']]+' · '+pair['status'])+'</summary>'
            data = {k: pair.get(k) for k in ('reference_text', 'variant_text', 'changes', 'substitutions',
                'insertions', 'deletions', 'uncertainty', 'confounds', 'input_status')}
            html += '<pre>'+escape(json.dumps(data, ensure_ascii=False, indent=2))+'</pre>'
            case = cases.get(pair.get('case_id'))
            if case:
                cid = case['case_id']
                html += '<label>Đánh giá của người đọc <select data-review-case="'+cid+'"><option value="">Chưa đánh giá</option>'
                html += ''.join('<option value="'+label+'">'+label+'</option>' for label in ('rescued','regressed','unchanged','uncertain'))+'</select></label>'
                html += '<label>Ghi chú <textarea data-review-note="'+cid+'"></textarea></label>'
            html += '</details>'
        html += '</article>'
    html += '<p>rescued: sửa OCR sai; regressed: làm OCR đúng thành sai; unchanged: độ đúng không đổi; uncertain: chưa đủ bằng chứng. Tập ưu tiên bất đồng không đại diện accuracy toàn corpus.</p>'
    html += '<button type="button" data-review-save>Tải nhãn JSON của trang</button> <label>Đọc lại nhãn trang <input type="file" accept=".json" data-review-load></label><p data-review-message></p>'
    return html+'</section>'
