"""Lossless, checksum-verified offline source packing below Kaggle's 1 MB limit."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "1_ocr_va_so_sanh.ipynb"
NOTEBOOKS = (NOTEBOOK, ROOT / '1a_ocr_anh_goc.ipynb')
MAX_BYTES = 1_000_000
PACKED_CELL_IDS = {'definitions','offline-setup','a8dd2655da73'}


def unpack_runtime_source(source):
    """Decode authoring input without executing any notebook code."""
    if '_VTM_PACKED_CODE =' not in source:return source
    assignments={n.targets[0].id:n.value for n in ast.parse(source).body if isinstance(n,ast.Assign)
                 and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    if '_VTM_PACKED_CODE' not in assignments:return source
    raw=zlib.decompress(base64.b85decode(ast.literal_eval(assignments['_VTM_PACKED_CODE'])))
    expected=ast.literal_eval(assignments['_VTM_CODE_SHA256'])
    if hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError('Embedded runtime source checksum mismatch')
    decoded=raw.decode('utf8');compile(decoded,'<unpacked-notebook-source>','exec')
    return decoded


def pack_runtime_source(source, identity):
    source=unpack_runtime_source(source)
    compile(source,identity,'exec')
    raw=source.encode('utf8');packed=base64.b85encode(zlib.compress(raw,9)).decode('ascii')
    lines=['# Mã nội bộ nén lossless; kiểm SHA-256 trước khi nạp. Chỉnh cấu hình ở ô đầu.',
           'import base64 as _vtm_b64, hashlib as _vtm_hash, zlib as _vtm_zlib',
           '_VTM_CODE_SHA256 = '+repr(hashlib.sha256(raw).hexdigest()),'_VTM_PACKED_CODE = (',
           *('    '+repr(packed[i:i+160]) for i in range(0,len(packed),160)),')',
           '_vtm_decoded = _vtm_zlib.decompress(_vtm_b64.b85decode(_VTM_PACKED_CODE))',
           'if _vtm_hash.sha256(_vtm_decoded).hexdigest() != _VTM_CODE_SHA256:',
           "    raise ValueError('Embedded runtime source checksum mismatch')",
           'exec(compile(_vtm_decoded.decode("utf8"), '+repr('<vtm:'+identity+'>')+', "exec"), globals())',
           'del _vtm_decoded, _VTM_PACKED_CODE, _VTM_CODE_SHA256']
    result='\n'.join(lines)+'\n'
    if unpack_runtime_source(result)!=source:raise ValueError('Runtime source round trip mismatch')
    return result


def unpack_notebook(notebook):
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            source=''.join(cell['source']);decoded=unpack_runtime_source(source)
            if decoded!=source:cell['source']=decoded.splitlines(keepends=True)
    return notebook


def pack_notebook(notebook):
    for cell in notebook['cells']:
        if cell['cell_type']=='code' and cell.get('id') in PACKED_CELL_IDS:
            source=''.join(cell['source'])
            if len(source.encode('utf8'))<2000:continue
            cell['source']=pack_runtime_source(source,cell['id']).splitlines(keepends=True)
            cell['metadata'].setdefault('jupyter',{})['source_hidden']=True
    return notebook


def notebook_bytes(notebook):
    pack_notebook(notebook)
    # Compact JSON changes the file representation, not the visible cell text.
    output=(json.dumps(notebook,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf8')
    if len(output)>=MAX_BYTES:
        raise ValueError(f'Notebook is {len(output):,} bytes; must be below {MAX_BYTES:,}')
    return output


def compact_source(source):
    assignments = {
        node.targets[0].id: node
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    node = assignments["VENDOR_SOURCES"]
    if not isinstance(node.value, ast.Dict):
        # An already packed cell stays byte-identical; still verify its payload.
        packed = ast.literal_eval(assignments["_VENDOR_SOURCES_B85"].value)
        expected = ast.literal_eval(assignments["_VENDOR_SOURCES_SHA256"].value)
        raw = zlib.decompress(base64.b85decode(packed))
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Embedded upstream source checksum mismatch")
        json.loads(raw)
        return source

    original = ast.literal_eval(node.value)
    raw = json.dumps(original, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    packed = base64.b85encode(zlib.compress(raw, 9)).decode("ascii")
    if json.loads(zlib.decompress(base64.b85decode(packed))) != original:
        raise ValueError("Upstream source round trip failed")
    lines = [
        "# Lossless upstream files, including licenses; standard library only.",
        "import base64 as _vendor_base64",
        "import hashlib as _vendor_hashlib",
        "import json as _vendor_json",
        "import zlib as _vendor_zlib",
        f"_VENDOR_SOURCES_SHA256 = {hashlib.sha256(raw).hexdigest()!r}",
        "_VENDOR_SOURCES_B85 = (",
        *(f"    {packed[i:i+120]!r}" for i in range(0, len(packed), 120)),
        ")",
        "_vendor_raw = _vendor_zlib.decompress(_vendor_base64.b85decode(_VENDOR_SOURCES_B85))",
        "if _vendor_hashlib.sha256(_vendor_raw).hexdigest() != _VENDOR_SOURCES_SHA256:",
        "    raise ValueError('Embedded upstream source checksum mismatch')",
        "VENDOR_SOURCES = _vendor_json.loads(_vendor_raw)",
        "del _VENDOR_SOURCES_B85, _vendor_raw",
    ]
    result = source.replace(ast.get_source_segment(source, node), "\n".join(lines), 1)
    compile(result, "<upstream-data>", "exec")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="*", type=Path, help='Defaults to both OCR notebooks')
    parser.add_argument("--check", action="store_true", help="Verify packing and size without writing")
    args = parser.parse_args()
    for path in args.notebook or NOTEBOOKS:
        before=path.read_bytes();notebook=json.loads(before)
        cells=[c for c in notebook['cells'] if c['cell_type']=='code' and 'VENDOR_SOURCES = ' in ''.join(c['source'])]
        if len(cells)!=1:raise ValueError('Expected exactly one embedded upstream data cell')
        source=''.join(cells[0]['source']);compacted=compact_source(source)
        if args.check:
            if compacted!=source:raise ValueError('Upstream data is not packed')
            unpack_notebook(notebook)  # Check runtime checksums and decoded compilation, without executing.
            output=before
            if len(output)>=MAX_BYTES:raise ValueError(f'{path.name}: {len(output):,} bytes; must be below {MAX_BYTES:,}')
        else:
            cells[0]['source']=compacted.splitlines(keepends=True)
            output=notebook_bytes(notebook)
            if output!=before:path.write_bytes(output)
        print(f'{path.name}: {len(before):,} -> {len(output):,} bytes; source checksums verified')


if __name__ == "__main__":
    main()
