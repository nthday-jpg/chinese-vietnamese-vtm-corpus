"""Losslessly pack notebook 1's upstream source data for offline Kaggle use."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "1_chuan_bi_anh_ocr.ipynb"
MAX_BYTES = 1_000_000


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
    parser.add_argument("notebook", nargs="?", type=Path, default=NOTEBOOK)
    parser.add_argument("--check", action="store_true", help="Verify packing and size without writing")
    args = parser.parse_args()
    before = args.notebook.read_bytes()
    notebook = json.loads(before)
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"
             and "VENDOR_SOURCES = " in "".join(c["source"])]
    if len(cells) != 1:
        raise ValueError("Expected exactly one embedded upstream data cell")
    source = "".join(cells[0]["source"])
    compacted = compact_source(source)
    if args.check:
        if compacted != source:
            raise ValueError("Upstream source data is not packed; run without --check first")
        output = before
    elif compacted == source:
        output = before
    else:
        cells[0]["source"] = compacted.splitlines(keepends=True)
        output = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    if len(output) >= MAX_BYTES:
        raise ValueError(f"Notebook is {len(output):,} bytes; must be below {MAX_BYTES:,}")
    if not args.check and output != before:
        args.notebook.write_bytes(output)
    print(f"{args.notebook.name}: {len(before):,} -> {len(output):,} bytes; upstream checksum verified")


if __name__ == "__main__":
    main()
