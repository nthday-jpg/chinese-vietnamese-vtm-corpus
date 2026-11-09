"""Order detected vertical columns without changing glyphs or source geometry."""
from __future__ import annotations

from bisect import bisect_right
from math import isfinite
from statistics import median

from .text_alignment import comparison_text


def line_rectangle(polygon):
    if not isinstance(polygon, (list, tuple)):
        return None
    if len(polygon) == 4 and all(isinstance(v, (int, float)) for v in polygon):
        x1, y1, x2, y2 = polygon
    else:
        if len(polygon) < 4 or any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in polygon):
            return None
        if any(not isinstance(v, (int, float)) or not isfinite(v) for p in polygon for v in p):
            return None
        x1, y1 = min(p[0] for p in polygon), min(p[1] for p in polygon)
        x2, y2 = max(p[0] for p in polygon), max(p[1] for p in polygon)
    if not all(isfinite(v) for v in (x1, y1, x2, y2)) or x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def order_ppocr_columns(raw, *, mode="auto"):
    """Return comparison geometry; the original checkpoint remains untouched.

    Auto requires at least two tall detections and 70% of detected characters
    in tall detections. Unknown/missing geometry keeps detection order. Within
    a shared column, fragments are read top to bottom. Every character offset
    is moved with its original box, even when the same glyph occurs repeatedly.
    """
    if mode not in {"auto", "vertical_rtl", "detection"}:
        raise ValueError("reading_order must be auto, vertical_rtl or detection")
    info = {"mode": "detection", "changed": False}
    lines = raw.get("text_lines", [])
    if mode == "detection" or not lines:
        return raw, info
    texts = [comparison_text(line.get("text", "")) for line in lines]
    if "".join(texts) != comparison_text(raw.get("text", "")):
        return raw, {**info, "reason": "line_text_mismatch"}
    boxes = [line_rectangle(line.get("polygon")) for line in lines]
    if any(box is None for box in boxes):
        return raw, {**info, "reason": "incomplete_line_geometry"}
    # Two-character entries in contents pages often have extra horizontal padding.
    vertical = [i for i, (x1, y1, x2, y2) in enumerate(boxes) if y2 - y1 >= 1.5 * (x2 - x1)]
    total = sum(map(len, texts))
    if mode == "auto" and (len(vertical) < 2 or not total or
                           sum(len(texts[i]) for i in vertical) < .7 * total):
        return raw, info
    tolerance = .45 * median(box[2] - box[0] for box in boxes)
    groups = []
    for index in sorted(range(len(lines)), key=lambda i: -(boxes[i][0] + boxes[i][2]) / 2):
        center = (boxes[index][0] + boxes[index][2]) / 2
        if groups and abs(groups[-1][0] - center) <= tolerance:
            groups[-1][1].append(index)
        else:
            groups.append((center, [index]))
    order = [index for _, group in groups for index in sorted(group, key=lambda i: boxes[i][1])]
    starts, offset = [], 0
    for text in texts:
        starts.append(offset)
        offset += len(text)
    new_starts, offset = {}, 0
    for index in order:
        new_starts[index] = offset
        offset += len(texts[index])
    characters = []
    for entry in raw.get("character_boxes", []):
        old = entry.get("offset")
        if not isinstance(old, int) or old < 0 or old >= total:
            continue
        index = bisect_right(starts, old) - 1
        local = old - starts[index]
        if texts[index][local:local + 1] == entry.get("text"):
            characters.append({**entry, "offset": new_starts[index] + local})
    geometry = {**raw, "text": "\n".join(lines[i]["text"] for i in order),
                "text_lines": [lines[i] for i in order],
                "character_boxes": sorted(characters, key=lambda entry: entry["offset"])}
    return geometry, {"mode": "vertical_rtl", "changed": order != list(range(len(lines))), "line_order": order}
