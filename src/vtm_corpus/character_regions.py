"""Map consensus positions to genuine character boxes; render source-page crops."""
from __future__ import annotations

from math import ceil, floor, isfinite

from .text_alignment import comparison_text


def normalized_character_box(entry: dict, width: int, height: int):
    """Validate a model-estimated character rectangle/polygon, in input pixels."""
    if entry.get("granularity") != "character" or len(entry.get("text", "")) != 1 or width <= 0 or height <= 0:
        return None
    return normalized_polygon(entry.get("polygon"), width, height)


def normalized_polygon(points, width: int, height: int):
    """Normalize an actual OCR polygon without inferring character boundaries."""
    if width <= 0 or height <= 0:
        return None
    if not isinstance(points, (list, tuple)):
        return None
    if len(points) == 4 and all(isinstance(x, (int, float)) for x in points):
        x1, y1, x2, y2 = points
        points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    if len(points) < 4 or any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in points):
        return None
    if any(not isinstance(v, (int, float)) or not isfinite(v) for point in points for v in point):
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    if min(xs) < 0 or min(ys) < 0 or max(xs) > width or max(ys) > height or min(xs) >= max(xs) or min(ys) >= max(ys):
        return None
    return {"bbox": [min(xs) / width, min(ys) / height, max(xs) / width, max(ys) / height],
            "polygon": [[x / width, y / height] for x, y in points]}


def locator_character_boxes(consensus: dict, records: dict, locator="ppocr"):
    record = records.get(locator)
    lookup = {}
    if locator in consensus.get("excluded_models", {}):
        return lookup
    if record and record.get("geometry", record["raw"]).get("coordinate_frame") == "input_image":
        raw = record.get("geometry", record["raw"])
        for entry in raw.get("character_boxes", []):
            offset = entry.get("offset")
            if not isinstance(offset, int) or offset < 0:
                continue
            text = consensus["texts"].get(locator) or ""
            if text[offset:offset + 1] != entry.get("text"):
                continue
            box = normalized_character_box(entry, raw.get("w", 0), raw.get("h", 0))
            if box:
                if offset in lookup:
                    lookup[offset] = None  # ambiguous duplicate coordinates
                else:
                    lookup[offset] = {**box, "model": locator, "method": entry.get("method"),
                                      "estimated": True, "offset": offset}
    return lookup


def attach_character_regions(consensus: dict, records: dict, *, locator="ppocr") -> None:
    """Use one locator model's coordinates; never fabricate boxes for its gaps.

    The votes from other models are linked by string alignment, not independent
    detections of that glyph. A box is still a model estimate, not a human label.
    """
    lookup = locator_character_boxes(consensus, records, locator)
    def stable(column):
        values = [v for v in column["predictions"].values() if v is not None]
        return len(values) >= 2 and len(set(values)) == 1 and all(values)

    def unique_anchor(value):
        if not value:
            return False
        for model, text in consensus["texts"].items():
            if text is None or model in consensus.get("excluded_models", {}):
                continue
            first = text.find(value)
            if first < 0 or text.find(value, first + 1) >= 0:
                return False
        return True

    columns = consensus["columns"]
    for index, column in enumerate(columns):
        offset = column["offsets"].get(locator)
        location = lookup.get(offset)
        # Only show a conflict crop when both sides have unique shared anchors
        # (or a page edge). Repeated/adjacent disputed runs remain unresolved.
        if location and column["disagreement"]:
            before, after = "", ""
            for neighbor in reversed(columns[max(0, index - 8):index]):
                if not stable(neighbor):
                    break
                before = neighbor["predictions"][locator] + before
            for neighbor in columns[index + 1:index + 9]:
                if not stable(neighbor):
                    break
                after += neighbor["predictions"][locator]
            supported = (index == 0 or unique_anchor(before)) and (index == len(columns) - 1 or unique_anchor(after))
            if not supported:
                location = None
        column["localization"] = location
        column["localization_status"] = "model_estimated_character" if location else "unresolved"


def span_has_shared_anchors(consensus, start, stop, locator):
    """Validate a disputed run as a whole, including adjacent disagreements."""
    columns = consensus["columns"]
    def stable(column):
        values = [v for v in column["predictions"].values() if v is not None]
        return len(values) >= 2 and len(set(values)) == 1 and all(values)
    def unique(value):
        if not value:
            return False
        for model, text in consensus["texts"].items():
            if text is None or model in consensus.get("excluded_models", {}):
                continue
            position = text.find(value)
            if position < 0 or text.find(value, position + 1) >= 0:
                return False
        return True
    before, after = "", ""
    for column in reversed(columns[max(0, start - 8):start]):
        if not stable(column):
            break
        before = (column["predictions"].get(locator) or "") + before
    for column in columns[stop:stop + 8]:
        if not stable(column):
            break
        after += column["predictions"].get(locator) or ""
    return (start == 0 or unique(before)) and (stop == len(columns) or unique(after))


def union_region(locations, *, locator, method, granularity):
    x1 = min(location["bbox"][0] for location in locations)
    y1 = min(location["bbox"][1] for location in locations)
    x2 = max(location["bbox"][2] for location in locations)
    y2 = max(location["bbox"][3] for location in locations)
    return {"bbox": [x1, y1, x2, y2], "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "model": locator, "method": method, "granularity": granularity, "estimated": True}


def attach_conflict_regions(consensus: dict, records: dict, *, locator="ppocr") -> None:
    """Group disputed columns and attach measured character or line-context boxes.

    A line crop is context only, including when the locator omitted a character.
    It never upgrades the individual column's unresolved character coordinates.
    """
    columns, groups = consensus["columns"], []
    for column in columns:
        if not column["disagreement"]:
            continue
        if groups and groups[-1][-1]["index"] + 1 == column["index"]:
            groups[-1].append(column)
        else:
            groups.append([column])
    lookup = locator_character_boxes(consensus, records, locator)
    record = records.get(locator) or {}
    raw = record.get("geometry", record.get("raw", {})) if locator not in consensus.get("excluded_models", {}) else {}
    lines, offset = [], 0
    text_lines = raw.get("text_lines", [])
    if (raw.get("coordinate_frame") == "input_image" and
            "".join(comparison_text(line.get("text", "")) for line in text_lines) == consensus["texts"].get(locator)):
        for line in text_lines:
            length = len(comparison_text(line.get("text", "")))
            box = normalized_polygon(line.get("polygon"), raw.get("w", 0), raw.get("h", 0))
            if length and box:
                lines.append({**box, "start": offset, "stop": offset + length})
            offset += length
    regions = []
    for number, group in enumerate(groups, 1):
        start, stop = group[0]["index"], group[-1]["index"] + 1
        offsets = [column["offsets"].get(locator) for column in group]
        present = [value for value in offsets if value is not None]
        region = {"number": number, "start": start, "stop": stop, "columns": [c["index"] for c in group],
                  "predictions": {name: None if text is None or name in consensus.get("excluded_models", {}) else "".join(c["predictions"][name] or "" for c in group)
                                  for name, text in consensus["texts"].items()},
                  "localization": None, "localization_status": "unresolved"}
        if span_has_shared_anchors(consensus, start, stop, locator):
            if len(present) == len(offsets) and all(lookup.get(value) for value in present):
                locations = [lookup[value] for value in present]
                location = (dict(locations[0], granularity="character") if len(locations) == 1 else
                    union_region(locations, locator=locator, method="ppocr_character_box_union", granularity="character_group"))
                location["source_offsets"] = present
                region.update(localization=location, localization_status="model_estimated_character_group")
            else:
                if present:
                    selected = [line for line in lines if line["start"] <= max(present) and line["stop"] > min(present)]
                    # Every locator offset must actually belong to a valid line.
                    if any(not any(line["start"] <= value < line["stop"] for line in selected) for value in present):
                        selected = []
                else:
                    anchors = [columns[i]["offsets"].get(locator) for i in (start - 1, stop)
                               if 0 <= i < len(columns)]
                    anchors = [value for value in anchors if value is not None]
                    selected = [line for line in lines if anchors and
                                all(line["start"] <= value < line["stop"] for value in anchors)]
                if selected:
                    region.update(localization=union_region(selected, locator=locator,
                        method="ppocr_line_polygon", granularity="line_context"),
                        localization_status="model_estimated_line_context")
        regions.append(region)
    consensus["conflict_regions"] = regions


def character_crops(image, location: dict, *, context_chars=3):
    """Crop original raster pixels, plus a context view highlighting the polygon."""
    from PIL import ImageDraw
    width, height = image.size
    x1, y1, x2, y2 = location["bbox"]
    rectangle = (max(0, floor(x1 * width)), max(0, floor(y1 * height)),
                 min(width, ceil(x2 * width)), min(height, ceil(y2 * height)))
    if rectangle[2] <= rectangle[0] or rectangle[3] <= rectangle[1]:
        raise ValueError("Invalid character crop")
    glyph = image.crop(rectangle)
    unit = min(glyph.size) if location.get("granularity") in {"character_group", "line_context"} else max(glyph.size)
    margin = unit * context_chars
    context_rect = (max(0, rectangle[0] - margin), max(0, rectangle[1] - margin),
                    min(width, rectangle[2] + margin), min(height, rectangle[3] + margin))
    context = image.crop(context_rect)
    polygon = [(x * width - context_rect[0], y * height - context_rect[1]) for x, y in location["polygon"]]
    draw = ImageDraw.Draw(context)
    draw.line(polygon + [polygon[0]], fill="red", width=max(1, round(max(image.size) / 1200)))
    return glyph, context, rectangle
