"""Lossless region planning, bounded inputs and page-coordinate reconstruction.

Whitespace projection is a layout heuristic, not a trained semantic detector.
Analysis may normalize contrast; OCR pixels are always cropped from the original.
No overlap, text deduplication, sharpening or character-box fabrication is used.
"""
from __future__ import annotations

from math import floor, isfinite, sqrt
from pathlib import Path


def image_budget(cfg):
    # Engineering defaults, not architectural maxima. Native engines apply their
    # own block budgets, so auto does not pre-shrink their page images.
    defaults = {"hunyuan": 4_194_304, "paddle": 1_048_576, "tonggu": 1_048_576,
                "ppocr": 4_194_304, "dummy": 4_194_304}
    pixels = cfg.region_max_pixels or defaults[cfg.engine]
    if cfg.model_max_pixels and cfg.engine == "hunyuan":
        pixels = min(pixels, cfg.model_max_pixels)
    return pixels, cfg.region_max_side


def projection_gaps(image, axis, options):
    """Find internal low-ink gutters; suppress long rules in analysis only."""
    from PIL import Image, ImageChops, ImageFilter, ImageOps
    gray = image.convert("L")
    gray.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    # Local background subtraction also works on dark/uneven manuscript paper.
    radius = max(2, min(gray.size) / 80)
    mask = ImageChops.subtract(gray.filter(ImageFilter.GaussianBlur(radius)), gray)
    mask = ImageOps.autocontrast(mask).point(lambda v: 255 if v > options.get("ink_threshold", 40) else 0)
    w, h = mask.size
    data = mask.tobytes()
    # Exclude outer 3% and dense ruling rows/columns from the projection only.
    left, right = max(1, int(w * .03)), min(w - 1, int(w * .97))
    top, bottom = max(1, int(h * .03)), min(h - 1, int(h * .97))
    if left >= right or top >= bottom:
        return []
    rows = [sum(v > 0 for v in data[y*w+left:y*w+right]) for y in range(top, bottom)]
    valid_rows = [y for y, count in zip(range(top, bottom), rows) if count < .7 * (right-left)]
    if not valid_rows:
        return []
    cols = [sum(data[y*w+x] > 0 for y in valid_rows) for x in range(left, right)]
    valid_cols = [x for x, count in zip(range(left, right), cols) if count < .7 * len(valid_rows)]
    if not valid_cols:
        return []
    if axis == "x":
        values = [sum(data[y*w+x] > 0 for y in valid_rows) / len(valid_rows) for x in range(w)]
        extent, length = image.width, w
    else:
        values = [sum(data[y*w+x] > 0 for x in valid_cols) / len(valid_cols) for y in range(h)]
        extent, length = image.height, h
    # Do not let a book frame or ruled separator count as a neighboring column.
    values = [0.0 if value >= .7 else value for value in values]
    threshold = options.get("gap_ink_fraction", .006)
    minimum = max(2, round(length * options.get("min_gap_fraction", .006)))
    # Avoid margins and tiny scraps masquerading as columns.
    edge = max(minimum, round(length * .06))
    gaps, start = [], None
    for i in range(edge, length - edge + 1):
        blank = i < length - edge and values[i] <= threshold
        if blank and start is None:
            start = i
        if not blank and start is not None:
            if i - start >= minimum:
                cut = round((start + i) / 2 * extent / length)
                # Both neighbors must contain meaningful ink.
                span = max(minimum * 3, round(length * .035))
                before = values[max(edge, start-span):start]
                after = values[i:min(length-edge, i+span)]
                support = max(3, round(length * .006))
                if (sum(v > threshold*2 for v in before) >= support
                        and sum(v > threshold*2 for v in after) >= support):
                    gaps.append(cut)
            start = None
    return gaps


def _partition_box(box, cuts, axis, direction):
    x1, y1, x2, y2 = box
    ends = [0, *cuts, x2-x1 if axis == "x" else y2-y1]
    result = ([(x1+a, y1, x1+b, y2) for a, b in zip(ends, ends[1:])] if axis == "x" else
              [(x1, y1+a, x2, y1+b) for a, b in zip(ends, ends[1:])])
    return result[::-1] if axis == "x" and direction == "vertical_rtl" else result


def plan_regions(image, cfg, *, page=0):
    """Return a disjoint, complete partition of the rendered page in read order."""
    cfg.__post_init__()
    full = (0, 0, image.width, image.height)
    info = {"method": "whole_page", "warnings": [], "coverage": 1.0, "overlap": False}
    if cfg.segmentation == "none":
        return [full], info
    options = cfg.segmentation_args
    allowed = {"column_cuts", "page_column_cuts", "columns_per_region", "max_regions",
               "ink_threshold", "gap_ink_fraction", "min_gap_fraction"}
    if set(options) - allowed:
        raise ValueError(f"Unknown segmentation options: {sorted(set(options)-allowed)}")
    cuts = options.get("page_column_cuts", {}).get(str(page+1), options.get("column_cuts"))
    native = cfg.segmentation == "auto" and cfg.engine in {"paddle", "tonggu"} and cuts is None
    if native:
        return [full], {**info, "method": "native_layout", "warnings": []}
    max_regions = int(options.get("max_regions", 64))
    group = int(options.get("columns_per_region", 3))
    if max_regions < 1 or group < 1:
        raise ValueError("max_regions and columns_per_region must be >= 1")
    boxes = [full]
    if cuts is not None:
        if (not isinstance(cuts, list) or any(not isinstance(v, (int, float)) or not isfinite(v)
                or not 0 < v < 1 for v in cuts) or cuts != sorted(set(cuts))):
            raise ValueError("column_cuts must be unique increasing fractions in (0, 1)")
        pixel_cuts = [round(x * image.width) for x in cuts]
        if len(set([0, *pixel_cuts, image.width])) != len(cuts)+2:
            raise ValueError("column_cuts collapse at this image resolution")
        boxes = _partition_box(full, pixel_cuts, "x", cfg.reading_direction)
        info["method"] = "manual_columns"
    elif cfg.segmentation == "columns" or (cfg.engine == "hunyuan" and cfg.reading_direction == "vertical_rtl"):
        found = projection_gaps(image, "x", options)
        # Group adjacent columns to retain language context. Group from the
        # reading start (right edge for historical vertical pages).
        if cfg.reading_direction == "vertical_rtl":
            selected = list(reversed(list(reversed(found))[group-1::group]))
        else:
            selected = found[group-1::group]
        if cfg.segmentation == "auto" and len(found) < 2*group-1:
            selected = []
            info["warnings"].append("too_few_gutters_for_context_groups; kept whole page")
        boxes = _partition_box(full, selected, "x", cfg.reading_direction)
        info.update(method="whitespace_columns", detected_gutters=found)
        info["warnings"].append("heuristic_layout; preview cuts on text/annotation pages before OCR")
        if not found:
            info["warnings"].append("no_confident_column_gutter; kept whole page")
    pixels, side = image_budget(cfg)
    def too_big(box):
        w, h = box[2]-box[0], box[3]-box[1]
        return w*h > pixels or (side > 0 and max(w,h) > side)
    # Split oversized regions at measured gutters, never at an arbitrary grid.
    pending, boxes = boxes, []
    while pending:
        box = pending.pop(0)
        if too_big(box):
            crop = image.crop(box)
            # Splitting a group of vertical columns horizontally would read the
            # tops of several columns before their bottoms. Only allow a row
            # split for a narrow column; otherwise retain it and report resize.
            axes = (["x", "y"] if crop.height >= 4*crop.width else ["x"]) if cfg.reading_direction == "vertical_rtl" else ["y", "x"]
            children = None
            for axis in axes:
                candidates = projection_gaps(crop, axis, options)
                size = crop.width if axis == "x" else crop.height
                candidates = [c for c in candidates if .15*size <= c <= .85*size]
                if candidates:
                    cut = min(candidates, key=lambda x: abs(x-size/2))
                    children = _partition_box(box, [cut], axis, cfg.reading_direction)
                    break
            if children and len(boxes)+len(pending)+2 <= max_regions:
                pending[0:0] = children
                continue
            info["warnings"].append("unsplittable_region_exceeds_budget")
        boxes.append(box)
    if len(boxes) > max_regions:
        raise ValueError("Region plan exceeds max_regions; increase columns_per_region")
    info.update(regions=len(boxes), coverage=1.0, overlap=False)
    return boxes, info


def prepare_region(image, box, cfg, *, native=False):
    from PIL import Image
    crop = image.crop(box)
    original = crop.size
    pixels, side = image_budget(cfg)
    # Native layout must see the source page; its recognizer controls each block.
    if native:
        return crop, {"crop_size": list(original), "input_size": list(original), "resized": False,
                      "scale_xy": [1.0, 1.0], "budget_scope": "native_blocks"}
    scale = min(cfg.region_upscale, sqrt(pixels/(crop.width*crop.height)))
    if side:
        scale = min(scale, side/max(crop.size))
    if cfg.resolution_strict and scale < 1:
        raise ValueError(f"Cannot preserve region {box}: {crop.size} exceeds pixel/side budget")
    target = (max(1, floor(crop.width*scale)), max(1, floor(crop.height*scale)))
    if target != original:
        crop = crop.resize(target, Image.Resampling.LANCZOS)
    return crop, {"crop_size": list(original), "input_size": list(target), "resized": target != original,
                  "downsampled": target[0] < original[0] or target[1] < original[1],
                  "scale_xy": [target[0]/original[0], target[1]/original[1]],
                  "max_pixels": pixels, "max_side": side, "budget_scope": "backend_input"}


def _page_geometry(extra, box, input_size, offset, region_index):
    """Map genuine crop detections back to the page; preserve their provenance."""
    from copy import deepcopy
    from .ocr_worker import ordered_page_blocks
    extra = deepcopy(extra)
    if "blocks" in extra:
        extra["blocks"] = ordered_page_blocks(extra["blocks"])
    sx, sy = (box[2]-box[0])/input_size[0], (box[3]-box[1])/input_size[1]
    def transform(points):
        if not isinstance(points, (list, tuple)):
            return points
        if len(points) == 4 and all(isinstance(v, (int, float)) for v in points):
            return [points[0]*sx+box[0], points[1]*sy+box[1], points[2]*sx+box[0], points[3]*sy+box[1]]
        return [[x*sx+box[0], y*sy+box[1]] for x,y in points]
    for entry in extra.get("character_boxes", []):
        entry["polygon"] = transform(entry.get("polygon"))
        entry["offset"] += offset
        entry["region_index"] = region_index
    for entry in extra.get("text_lines", []):
        entry["polygon"] = transform(entry.get("polygon"))
        entry["region_index"] = region_index
    for field in ("blocks", "boxes"):
        for entry in extra.get(field, []):
            entry["box"] = transform(entry.get("box"))
            entry["region_index"] = region_index
    return extra


def run_prepared_page(backend, image, image_path, cfg, *, page=0):
    """Run all regions transactionally; partial region text never becomes a page."""
    import tempfile
    if cfg.segmentation == "none":
        return backend.run(str(image_path))
    # The shared render cache remains immutable while inference reads crops.
    with tempfile.TemporaryDirectory(prefix="vtm-regions-") as directory:
        return _run_prepared_regions(backend, image, cfg, page=page, directory=Path(directory))


def _run_prepared_regions(backend, image, cfg, *, page, directory):
    from .text_alignment import comparison_text
    from .reading_order import order_ppocr_columns
    boxes, layout = plan_regions(image, cfg, page=page)
    if image.info.get("render", {}).get("downsampled"):
        layout["warnings"].append("page_render_downsampled_before_segmentation; raise max_side")
        if cfg.resolution_strict:
            raise ValueError("Page render was downsampled before segmentation; raise max_side")
    records, texts, merged, offset = [], [], {}, 0
    native = layout["method"] == "native_layout"
    for i, box in enumerate(boxes):
        crop, meta = prepare_region(image, box, cfg, native=native)
        path = directory / f"region_{i:03d}.png"
        crop.save(path, format="PNG")
        try:
            text, extra = backend.run(str(path))
        finally:
            path.unlink(missing_ok=True)
        observed = extra.get("model_input", {}).get("processed_sizes_wh", [])
        if len(observed) == 1:
            actual_w, actual_h = observed[0]
            meta["processor_size"] = [actual_w, actual_h]
            meta["processor_scale_xy"] = [actual_w/crop.width, actual_h/crop.height]
            meta["processor_downsampled"] = actual_w < crop.width or actual_h < crop.height
            if meta["processor_downsampled"]:
                layout["warnings"].append(f"region_{i}: processor_downsampled (includes patch-grid rounding)")
                if cfg.resolution_strict:
                    raise ValueError(f"Processor downsampled region {i}: {crop.size} -> {observed[0]}")
        if cfg.engine == "ppocr":
            ordered, order_info = order_ppocr_columns({"text": text, **extra},
                mode=("auto" if cfg.segmentation == "auto" else "vertical_rtl")
                     if cfg.reading_direction == "vertical_rtl" else "detection")
            text = ordered.pop("text")
            extra = {**ordered, "region_reading_order": order_info}
        records.append({"index": i, "page_box": list(box), **meta, "text": text, "raw": extra})
        mapped = _page_geometry(extra, box, crop.size, offset, i)
        for field in ("character_boxes", "text_lines", "blocks", "boxes"):
            if field in mapped:
                merged.setdefault(field, []).extend(mapped[field])
        offset += len(comparison_text(text))
        texts.append(text)
    # Page text and block ordering must describe exactly the same concatenation.
    for i, block in enumerate(merged.get("blocks", [])):
        block["order"] = i
    issues = sorted({issue for rec in records for issue in rec["raw"].get("quality_issues", [])})
    hit_flags = [rec["raw"].get("hit_max") for rec in records]
    hit_max = True if any(v is True for v in hit_flags) else (None if any(v is None for v in hit_flags) else False)
    if any(rec.get("downsampled") for rec in records):
        layout["warnings"].append("backend_input_downsampled; inspect regions or enable resolution_strict")
    observation = {"page_size": list(image.size), "render": image.info.get("render", {}),
                   "layout": layout, "regions": records,
                   "model_min_pixels": cfg.model_min_pixels, "model_max_pixels": cfg.model_max_pixels,
                   "processor_observation": "see region raw.model_input; native internal tensors may be unavailable"}
    return "\n".join(texts), {**merged, "hit_max": hit_max, "quality_issues": issues,
        "coordinate_frame": "input_image", "image_preparation": observation,
        "reading_direction": cfg.reading_direction, "region_order_applied": True}


def preview_preparation(src, page, cfg, *, output=None):
    """Model-free preview using 1-based PDF page numbers; never starts OCR."""
    from PIL import ImageDraw
    from .ocr_worker import render
    image = render(str(src), page-1, cfg.dpi, cfg.max_side)
    boxes, layout = plan_regions(image, cfg, page=page-1)
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    rows = []
    for i, box in enumerate(boxes):
        _, meta = prepare_region(image, box, cfg, native=layout["method"] == "native_layout")
        draw.rectangle((box[0], box[1], box[2]-1, box[3]-1), outline="red", width=max(1,image.width//600))
        draw.text((box[0]+3, box[1]+3), str(i+1), fill="red")
        rows.append({"region": i+1, "page_box": list(box), **meta})
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        preview.save(output)
    return preview, {"source": str(src), "page": page, "page_size": list(image.size),
                     "render": image.info.get("render", {}),
                     "engine": cfg.engine, "layout": layout, "regions": rows}
