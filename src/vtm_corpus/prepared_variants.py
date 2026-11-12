"""Read-only bridge to notebook 1's restoration extension. No restoration inference."""
from __future__ import annotations

import json
from math import isclose, isfinite

from .image_dataset import (dataset_hash, dataset_read, dataset_load_image, dataset_page,
                            dataset_validate_page)


OCR_IMAGE_VARIANTS = {
    "original": (1, "source_observation"),
    "clahe": (1, "deterministic_transformation"),
    "bicubic_x2": (2, "deterministic_transformation"),
    "bicubic_x4": (4, "deterministic_transformation"),
    "denoised": (1, "deterministic_transformation"),
    "realesrgan_x4": (4, "neural_restoration"),
    "tadisr_x4": (4, "neural_restoration"),
    "difftsr": (None, "neural_restoration"),
    "unihir_restoration": (1, "generative_reconstruction"),
}

RESTORATION_FAILURE_STATES = {
    'failed', 'missing_assets', 'runtime_incompatible', 'inference_failed',
    'timeout', 'resource_exceeded',
}


class VariantUnavailable(ValueError):
    def __init__(self, states):
        self.states = states
        super().__init__("Prepared variant unavailable: " + json.dumps(states, ensure_ascii=False))


def restoration_index(root, descriptor):
    """Validate extension identity and index rows; pixels are verified when consumed."""
    ext = descriptor.get("restoration")
    if not ext:
        return {}, None
    if ext.get("format") != "vtm-image-restoration" or ext.get("version") != 1:
        raise ValueError("Unsupported restoration extension format/version")
    report = json.loads(dataset_read(root, ext["report"]))
    if report.get("format") != ext["format"] or report.get("version") != 1:
        raise ValueError("Restoration report identity mismatch")
    canonical = {k: v for k, v in descriptor.items() if k not in {"restoration", "snapshot_id"}}
    if dataset_hash(canonical) != report.get("canonical_snapshot"):
        raise ValueError("Restoration canonical snapshot mismatch")
    rows, counts = {}, {}
    for line in dataset_read(root, report["manifest"]).splitlines():
        row = json.loads(line)
        key = (row["source_sha256"], row["page_index"], row["region_id"], row["variant"])
        if key in rows or row.get("page") != row["page_index"] + 1:
            raise ValueError("Duplicate/invalid restoration region identity")
        rows[key] = row
        state = row.get("state")
        if state not in {"ok", "not_applicable", "skipped_by_configuration"} | RESTORATION_FAILURE_STATES:
            raise ValueError("Invalid restoration variant state")
        counts[state] = counts.get(state, 0) + 1
    expected = set()
    for page in descriptor["pages"]:
        rec = dataset_page(root, descriptor, page["source_sha256"], page["page"])
        for region in rec["regions"]:
            for variant in report["selected_variants"]:
                key = (page["source_sha256"], page["page"], region["id"], variant)
                expected.add(key)
                if key not in rows:
                    raise ValueError("Restoration manifest silently omits selected variant/region")
                row = rows[key]
                if (row.get("book") != page["book"] or row.get("source_image_hash") != rec["source"]["sha256"]
                        or row.get("source_image") != rec["source"]
                        or row.get("canonical_image") != region["images"]["original"]
                        or any(row.get(k) != region[k] for k in ("source_box", "crop_type", "reading_order"))):
                    raise ValueError("Restoration source/crop identity mismatch")
                expected_provenance = ("neural_restoration" if variant == "tadisr_text_mask"
                                       else OCR_IMAGE_VARIANTS.get(variant, (None, None))[1])
                if row.get("provenance") != expected_provenance or expected_provenance is None:
                    raise ValueError("Restoration provenance mismatch")
                if variant == 'original' and row['state'] == 'ok' and any(
                        row['image'].get(k) != region['images']['original'][k] for k in ('sha256', 'pixel_sha256', 'size')):
                    raise ValueError('Restoration original row differs from canonical pixels')
    if set(rows) != expected or counts != report["states"]:
        raise ValueError("Restoration manifest inventory mismatch")
    if any(counts.get(s, 0) for s in RESTORATION_FAILURE_STATES) and not report.get("partial"):
        raise ValueError("Failed restoration falsely labelled complete")
    if bool(ext.get('partial')) != bool(report.get('partial')):
        raise ValueError('Restoration extension partial status mismatch')
    return rows, report


def variant_region_entry(root, descriptor, record, region, variant, *, index=None, report=None,
                         allow_generated=False):
    if variant not in OCR_IMAGE_VARIANTS:
        raise ValueError(f"Unsupported OCR variant: {variant}; diagnostic masks are not OCR images")
    if variant in {"difftsr", "unihir_restoration"} and not allow_generated:
        raise ValueError("Generated-content comparison requires explicit opt-in")
    if variant in region["images"]:
        entry = region["images"][variant]
        image = dataset_load_image(root, entry)
        return entry, {"state": "ok", "provenance": OCR_IMAGE_VARIANTS[variant][1]}
    if index is None:
        index, report = restoration_index(root, descriptor)
    row = index.get((record["source_sha256"], record["page"], region["id"], variant))
    if row is None:
        raise VariantUnavailable([{"region": region["id"], "variant": variant, "state": "missing_variant"}])
    if row["state"] != "ok":
        raise VariantUnavailable([{k: row.get(k) for k in ("region_id", "variant", "state", "reason")}])
    if variant in {"difftsr", "unihir_restoration"}:
        review = report["configuration"].get("reviewed_regions", {}).get(
            f'{row["book"]}:{row["page"]}:{row["region_id"]}', {})
        if variant == "difftsr":
            box = region["source_box"]
            ratio = (box[2]-box[0])/(box[3]-box[1])
            if not (review.get("upright_horizontal_single_line") and review.get("allow_simplified_text_prior")
                    and 1 <= review.get("character_count", 0) <= 24 and abs(ratio/4-1) <= .05):
                raise ValueError("DiffTSR applicability metadata/input contract unverified")
        elif not (review.get("damaged_inscription") and report["configuration"].get("accept_unihir_noncommercial")):
            raise ValueError("UniHIR applicability metadata unverified")
    image = dataset_load_image(root, row["image"])
    box = region["source_box"]
    scale = OCR_IMAGE_VARIANTS[variant][0]
    if scale is not None and list(image.size) != [(box[2]-box[0])*scale, (box[3]-box[1])*scale]:
        raise ValueError("Restoration scale/dimensions mismatch")
    if variant == "difftsr" and image.size != (512, 128):
        raise ValueError("DiffTSR processor canvas must be 512x128")
    mapping = [(box[2]-box[0])/image.width, 0, 0, (box[3]-box[1])/image.height, box[0], box[1]]
    affine = row["image"].get("image_to_source", [])
    if len(affine) != 6 or any(not isinstance(a, (int, float)) or not isfinite(a)
                             or not isclose(a, b, abs_tol=1e-9) for a, b in zip(affine, mapping)):
        raise ValueError("Restoration coordinate mapping mismatch")
    for ref in row.get("extras", {}).values():
        dataset_read(root, ref)
    return {**row["image"], "operation": row.get("operation", {}), "key": row["key"],
            "provenance": row["provenance"], "model_identity": row.get("model_identity"),
            "coordinate_mapping_evidence": row.get("coordinate_mapping_evidence"),
            "warnings": row.get("warnings", [])}, row


def resolve_variant_page(root, descriptor, record, cfg, *, index=None, report=None):
    from copy import deepcopy
    record = deepcopy(record)
    kind = "page" if cfg.image_crop_type == "native" else cfg.image_crop_type
    states = []
    for region in record["regions"]:
        if region["crop_type"] != kind:
            continue
        try:
            entry, _ = variant_region_entry(root, descriptor, record, region, cfg.image_variant,
                index=index, report=report, allow_generated=cfg.allow_generated_variants)
            region["images"][cfg.image_variant] = entry
        except VariantUnavailable as exc:
            states.extend(exc.states)
    if states:
        raise VariantUnavailable(states)
    return record


def bind_restoration_variant(cfg, root, descriptor, source_sha, pages, *, allow_unavailable=False):
    """Do not put other selected variants into this variant's cache identity."""
    index, report = restoration_index(root, descriptor)
    states, identity = {}, []
    # Identity covers this source's input assets, independent of OCR page selection.
    for page in descriptor["pages"]:
        if page["source_sha256"] != source_sha:
            continue
        record = dataset_page(root, descriptor, source_sha, page["page"])
        layout = dataset_validate_page(root, record)
        if layout['reading_direction'] != cfg.reading_direction:
            raise ValueError('Dataset and OCR reading directions differ')
        try:
            resolved = resolve_variant_page(root, descriptor, record, cfg, index=index, report=report)
            kind = "page" if cfg.image_crop_type == "native" else cfg.image_crop_type
            assets = [{"region": r["id"], "box": r["source_box"], "order": r["reading_order"],
                       "image": r["images"][cfg.image_variant]} for r in resolved["regions"] if r["crop_type"] == kind]
            if not assets:
                raise ValueError('Prepared crop type contains no regions')
            # Metadata such as wall time, cache receipts and old filesystem
            # locations is not input to OCR and must not invalidate its output.
            for asset in assets:
                asset['image'] = {k: v for k, v in asset['image'].items() if k in {
                    'sha256', 'pixel_sha256', 'size', 'mode', 'image_to_source',
                    'provenance', 'model_identity', 'operation', 'coordinate_mapping_evidence'}}
            identity.append({"page": page["page"], "source": record["source"], "assets": assets})
            states[page["page"]] = {"state": "ok"}
        except VariantUnavailable as exc:
            states[page["page"]] = {"state": "unavailable", "regions": exc.states}
    selected = {p: states.get(p, {"state": "missing_page"}) for p in pages}
    if not allow_unavailable and any(s["state"] != "ok" for s in selected.values()):
        raise VariantUnavailable(selected)
    cfg.image_content_key = dataset_hash({"reader": "restoration-ocr-v1", "source_pdf": source_sha,
        "variant": cfg.image_variant, "inputs": identity})
    return selected
