"""Independent source-ink candidates followed by conservative OCR overlap diagnostics.

No transcription or OCR geometry is accepted by generate_visual_candidates().
All overlap denominators are analysis-mask ink pixels, never character counts.
"""
from __future__ import annotations

import hashlib
import math

from .character_regions import normalized_polygon


SPATIAL_DEFAULTS = {
    "analysis_max_side": 1600, "min_contrast": 10, "adaptive_window": 31,
    "min_component_area": 3, "min_group_components": 3, "group_gap_factor": 1.8,
    "min_density": .08, "max_density": .70, "min_candidate_ink": 40, "border_fraction": .025,
    "covered_threshold": .85,
    "partial_threshold": .10, "max_candidates": 256, "max_components": 30000,
    "save_overlays": True, "save_candidate_masks": False, "max_detail_crops": 12,
    "max_source_pixels": 100_000_000,
}


def spatial_settings(options=None):
    options = options or {}
    if set(options) - SPATIAL_DEFAULTS.keys():
        raise ValueError("Unknown spatial coverage options")
    settings = {**SPATIAL_DEFAULTS, **options}
    for key in ("analysis_max_side", "min_contrast", "adaptive_window", "min_component_area",
                "min_group_components", "min_candidate_ink", "max_candidates", "max_components", "max_source_pixels"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"Invalid spatial option: {key}")
    if not 32 <= settings["analysis_max_side"] <= 4096 or not 1 <= settings["min_contrast"] < 255:
        raise ValueError("Invalid spatial analysis size/contrast")
    if settings['adaptive_window'] < 3 or settings['adaptive_window'] % 2 != 1:
        raise ValueError("adaptive_window must be odd and >=3")
    if not 0 <= settings['partial_threshold'] < settings['covered_threshold'] <= 1:
        raise ValueError("Invalid overlap thresholds")
    if not 0 < settings['min_density'] < settings['max_density'] < 1:
        raise ValueError("Invalid density thresholds")
    if not 0 <= settings['border_fraction'] < .2:
        raise ValueError('Invalid border fraction')
    if not math.isfinite(settings['group_gap_factor']) or not .2 <= settings['group_gap_factor'] <= 5:
        raise ValueError("Invalid grouping gap")
    if type(settings['max_detail_crops']) is not int or settings['max_detail_crops'] < 0:
        raise ValueError("Invalid crop limit")
    return settings


def generate_visual_candidates(image, *, options=None, reading_direction="vertical_rtl"):
    """Classical heuristic only; even strong candidates are not confirmed text."""
    import cv2
    import numpy as np
    from PIL import Image
    p = spatial_settings(options)
    if reading_direction not in {"vertical_rtl", "horizontal_ltr"}:
        raise ValueError("Unsupported reading direction")
    if image.width*image.height > p['max_source_pixels']:
        raise ValueError("Spatial source exceeds max_source_pixels")
    analysis = image.convert("L")
    analysis.thumbnail((p['analysis_max_side'], p['analysis_max_side']), Image.Resampling.LANCZOS)
    gray = np.asarray(analysis)
    h, w = gray.shape
    # Absolute residuals avoid promoting weak uniform scanner noise by autocontrast.
    residual = np.maximum.reduce([cv2.GaussianBlur(gray, (0, 0), sigma).astype(np.int16)-gray
                                 for sigma in (max(2, min(w, h)/100), max(4, min(w, h)/40))])
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, p['adaptive_window'], p['min_contrast']/2)
    mask = ((residual >= p['min_contrast']) & (adaptive > 0)).astype(np.uint8)
    # Remove long straight rules only from the analytical mask, retaining their boxes.
    rules = np.maximum(cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((max(20, h//3), 1), np.uint8)),
                       cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, max(20, w//3)), np.uint8)))
    ink = mask & (1-rules)
    # Reconnect small broken strokes before estimating component scale.
    joined = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(joined, connectivity=8)
    complete = n-1 <= p['max_components']
    selected = []
    rejected_large = []
    clean = np.zeros_like(ink)
    for i in range(1, min(n, p['max_components']+1)):
        x, y, bw, bh, area = map(int, stats[i])
        if area < p['min_component_area']:
            continue
        if max(bw, bh) > .3*max(w, h):
            rejected_large.append((x, y, bw, bh, area))
            continue
        if max(bw/bh, bh/bw) > 16 or area/(bw*bh) > .98 and min(bw, bh) < 2:
            continue
        selected.append(i)
    if selected:
        clean = np.isin(labels, selected).astype(np.uint8)
    unit = float(np.median([max(stats[i, 2], stats[i, 3]) for i in selected])) if selected else 3.0
    gap = max(3, min(round(unit*p['group_gap_factor']), max(3, min(w, h)//12)))
    kernel = (gap, max(1, round(unit*.2))) if reading_direction == 'vertical_rtl' else (max(1, round(unit*.2)), gap)
    grouped = cv2.dilate(clean, np.ones(kernel, np.uint8))
    count, groups, group_stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    candidates = []
    selected_set = set(selected)
    if count-1 > p['max_candidates']:
        complete = False
    for i in range(1, min(count, p['max_candidates']+1)):
        gx, gy, gw, gh = map(int, group_stats[i, :4])
        membership = groups[gy:gy+gh, gx:gx+gw] == i
        ys, xs = np.where(membership & (ink[gy:gy+gh, gx:gx+gw] > 0))
        if not len(xs):
            continue
        xs, ys = xs+gx, ys+gy
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)
        component_ids = [j for j in np.unique(labels[gy:gy+gh, gx:gx+gw][membership]) if j in selected_set]
        density = len(xs)/((x2-x1)*(y2-y1))
        centers = sorted(float(centroids[j, 1 if reading_direction == 'vertical_rtl' else 0]) for j in component_ids)
        spacings = np.diff(centers)
        spacing_cv = float(np.std(spacings)/max(1, np.mean(spacings))) if len(spacings) else None
        reasons = []
        if len(component_ids) < p['min_group_components']:
            reasons.append('too_few_components')
        if len(xs) < p['min_candidate_ink']:
            reasons.append('insufficient_candidate_ink')
        margin = p['border_fraction']
        if x1 < w*margin or y1 < h*margin or x2 > w*(1-margin) or y2 > h*(1-margin):
            reasons.append('page_edge_or_border; inspect_possible_marginal_text')
        if not p['min_density'] <= density <= p['max_density']:
            reasons.append('atypical_ink_density')
        if spacing_cv is not None and spacing_cv > 2:
            reasons.append('irregular_component_spacing')
        bw, bh = x2-x1, y2-y1
        if (bw > .45*w and bh > .45*h) or len(component_ids) > 500:
            reasons.append('large_ink_texture_or_illustration')
        if not complete:
            reasons.append('component_limit')
        box = [x1*image.width/w, y1*image.height/h, x2*image.width/w, y2*image.height/h]
        candidates.append({'bbox': box, 'analysis_box': [x1, y1, x2, y2],
            'visual_status': 'ambiguous_visual_candidate' if reasons else 'potential_text',
            'method': 'source_multiscale_residual_adaptive_cc_directional_grouping_v1',
            'measurements': {'ink_pixels': int(len(xs)), 'density': density,
                             'components': len(component_ids), 'spacing_cv': spacing_cv},
            'uncertainty': reasons + ['classical_visual_heuristic; manual_inspection_required']})
    for x, y, bw, bh, area in rejected_large[:p['max_candidates']]:
        candidates.append({'bbox': [x*image.width/w, y*image.height/h, (x+bw)*image.width/w, (y+bh)*image.height/h],
            'analysis_box': [x, y, x+bw, y+bh], 'visual_status': 'ambiguous_visual_candidate',
            'method': 'source_oversized_component_context_v1',
            'measurements': {'ink_pixels': int(ink[y:y+bh, x:x+bw].sum()), 'density': float(area/(bw*bh)),
                             'components': 1, 'spacing_cv': None},
            'uncertainty': ['large_connected_ink; touching_text_or_nontext_unresolved']})
    candidates.sort(key=lambda c: (-c['bbox'][0], c['bbox'][1]) if reading_direction == 'vertical_rtl'
                    else (c['bbox'][1], c['bbox'][0]))
    if len(candidates) > p['max_candidates']:
        complete = False
        candidates = candidates[:p['max_candidates']]
    for c in candidates:
        c['id'] = 'visual_' + hashlib.sha256(str(c['analysis_box']).encode()).hexdigest()[:12]
    return {'candidates': candidates, 'ink_mask': ink, 'analysis_size': [w, h],
            'source_size': list(image.size), 'analysis_to_source': [image.width/w, image.height/h],
            'complete': complete, 'parameters': p, 'reading_direction': reading_direction,
            'removed_rule_ink_pixels': int(rules.sum()), 'component_count': n-1}


def collect_spatial_evidence(records, source, *, max_boxes=50000):
    """Accept genuine polygons in a verified frame, never crop bounds/layout-only boxes."""
    boxes, models = [], {}
    for engine, record in records.items():
        if record is None:
            models[engine] = {'status': 'missing_output', 'boxes': 0}
            continue
        raw = record.get('geometry', record.get('raw', record))
        prep = raw.get('image_preparation', {})
        dataset = prep.get('dataset', {})
        status = 'geometry_unavailable'
        if dataset:
            valid = (dataset.get('source_sha256') == source['source_sha256'] and
                     dataset.get('source', {}).get('sha256') == source.get('image_sha256') and
                     [raw.get('w'), raw.get('h')] == source['size'])
            if not valid:
                models[engine] = {'status': 'source_geometry_mismatch', 'boxes': 0}
                continue
        elif not source.get('legacy_frame_verified', False):
            models[engine] = {'status': 'source_geometry_mismatch', 'boxes': 0}
            continue
        if raw.get('coordinate_frame') != 'input_image':
            models[engine] = {'status': status, 'boxes': 0,
                              'limitation': 'No verified adapter for localization; text is not an absence vote'}
            continue
        if any(r.get('provenance') == 'generative_reconstruction' or
               r.get('raw', {}).get('coordinate_frame') == 'unverified_native_transform' or
               str(r.get('coordinate_mapping_evidence', '')).startswith('nominal_canvas')
               for r in prep.get('regions', [])):
            models[engine] = {'status': 'geometry_unavailable', 'boxes': 0,
                              'limitation': 'Generated canvas is not observed glyph geometry'}
            continue
        width, height = raw.get('w', 0), raw.get('h', 0)
        own = []
        for field, granularity, key in (('character_boxes', 'character', 'polygon'),
                                         ('text_lines', 'line', 'polygon'), ('blocks', 'block', 'box'),
                                         ('spatial_regions', 'block', 'box')):
            for entry in raw.get(field, []):
                if not entry.get('text') or (granularity == 'character' and
                    (entry.get('granularity') != 'character' or len(entry['text']) != 1)):
                    continue
                polygon = normalized_polygon(entry.get(key), width, height)
                if polygon is None:
                    status = 'source_geometry_mismatch'
                    continue
                own.append({'engine': engine, 'granularity': granularity, 'estimated': True,
                    'polygon': [[x*source['size'][0], y*source['size'][1]] for x, y in polygon['polygon']],
                    'coordinate_uncertainty': 'model_estimate; raster_scale_and_rounding',
                    'method': entry.get('method', field)})
                if len(own) >= max_boxes:
                    break
            if len(own) >= max_boxes:
                break
        models[engine] = {'status': 'localized' if own else status, 'boxes': len(own),
                          'invalid_geometry': status == 'source_geometry_mismatch',
                          'complete': len(own) < max_boxes}
        boxes.extend(own)
    return boxes, models


def analyze_spatial_coverage(image, records, source, *, options=None, reading_direction='vertical_rtl'):
    import cv2
    import numpy as np
    if source.get('provenance', 'source_observation') != 'source_observation':
        raise ValueError('Independent coverage requires verified source observation, never generated masks/restoration')
    # This call is deliberately before any access to OCR records.
    visual = generate_visual_candidates(image, options=options, reading_direction=reading_direction)
    boxes, models = collect_spatial_evidence(records, source)
    w, h = visual['analysis_size']
    sw, sh = image.size
    character = np.zeros((h, w), np.uint8)
    coarse = np.zeros_like(character)
    per_engine = {e: np.zeros_like(character) for e, m in models.items() if m['boxes']}
    for box in boxes:
        points = np.array([[round(x*w/sw), round(y*h/sh)] for x, y in box['polygon']], np.int32)
        cv2.fillPoly(character if box['granularity'] == 'character' else coarse, [points], 1)
        cv2.fillPoly(per_engine[box['engine']], [points], 1)
    union = character | coarse
    candidates, candidate_ink = [], np.zeros_like(character)
    for candidate in visual['candidates']:
        x1, y1, x2, y2 = candidate['analysis_box']
        ink = visual['ink_mask'][y1:y2, x1:x2] > 0
        candidate_ink[y1:y2, x1:x2] |= ink.astype(np.uint8)
        denominator = int(ink.sum())
        ratio = float((ink & (union[y1:y2, x1:x2] > 0)).sum()/denominator)
        char_ratio = float((ink & (character[y1:y2, x1:x2] > 0)).sum()/denominator)
        by_engine = {e: float((ink & (m[y1:y2, x1:x2] > 0)).sum()/denominator) for e, m in per_engine.items()}
        p = visual['parameters']
        if candidate['visual_status'] != 'potential_text':
            kind = 'ambiguous_visual_candidate'
        elif not per_engine:
            kind = ('source_geometry_mismatch' if any(m['status'] == 'source_geometry_mismatch' for m in models.values())
                    else 'geometry_unavailable')
        elif char_ratio >= p['covered_threshold']:
            kind = 'covered_by_character_evidence'
        elif ratio >= p['covered_threshold']:
            kind = 'covered_by_coarse_ocr_region'
        elif ratio >= p['partial_threshold']:
            kind = 'partially_covered_region'
        else:
            kind = 'potential_uncovered_text_region'
        finding_id = hashlib.sha256(str((source['source_sha256'], source['page'],
                                        source.get('image_sha256'), candidate['id'])).encode()).hexdigest()[:24]
        candidates.append({**candidate, 'finding_id': finding_id, 'kind': kind, 'source_sha256': source['source_sha256'],
            'source_image_sha256': source.get('image_sha256'), 'book': source['book'], 'page': source['page'],
            'candidate_ink_overlap_ratio': ratio if per_engine else None,
            'character_ink_overlap_ratio': char_ratio if per_engine else None,
            'overlap_by_engine': by_engine, 'localized_engines': list(per_engine),
            'overlapped_engines': [e for e, fraction in by_engine.items() if fraction > 0],
            'engines_without_localization': [e for e in models if e not in per_engine],
            'potential_shared_omission': kind == 'potential_uncovered_text_region' and len(per_engine) == len(records),
            'evidence_granularity': sorted({b['granularity'] for b in boxes}),
            'parameters': p, 'source_reference': source.get('image_reference'),
            'uncertainty': candidate['uncertainty'] + (['coarse_boxes_do_not_establish_character_completeness']
                                                     if ratio > char_ratio else [])})
    denominator = int(candidate_ink.sum())
    numerator = int((candidate_ink & union).sum())
    complete = visual['complete'] and all(m.get('complete', True) for m in models.values())
    summary = {'candidates': len(candidates), 'localized_candidates': sum((c['candidate_ink_overlap_ratio'] or 0) > 0 for c in candidates),
        'potential_uncovered': sum(c['kind'] == 'potential_uncovered_text_region' for c in candidates),
        'partial_candidates': sum(c['kind'] == 'partially_covered_region' for c in candidates),
        'manual_review': sum(c['kind'] != 'covered_by_character_evidence' for c in candidates),
        'candidate_ink_pixels': denominator, 'overlapped_candidate_ink_pixels': numerator if per_engine else None,
        'candidate_ink_overlap_ratio': numerator/denominator if denominator and per_engine else None}
    return {'schema': 'vtm-spatial-coverage-v1', 'status': 'analyzed' if complete else 'analysis_incomplete',
        'complete': complete, 'source': source, 'findings': candidates, 'ocr_evidence': boxes, 'models': models,
        'parameters': visual['parameters'], 'analysis_size': visual['analysis_size'], 'summary': summary,
        'visual_measurements': {k: visual[k] for k in ('component_count', 'removed_rule_ink_pixels', 'analysis_to_source')},
        'denominator': 'union of candidate ink pixels on bounded source-analysis mask; not characters',
        'limitations': ['Heuristic visual candidates, not calibrated text detection or OCR accuracy.',
                        'Text-only engines abstain from localization; coarse overlap is context only.']}, visual['ink_mask']
