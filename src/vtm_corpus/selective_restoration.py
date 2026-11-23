"""Conservative source-image triage and hard budgets for expensive restoration.

These measurements are proxies, not degradation labels or OCR accuracy. No
neural runtime is needed to decide that no region should be restored.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .image_dataset import dataset_read, dataset_load_image
from .restoration_registry import preparation_region_review, preparation_applicability


def plan_disagreement_restoration(root, base, config):
    """All six pairs must pass; rank and budget unique canonical source crops."""
    cfg = copy.deepcopy(config)
    options = cfg['selective_restoration']
    evidence = cfg.get('_ocr_disagreement_evidence', {})
    methods = [m for m in cfg['variants'] if m in ('tadisr_x4', 'difftsr', 'unihir_restoration')]
    if methods and options.get('thresholds') is None:
        cfg['_restoration_selected_keys']={m:[] for m in methods}
        cfg['_selective_unihir_keys']=[]
        cfg['_selective_restoration_report']=dict(schema='vtm-selective-restoration-v2',selected=[],skipped=[],
            by_method={},options=options,interpretation='thresholds_not_configured; analyze original OCR first')
        cfg['variants']=[v for v in cfg['variants'] if v not in methods and v!='tadisr_text_mask']
        return cfg
    units = {}
    for page in base['pages']:
        record = json.loads(dataset_read(root, page['record']))
        available = {r['crop_type'] for r in record['regions']} & set(cfg['crop_types'])
        for method in methods:
            priority = (['page', 'group', 'column'] if method == 'unihir_restoration' else
                        ['column', 'group', 'page'])
            kind = next((k for k in priority if k in available), None)
            for region in record['regions']:
                if region['crop_type'] != kind or (cfg.get('region_ids') and region['id'] not in cfg['region_ids']):
                    continue
                identity = (method, page['source_sha256'], page['page'], region['id'])
                alias = dict(key=f"{page['book']}:{page['page']+1}:{region['id']}",
                             book=page['book'], page=page['page']+1, region_id=region['id'])
                item = units.setdefault(identity, dict(**alias, aliases=[], variant=method,
                    source_sha256=page['source_sha256'], source_box=region['source_box'],
                    source_pixels=region['images']['original']['size'][0]*region['images']['original']['size'][1],
                    crop_type=kind, contract_reasons=[]))
                item['aliases'].append(alias)
                state, reason = preparation_applicability(method, page['book'], page['page'], region, cfg)
                review = preparation_region_review(page['book'], page['page'], region['id'], cfg)
                if method in review.get('excluded_variants', []):
                    state, reason = 'not_applicable', 'explicitly_excluded'
                if method == 'difftsr':
                    box = region['source_box']; ratio = (box[2]-box[0])/(box[3]-box[1])
                    if not (review.get('upright_horizontal_single_line') and
                            review.get('allow_simplified_text_prior') and
                            1 <= review.get('character_count', 0) <= 24 and abs(ratio/4-1) <= .05):
                        state, reason = 'not_applicable', 'requires_reviewed_horizontal_line_and_DiffTSR_contract'
                if state != 'ready':
                    item['contract_reasons'].append(reason)
    selected, skipped, budgets, keys = [], [], {}, {m:[] for m in methods}
    for method in methods:
        if method not in options['thresholds']:
            raise ValueError('Configure a disagreement threshold for '+method)
        threshold = options['thresholds'][method]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError('Restoration disagreement thresholds must be in [0, 1]')
        budget = (dict(max_regions=options.get('max_regions', 12),
                       max_regions_per_book=options.get('max_regions_per_book', 2),
                       max_total_source_pixels=options.get('max_total_source_pixels', 12_000_000))
                  if method == 'unihir_restoration' else
                  options.get('method_budgets', {}).get(method, dict(max_regions=256, max_total_source_pixels=32_000_000)))
        for name, value in budget.items():
            if name not in {'max_regions', 'max_regions_per_book', 'max_total_source_pixels'} or (value is not None and (type(value) is not int or value < 0)):
                raise ValueError('Invalid restoration budget: '+name)
        candidates = []
        for item in units.values():
            if item['variant'] != method:
                continue
            score = evidence.get('scores', {}).get(f"{item['source_sha256']}:{item['page']-1}:{item['region_id']}", {})
            item['ocr_disagreement'] = score
            item['threshold'] = threshold
            reason = ('; '.join(sorted(set(item.pop('contract_reasons')))) or
                      (score.get('reason', score.get('state', 'original_ocr_not_available')) if score.get('state') != 'ok' else
                       'pairwise_disagreement_below_threshold' if score['min_pairwise_rate'] < threshold else None))
            if reason:
                skipped.append({**item, 'reason':reason})
            else:
                candidates.append(item)
        count, used, by_book = 0, 0, {}
        for item in sorted(candidates, key=lambda x:(-x['ocr_disagreement']['min_pairwise_rate'],
                                                    -x['ocr_disagreement']['mean_pairwise_rate'], x['source_sha256'], x['page'], x['region_id'])):
            source = item['source_sha256']
            values = dict(max_regions=count+1, max_regions_per_book=by_book.get(source, 0)+1,
                          max_total_source_pixels=used+item['source_pixels'])
            if any(limit is not None and values[name] > limit for name,limit in budget.items()):
                skipped.append({**item, 'reason':'selective_restoration_budget'})
                continue
            selected.append({**item, 'reason':'all_eligible_original_ocr_pairs_above_threshold'})
            count += 1; used += item['source_pixels']; by_book[source] = by_book.get(source, 0)+1
            keys[method].extend(alias['key'] for alias in item['aliases'])
        budgets[method] = dict(selected_unique_regions=count, source_pixels=used, limits=budget, threshold=threshold)
    cfg['_restoration_selected_keys'] = keys
    cfg['_selective_unihir_keys'] = keys.get('unihir_restoration', [])
    cfg['_selective_restoration_report'] = dict(schema='vtm-selective-restoration-v2', selected=selected,
        skipped=skipped, by_method=budgets, options=options, original_manifest=evidence.get('original_manifest'),
        interpretation='all six normalized Levenshtein rates from original OCR; triage only, not verified damage or accuracy')
    cfg['variants'] = [v for v in cfg['variants'] if v not in keys or keys[v]]
    if 'tadisr_x4' not in cfg['variants']:
        cfg['variants'] = [v for v in cfg['variants'] if v != 'tadisr_text_mask']
    cfg.pop('_ocr_disagreement_evidence', None)
    return cfg


def restoration_source_metrics(image):
    import cv2
    import numpy as np
    original_size = image.size
    preview = image.convert('L')
    preview.thumbnail((1024, 1024))
    gray = np.asarray(preview)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    components = stats[1:]
    components = components[(components[:, cv2.CC_STAT_AREA] >= 3) &
                            (components[:, cv2.CC_STAT_AREA] <= gray.size * .02)]
    foreground = gray[ink > 0]
    contrast = float(np.percentile(gray, 90) - np.median(foreground)) if foreground.size else 0.
    return dict(size=list(original_size), analysis_size=list(preview.size), ink_fraction=float(np.mean(ink > 0)),
                contrast=max(0., contrast), component_count=int(len(components)),
                median_component_height=float(np.median(components[:, cv2.CC_STAT_HEIGHT]))*original_size[1]/preview.height if len(components) else 0.)


def plan_selective_restoration(root, base, config):
    """Rank disjoint source crops before any model asset/runtime preflight."""
    cfg = copy.deepcopy(config)
    options = cfg.get('selective_restoration', {})
    if options.get('enabled') and options.get('strategy') == 'ocr_disagreement':
        return plan_disagreement_restoration(root, base, cfg)
    if not options.get('enabled', False) or 'unihir_restoration' not in cfg['variants']:
        return cfg
    maximum = options.get('max_regions', 2)
    per_page = options.get('max_regions_per_page', 1)
    pixel_budget = options.get('max_total_source_pixels', 4_000_000)
    for name, value in [('max_regions', maximum), ('max_regions_per_page', per_page), ('max_total_source_pixels', pixel_budget)]:
        if type(value) is not int or value < 0:
            raise ValueError(f'selective_restoration.{name} must be a nonnegative integer')
    candidates, rejected, seen = [], [], set()
    for page in base['pages']:
        record = json.loads(dataset_read(root, page['record']))
        for region in record['regions']:
            if region['crop_type'] != options.get('crop_type', 'group') or region['crop_type'] not in cfg['crop_types']:
                continue
            if cfg.get('region_ids') and region['id'] not in cfg['region_ids']:
                continue
            key = f"{page['book']}:{page['page'] + 1}:{region['id']}"
            identity = (page['source_sha256'], page['page'], tuple(region['source_box']))
            item = dict(key=key, book=page['book'], page=page['page']+1,
                        region_id=region['id'], source_box=region['source_box'])
            if identity in seen:
                rejected.append({**item, 'reason': 'duplicate_source_crop'})
                continue
            seen.add(identity)
            size = region['images']['original']['size']
            if not maximum or not per_page or size[0]*size[1] > pixel_budget:
                rejected.append({**item, 'reason': 'selective_restoration_budget'})
                continue
            with dataset_load_image(root, region['images']['original']) as image:
                metrics = restoration_source_metrics(image)
            item['metrics'] = metrics
            reviews = cfg.get('reviewed_regions', {})
            review = reviews.get(key, reviews.get(f"{str(page['book']).removesuffix('.pdf')}:{page['page']+1}:{region['id']}", {}))
            approved = ('unihir_restoration' in review.get('approved_variants', [])
                        and bool(review.get('restoration_reason')))
            excluded = 'unihir_restoration' in review.get('excluded_variants', [])
            nonblank = .003 <= metrics['ink_fraction'] <= .40 and metrics['component_count'] >= 12
            faint = metrics['contrast'] < options.get('contrast_threshold', 64)
            small = (min(metrics['size']) < options.get('short_side_threshold', 512) and
                     metrics['median_component_height'] < options.get('component_height_threshold', 6))
            if excluded or not (approved or (nonblank and (faint or small))):
                rejected.append({**item, 'reason': 'explicitly_excluded' if excluded else 'no_conservative_source_trigger'})
                continue
            item['reason'] = ('reviewed: ' + review['restoration_reason'] if approved else
                              'faint_ink_proxy' if faint else 'small_ink_components_proxy')
            item['score'] = (1000 if approved else 0) + max(0, 64-metrics['contrast']) + max(0, 6-metrics['median_component_height'])
            candidates.append(item)
    selected, pages, used = [], {}, 0
    for item in sorted(candidates, key=lambda x: (-x['score'], x['key'])):
        size = item['metrics']['size']; pixels = size[0]*size[1]
        page_key = (item['book'], item['page'])
        if len(selected) >= maximum or pages.get(page_key, 0) >= per_page or used+pixels > pixel_budget:
            rejected.append({**item, 'reason': 'selective_restoration_budget'})
            continue
        selected.append(item); used += pixels; pages[page_key] = pages.get(page_key, 0)+1
    cfg['_selective_unihir_keys'] = [item['key'] for item in selected]
    cfg['_selective_restoration_report'] = dict(schema='vtm-selective-restoration-v1',
        selected=selected, skipped=rejected, source_pixels=used, options=options,
        interpretation='source proxies only; not verified degradation, necessity, or accuracy')
    if not selected:
        cfg['variants'] = [name for name in cfg['variants'] if name != 'unihir_restoration']
    return cfg


def selective_region_applicability(model, book, page, region, config):
    if model in config.get('_restoration_selected_keys', {}):
        key = f"{book}:{page+1}:{region['id']}"
        if key not in config['_restoration_selected_keys'][model]:
            return 'not_applicable', 'selective_restoration: original OCR disagreement below threshold, input contract or budget; original retained'
        return 'ready', None
    if model != 'unihir_restoration' or '_selective_unihir_keys' not in config:
        return 'ready', None
    key = f"{book}:{page+1}:{region['id']}"
    if key not in config['_selective_unihir_keys']:
        return 'not_applicable', 'selective_unihir: source trigger absent or budget exhausted; original retained'
    return 'ready', None
