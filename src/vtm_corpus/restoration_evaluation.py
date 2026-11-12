"""Read-only H10 diagnostics over canonical images and existing OCR checkpoints.

No model imports, restoration, text correction, or automatic correctness labels.
"""
from __future__ import annotations

from itertools import combinations
import json
import math
from pathlib import Path

from .image_dataset import (dataset_open, dataset_page, dataset_validate_page,
    dataset_load_image, dataset_hash, dataset_file_hash, dataset_path)
from .prepared_variants import restoration_index, variant_region_entry, VariantUnavailable
from .variant_stability import compare_variant_text, stability_regions
from .text_alignment import comparison_text


RESTORATION_EVALUATION_VERSION = 'vtm-restoration-evaluation-v1'
REVIEW_LABELS = {'rescued', 'regressed', 'unchanged', 'uncertain'}
RESTORATION_EVALUATION_DEFAULTS = {
    'max_regions': 256, 'max_review_cases': 48, 'analysis_max_side': 1024,
    'edge_threshold': 24, 'difference_threshold': 24, 'warning_fraction': .015,
    'blind_review': False,
}


def restoration_evaluation_settings(options=None):
    options = options or {}
    if set(options) - RESTORATION_EVALUATION_DEFAULTS.keys():
        raise ValueError('Unknown restoration evaluation option')
    p = {**RESTORATION_EVALUATION_DEFAULTS, **options}
    for key in ('max_regions', 'max_review_cases', 'analysis_max_side', 'edge_threshold', 'difference_threshold'):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not 0 < p['warning_fraction'] < 1 or type(p['blind_review']) is not bool:
        raise ValueError('Invalid warning_fraction/blind_review')
    return p


def restoration_comparison_config(config, reading_order='auto'):
    """Only input variant, selection and execution location may differ in a pair."""
    excluded = {'books', 'pages', 'input_dir', 'workdir', 'run_name', 'limit',
        'worker_python', 'worker_pythonpath', 'worker_wsl_distro', 'n_gpu',
        'page_timeout', 'budget_h', 'stall_min', 'max_restarts', 'n_layout',
        'layout_budget_h', 'export', 'fresh', 'resume_from', 'image_dataset',
        'image_dataset_key', 'image_content_key', 'image_variant', 'allow_generated_variants'}
    return {'ocr': {k: v for k, v in config.items() if k not in excluded},
            'comparison_reading_order': reading_order}


def restoration_agreement(records, engines, *, options=None):
    """Engine pairs are the denominator; missing/invalid readings do not vote."""
    valid = {e: r for e, r in records.items() if e in engines and r.get('status') == 'ok'}
    distances, equal = [], 0
    for a, b in combinations(valid.values(), 2):
        d = compare_variant_text(a['text'], b['text'], options=options)
        equal += bool(d['exact_agreement'])
        if d['normalized_edit_distance'] is not None:
            distances.append(d['normalized_edit_distance'])
    n = len(valid) * (len(valid)-1)//2
    return {'available_engines': sorted(valid), 'missing_or_ineligible_engines': sorted(set(engines)-valid.keys()),
        'engine_pair_denominator': n, 'configured_engine_pairs': len(engines)*(len(engines)-1)//2,
        'exact_agreement_pairs': equal, 'exact_agreement_rate': equal/n if n else None,
        'distance_pair_denominator': len(distances),
        'mean_normalized_edit_distance': sum(distances)/len(distances) if distances else None}


def visual_fidelity(original, candidate, source_box, image_to_source, *,
                    mapping_evidence='crop_and_scale_geometry', provenance=None, options=None):
    """Align onto a bounded source lattice before measuring differences.

    The bicubic control is generated for pixel diagnostics only, never for OCR.
    Nominal geometry of generative reconstruction cannot establish alignment.
    """
    import numpy as np
    from PIL import Image, ImageFilter
    p = restoration_evaluation_settings(options)
    unavailable = {'status': 'source_geometry_mismatch', 'warnings': [], 'metrics': None}
    if provenance == 'generative_reconstruction' or mapping_evidence != 'crop_and_scale_geometry':
        return {**unavailable, 'status': 'alignment_unverified',
                'reason': 'Nominal/generative canvas is not observed stroke alignment'}
    x1, y1, x2, y2 = source_box
    if original.size != (x2-x1, y2-y1) or candidate.width < 1 or candidate.height < 1:
        return unavailable
    expected = [(x2-x1)/candidate.width, 0, 0, (y2-y1)/candidate.height, x1, y1]
    if (len(image_to_source) != 6 or any(not isinstance(a, (int, float)) or not math.isfinite(a)
            or not math.isclose(a, b, abs_tol=1e-9) for a, b in zip(image_to_source, expected))):
        return unavailable
    factor = min(1, p['analysis_max_side']/max(original.size))
    size = tuple(max(1, round(v*factor)) for v in original.size)
    def array(im):
        return np.asarray(im.convert('L').resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
    def edge(a):
        dx = np.abs(np.diff(a, axis=1, prepend=a[:, :1]))
        dy = np.abs(np.diff(a, axis=0, prepend=a[:1, :]))
        return np.maximum(dx, dy) >= p['edge_threshold']
    control = original.resize(candidate.size, Image.Resampling.BICUBIC)
    a, b, c = array(original), array(candidate), array(control)
    control.close()
    ea, eb, ec = edge(a), edge(b), edge(c)
    added, removed = eb & ~ea, ea & ~eb
    residual_added, residual_removed = eb & ~ec, ec & ~eb
    difference = np.abs(b-c)
    def local_contrast(a):
        im = Image.fromarray(a.astype('uint8'))
        blur = np.asarray(im.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
        return float(np.mean(np.abs(a-blur)))
    smooth = Image.fromarray(np.clip(difference, 0, 255).astype('uint8')).filter(ImageFilter.GaussianBlur(1))
    persistent = float((np.asarray(smooth) >= p['difference_threshold']).mean())
    warnings = []
    if residual_added.mean() > p['warning_fraction']:
        warnings.append('possible_stroke_appearance_beyond_bicubic')
    if residual_removed.mean() > p['warning_fraction']:
        warnings.append('possible_stroke_disappearance_beyond_bicubic')
    if persistent > p['warning_fraction']:
        warnings.append('persistent_local_difference_needs_source_review')
    return {'status': 'measured', 'source_box': source_box, 'analysis_size': list(size),
        'analysis_to_source': [original.width/size[0], 0, 0, original.height/size[1], x1, y1], 'warnings': warnings,
        'metrics': {'gray_mae': float(np.abs(b-a).mean()),
            'bicubic_gray_mae': float(np.abs(c-a).mean()), 'beyond_bicubic_mae': float(difference.mean()),
            'edge_appearance_fraction': float(added.mean()), 'edge_disappearance_fraction': float(removed.mean()),
            'beyond_bicubic_edge_appearance_fraction': float(residual_added.mean()),
            'beyond_bicubic_edge_disappearance_fraction': float(residual_removed.mean()),
            'persistent_difference_fraction': persistent,
            'original_local_contrast': local_contrast(a), 'variant_local_contrast': local_contrast(b)},
        'interpretation': 'Warnings only; interpolation, sharpening and noise removal also change edges. Not visual accuracy.'}


def restoration_image_ref(root, entry):
    return {'path': str(dataset_path(root, entry['path'])), 'sha256': entry['sha256'], 'size': entry['size']}


def evaluate_restoration_page(records_by_variant, manifests_by_variant, engines, *, book, page,
                              source_sha256, input_states=None, options=None, stability_options=None,
                              reading_order='auto', enable_visual=True):
    """Use existing prepared records. Every pair is local to one source and region."""
    p = restoration_evaluation_settings(options)
    if 'original' not in records_by_variant:
        raise ValueError('Restoration ablation requires explicit original; no fallback')
    result = {'schema': RESTORATION_EVALUATION_VERSION, 'book': book, 'page': page,
        'source_sha256': source_sha256, 'parameters': p, 'regions': [], 'review_cases': [],
        'limitations': ['Agreement/stability are not accuracy.', 'Variants are dependent observations.',
            'Review labels are human decisions, never inferred from these diagnostics.']}
    if not source_sha256:
        return {**result, 'status': 'source_unverified', 'complete': False}
    assets, contexts, configs, failures, dataset_contexts = {}, {}, {}, {}, {}
    for variant, manifests in manifests_by_variant.items():
        for engine, manifest in manifests.items():
            cfg = manifest['config']
            configs[(variant, engine)] = restoration_comparison_config(cfg, reading_order)
        # Canonical assets must be identical across engines for this variant.
        inputs = {(m['config'].get('image_dataset'), m['config'].get('image_dataset_key'),
                   m['config'].get('image_crop_type')) for m in manifests.values()}
        if len(inputs) != 1:
            failures[variant] = 'engine_input_dataset_or_crop_mismatch'; continue
        path, snapshot, kind = next(iter(inputs))
        if not path:
            failures[variant] = 'prepared_dataset_required'; continue
        context_key = (path, snapshot, kind)
        if context_key not in dataset_contexts:
            root, desc = dataset_open(path)
            if desc['snapshot_id'] != snapshot:
                raise ValueError('Restoration evaluation snapshot mismatch')
            record = dataset_page(root, desc, source_sha256, page-1)
            if record is None:
                failures[variant] = 'missing_source_page'; continue
            layout = dataset_validate_page(root, record)
            index, report = restoration_index(root, desc)
            dataset_contexts[context_key] = (root, desc, record, layout, index, report)
        root, desc, record, layout, index, report = dataset_contexts[context_key]
        canonical_kind = 'page' if kind == 'native' else kind
        contexts[variant] = (root, desc, record, layout, index, report)
        for region in record['regions']:
            if region['crop_type'] != canonical_kind:
                continue
            try:
                image, meta = variant_region_entry(root, desc, record, region, variant, index=index,
                    report=report, allow_generated=all(m['config'].get('allow_generated_variants')
                        for m in manifests.values()) if variant in {'difftsr', 'unihir_restoration'} else False)
                assets[(variant, region['id'])] = (region, image, meta)
            except VariantUnavailable as exc:
                assets[(variant, region['id'])] = (region, None, {'state': 'unavailable', 'details': exc.states})
    originals = sorted((key[1] for key in assets if key[0] == 'original'))
    cases = []
    for region_id in originals[:p['max_regions']]:
        region, original_entry, _ = assets[('original', region_id)]
        box = region['source_box']
        item = {'region_id': region_id, 'source_box': box, 'crop_type': region['crop_type'],
            'reading_order': region['reading_order'], 'variants': {}, 'comparisons': [], 'agreement': {},
            'joint_reading_changes': []}
        for variant in records_by_variant:
            asset = assets.get((variant, region_id))
            readings = {}
            for engine in engines:
                record = records_by_variant[variant].get(engine)
                observed = stability_regions(record).get(region_id)
                state = 'missing_output' if not record else 'region_unobserved' if not observed else 'ok'
                if not asset or not asset[1]:
                    state = 'variant_asset_unavailable'
                if observed and (observed.get('page_box') != box or not asset or asset[0]['source_box'] != box
                        or asset[0]['reading_order'] != region['reading_order']):
                    state = 'source_geometry_mismatch'
                if observed and asset and asset[1] and observed.get('variant_sha256') != asset[1]['sha256']:
                    state = 'checkpoint_pixel_identity_mismatch'
                raw = (record or {}).get('raw', {})
                dataset_info = raw.get('image_preparation', {}).get('dataset', {})
                if observed and (dataset_info.get('source_sha256') != source_sha256
                        or dataset_info.get('variant') != variant):
                    state = 'checkpoint_provenance_unverified'
                if (observed and contexts.get(variant) and contexts.get('original')
                        and (contexts[variant][2]['source'] != contexts['original'][2]['source']
                             or dataset_info.get('source') != contexts[variant][2]['source'])):
                    state = 'checkpoint_source_image_mismatch'
                if record and (record.get('quality_issues') or raw.get('hit_max') is True):
                    state = 'quality_ineligible'
                readings[engine] = {'status': state, 'text': observed.get('text') if observed else None,
                    'processor_identity': raw.get('image_preparation', {}).get('processor_identity'),
                    'prepared_size': observed.get('variant_size') if observed else None,
                    'worker_size': observed.get('input_size') if observed else None,
                    'processor_size': observed.get('processor_size') if observed else None,
                    'ocr_seconds': observed.get('ocr_seconds') if observed else None,
                    'page_inference_seconds': raw.get('inference_sec'),
                    'input_state': (input_states or {}).get(variant, {}).get(engine)}
            info = {'status': asset[2]['state'] if asset else failures.get(variant, 'missing_region'),
                    'readings': readings}
            if asset and asset[1]:
                canonical, entry, meta = asset
                info.update(image=restoration_image_ref(contexts[variant][0], entry),
                    pixel_sha256=entry['pixel_sha256'], model_identity=meta.get('model_identity'),
                    provenance=meta['provenance'], restoration_seconds=meta.get('seconds'),
                    restoration_timing='cached generation time; not current evaluation wall time')
                if (enable_visual and variant != 'original' and canonical['source_box'] == box
                        and contexts[variant][2]['source'] == contexts['original'][2]['source']):
                    with dataset_load_image(contexts['original'][0], original_entry) as a, \
                            dataset_load_image(contexts[variant][0], entry) as b:
                        info['visual_fidelity'] = visual_fidelity(a, b, box, entry['image_to_source'],
                            mapping_evidence=entry.get('coordinate_mapping_evidence', 'crop_and_scale_geometry'),
                            provenance=meta['provenance'], options=p)
            item['variants'][variant] = info
            item['agreement'][variant] = restoration_agreement(readings, engines, options=stability_options)
        for variant, info in item['variants'].items():
            if variant == 'original':
                continue
            size = (info.get('image') or {}).get('size')
            baselines = [v for v in ('bicubic_x2', 'bicubic_x4') if v != variant
                and (item['variants'].get(v, {}).get('image') or {}).get('size') == size and size]
            info['matched_bicubic'] = baselines[0] if baselines else None
            info['bicubic_status'] = 'available' if baselines else 'missing_same_resolution_bicubic'
            for baseline in ['original', *baselines[:1]]:
                for engine in engines:
                    left, right = item['variants'][baseline]['readings'][engine], info['readings'][engine]
                    pair = {'engine': engine, 'reference_variant': baseline, 'variant': variant,
                        'region_id': region_id, 'source_box': box,
                        'source_sha256': source_sha256,
                        'configuration_identity': dataset_hash(configs.get((variant, engine))),
                        'processor_identity': dataset_hash(right['processor_identity']) if right['processor_identity'] else None,
                        'reference_pixel_sha256': item['variants'][baseline].get('pixel_sha256'),
                        'variant_pixel_sha256': info.get('pixel_sha256')}
                    if configs.get((baseline, engine)) != configs.get((variant, engine)):
                        pair.update(status='inference_configuration_mismatch', changes=[])
                    elif left['status'] != 'ok' or right['status'] != 'ok':
                        pair.update(status='ineligible_pair', input_status=[left['status'], right['status']], changes=[])
                    elif left['processor_identity'] and right['processor_identity'] and left['processor_identity'] != right['processor_identity']:
                        pair.update(status='processor_runtime_identity_mismatch', changes=[])
                    else:
                        pair.update(compare_variant_text(left['text'], right['text'], options=stability_options))
                        pair['confounds'] = []
                        identity = left['processor_identity']
                        if not identity or not right['processor_identity']:
                            pair['confounds'].append('legacy_processor_runtime_identity_unobserved')
                        elif identity.get('observation') != 'observed':
                            pair['confounds'].append(identity['observation'])
                        if not left['processor_size'] or not right['processor_size']:
                            pair['confounds'].append('processor_size_unobserved')
                        elif left['processor_size'] != right['processor_size']:
                            pair['confounds'].append('different_processor_sizes')
                        if info['bicubic_status'] != 'available':
                            pair['confounds'].append(info['bicubic_status'])
                        pair['case_id'] = dataset_hash({**pair, 'left': left['text'], 'right': right['text']})
                        if baseline == 'original':
                            cases.append({'case_id': pair['case_id'], 'source_sha256': source_sha256,
                                'book': book, 'page': page, 'region_id': region_id, 'engine': engine,
                                'variant': variant, 'reference_variant': baseline,
                                'changed': pair.get('exact_agreement') is False, 'label': None})
                    item['comparisons'].append(pair)
            before, after = item['agreement']['original'], item['agreement'][variant]
            shared = sorted(e for e in set(before['available_engines']) & set(after['available_engines'])
                if configs.get(('original', e)) == configs.get((variant, e))
                and item['variants']['original']['readings'][e]['processor_identity'] == info['readings'][e]['processor_identity'])
            controlled_before = restoration_agreement(item['variants']['original']['readings'], shared, options=stability_options)
            controlled_after = restoration_agreement(info['readings'], shared, options=stability_options)
            denominator = controlled_after['engine_pair_denominator']
            info['agreement_change'] = {'engines': shared, 'engine_pair_denominator': denominator,
                'exact_rate_delta': (controlled_after['exact_agreement_rate']-controlled_before['exact_agreement_rate'])
                    if denominator else None,
                'disagreement_distance_delta': (controlled_after['mean_normalized_edit_distance']-controlled_before['mean_normalized_edit_distance'])
                    if controlled_after['mean_normalized_edit_distance'] is not None and controlled_before['mean_normalized_edit_distance'] is not None else None}
            changed = [pair for pair in item['comparisons'] if pair['variant'] == variant
                and pair['reference_variant'] == 'original' and pair.get('exact_agreement') is False]
            groups = {}
            for pair in changed:
                groups.setdefault(comparison_text(pair['variant_text']), []).append(pair['engine'])
            for text, group in groups.items():
                if len(group) >= 2:
                    item['joint_reading_changes'].append({'variant': variant, 'engines': group,
                        'reading': text, 'configured_engine_denominator': len(engines), 'correctness': 'unverified'})
        result['regions'].append(item)
    cases.sort(key=lambda c: (not c['changed'], c['case_id']))
    result.update(status='measured' if originals else 'no_canonical_regions',
        complete=len(originals) <= p['max_regions'] and not failures,
        input_complete=all(v['status'] == 'ok' and all(r['status'] == 'ok' for r in v['readings'].values())
            for region in result['regions'] for v in region['variants'].values()),
        input_failures=failures, selected_regions=len(originals), analyzed_regions=len(result['regions']),
        review_cases=cases[:p['max_review_cases']],
        sampling={'method': 'changed_first_then_case_id_within_selected_books_pages_regions',
            'eligible_case_denominator': len(cases), 'exported_cases': min(len(cases), p['max_review_cases']),
            'scope': 'Selected regions; not a representative corpus accuracy estimate'})
    return result


def read_restoration_labels(path, cases):
    """Import human labels only for exactly matching evidence identities."""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    known = {c['case_id'] for c in cases}
    labels, seen = [], set()
    if payload.get('schema') != 'vtm-restoration-review-v1':
        raise ValueError('Unsupported manual review format')
    for row in payload['labels']:
        if row['case_id'] not in known or row['case_id'] in seen or row['label'] not in REVIEW_LABELS:
            raise ValueError('Unknown/duplicate evidence identity or invalid manual label')
        seen.add(row['case_id']); labels.append(row)
    return labels


def summarize_restoration_labels(labels, *, exported_cases, sampling):
    counts = {label: sum(r['label'] == label for r in labels) for label in sorted(REVIEW_LABELS)}
    return {'counts': counts, 'labeled_denominator': len(labels), 'exported_case_denominator': exported_cases,
        'rescued_rate_among_labeled': counts['rescued']/len(labels) if labels else None,
        'regressed_rate_among_labeled': counts['regressed']/len(labels) if labels else None,
        'sampling': sampling, 'scope': 'Human reviewed sample only; not corpus accuracy'}
