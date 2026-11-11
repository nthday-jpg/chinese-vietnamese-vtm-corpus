"""Within-engine perturbation diagnostics; variants never become model votes."""
from __future__ import annotations

from collections import Counter

from .text_alignment import comparison_text, align_characters, AlignmentLimitError


STABILITY_DEFAULTS = {'reference_variant': 'original', 'compare_by_engine': True,
    'compare_across_engines': True, 'save_difference_crops': True,
    'max_alignment_cells': 8_000_000, 'max_text_chars': 20000,
    'max_difference_crops': 12, 'max_changes': 256, 'max_regions': 256}


def stability_settings(options=None):
    if set(options or {}) - STABILITY_DEFAULTS.keys():
        raise ValueError('Unknown variant stability options')
    p = {**STABILITY_DEFAULTS, **(options or {})}
    for key in ('max_alignment_cells', 'max_text_chars', 'max_changes', 'max_regions'):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError(f'Invalid stability budget: {key}')
    if type(p['max_difference_crops']) is not int or p['max_difference_crops'] < 0:
        raise ValueError('Invalid difference crop limit')
    return p


def stability_order_change(left, right, *, width=4, max_anchors=64):
    anchors = []
    for start in range(0, max(0, len(left)-width+1), width):
        token = left[start:start+width]
        position = right.find(token)
        if left.count(token) == 1 and position >= 0 and right.find(token, position+1) < 0:
            anchors.append((start, position))
        if len(anchors) >= max_anchors:
            break
    return len(anchors) >= 3 and any(a[1] > b[1] for a, b in zip(anchors, anchors[1:]))


def compare_variant_text(reference, candidate, *, options=None):
    p = stability_settings(options)
    # Only whitespace is removed. Raw text is always retained separately.
    result = {'reference_text': reference, 'variant_text': candidate, 'normalization': 'remove_whitespace_only',
        'status': 'missing_output', 'exact_agreement': None, 'normalized_edit_distance': None,
        'substitutions': None, 'insertions': None, 'deletions': None, 'changes': [], 'uncertainty': []}
    if reference is None or candidate is None:
        return result
    a, b = comparison_text(reference), comparison_text(candidate)
    result.update(exact_agreement=a == b, reference_chars=len(a), variant_chars=len(b))
    if max(len(a), len(b)) > p['max_text_chars']:
        return {**result, 'status': 'analysis_incomplete', 'uncertainty': ['max_text_chars']}
    try:
        alignment = align_characters(a, b, max_cells=p['max_alignment_cells'])
    except AlignmentLimitError:
        return {**result, 'status': 'analysis_incomplete', 'uncertainty': ['max_alignment_cells']}
    distance = sum(x != y for x, y in alignment)
    result['normalized_edit_distance'] = distance/max(1, len(a), len(b))
    reordered = stability_order_change(a, b) if a != b else False
    result['reading_order_disagreement'] = 'suspected' if reordered else 'not_detected'
    # Repeated deletion/insertion positions and long changed spans lack unique anchors.
    changed_chars = {x or y for x, y in alignment if x != y and (not x or not y)}
    repeated = any(a.count(c) > 1 or b.count(c) > 1 for c in changed_chars)
    long_change = distance > 8 and result['normalized_edit_distance'] > .35
    if reordered or repeated or long_change:
        result.update(status='ambiguous_alignment', uncertainty=[reason for reason, condition in (
            ('repeated_character_gap_positions', repeated), ('suspected_reading_order', reordered),
            ('long_changed_span', long_change)) if condition])
        return result
    result.update(status='stable' if a == b else 'changed', substitutions=0, insertions=0, deletions=0)
    i, j = 0, 0
    for x, y in alignment:
        if x != y:
            kind = 'insertions' if not x else 'deletions' if not y else 'substitutions'
            result[kind] += 1
            if len(result['changes']) < p['max_changes']:
                result['changes'].append({'kind': kind, 'reference_offset': i if x else None,
                    'variant_offset': j if y else None, 'reference': x, 'alternative': y,
                    'coordinate_status': 'text_offset_only'})
        i, j = i+len(x), j+len(y)
    result['changes_complete'] = distance <= p['max_changes']
    return result


def stability_regions(record):
    if record is None:
        return {}
    raw = record.get('raw', record)
    regions = raw.get('image_preparation', {}).get('regions', [])
    return {r['canonical_region_id']: r for r in regions if r.get('canonical_region_id')}


def analyze_variant_stability(records_by_variant, engines, *, options=None, input_states=None):
    """Input: variant -> engine -> prepared checkpoint or None, for one page."""
    p = stability_settings(options)
    reference = p['reference_variant']
    if reference not in records_by_variant:
        raise ValueError('Reference variant must be explicitly selected; no substitution')
    matrix, comparisons, risks = {}, [], []
    for variant, records in records_by_variant.items():
        matrix[variant] = {}
        for engine in engines:
            rec = records.get(engine)
            raw = (rec or {}).get('raw', rec or {})
            matrix[variant][engine] = {'status': 'ok' if rec else 'missing_output',
                'text': rec['text'] if rec else None, 'hit_max': raw.get('hit_max'),
                'quality_issues': (rec or {}).get('quality_issues', []),
                'regions': [{k: r.get(k) for k in ('canonical_region_id', 'page_box', 'variant_size', 'input_size',
                    'processor_size', 'processor_scale_xy', 'processor_downsampled', 'variant_sha256',
                    'provenance', 'model_identity', 'operation', 'input_to_source', 'coordinate_mapping_evidence')}
                    for r in raw.get('image_preparation', {}).get('regions', [])],
                'processor_observation': raw.get('model_input', {}).get('observation',
                    raw.get('image_preparation', {}).get('processor_observation', 'unavailable'))}
            if not rec and input_states:
                matrix[variant][engine]['input_status'] = input_states.get(variant, {}).get(engine)
    if p['compare_by_engine']:
        for engine in engines:
            ref = records_by_variant[reference].get(engine)
            ref_regions = stability_regions(ref)
            for variant, records in records_by_variant.items():
                if variant == reference:
                    continue
                rec = records.get(engine)
                pair = compare_variant_text(ref['text'] if ref else None, rec['text'] if rec else None, options=p)
                pair.update(engine=engine, reference_variant=reference, variant=variant, region_id='page_text', regions=[])
                regions = stability_regions(rec)
                region_ids = sorted(ref_regions.keys() | regions.keys())
                for region_id in region_ids[:p['max_regions']]:
                    a, b = ref_regions.get(region_id), regions.get(region_id)
                    diagnostic = compare_variant_text(a['text'] if a else None, b['text'] if b else None, options=p)
                    diagnostic.update(region_id=region_id, source_box=(a or b)['page_box'],
                        reference_input=a.get('input_size') if a else None, variant_input=b.get('input_size') if b else None,
                        reference_processor=a.get('processor_size') if a else None,
                        variant_processor=b.get('processor_size') if b else None)
                    if a and b:
                        old_geometry = sum(len(a.get('raw', {}).get(k, [])) for k in ('character_boxes', 'text_lines', 'blocks', 'spatial_regions'))
                        new_geometry = sum(len(b.get('raw', {}).get(k, [])) for k in ('character_boxes', 'text_lines', 'blocks', 'spatial_regions'))
                        if old_geometry and not new_geometry:
                            risks.append({'kind': 'localized_evidence_disappeared_after_transformation',
                                'engine': engine, 'variant': variant, 'region_id': region_id,
                                'reason': 'Detector/adapter change; not proof of missing text'})
                    if a and b and a['page_box'] != b['page_box']:
                        diagnostic.update(status='source_geometry_mismatch', changes=[])
                    pair['regions'].append(diagnostic)
                pair['regions_complete'] = len(region_ids) <= p['max_regions']
                pair['confounds'] = []
                if not pair['regions'] or any(not r['reference_processor'] or not r['variant_processor'] for r in pair['regions']):
                    pair['confounds'].append('backend_processor_sizes_unobserved')
                if any(r['reference_processor'] and r['variant_processor'] and
                       r['reference_processor'] != r['variant_processor'] for r in pair['regions']):
                    pair['confounds'].append('different_internal_pixel_budgets; restoration_effect_not_isolated')
                comparisons.append(pair)
                if pair['exact_agreement'] is False:
                    risks.append({'kind': 'suspected_transformation_associated_reading_change', 'engine': engine,
                                  'variant': variant, 'region_id': 'page_text', 'confirmed_false_stroke': False})
                    if pair.get('insertions') or pair.get('deletions'):
                        risks.append({'kind': 'recognized_text_appeared_or_disappeared', 'engine': engine, 'variant': variant})
    original_texts = [matrix[reference][e]['text'] for e in engines]
    if all(t is not None for t in original_texts) and len({comparison_text(t) for t in original_texts}) == 1:
        for variant in matrix:
            if variant != reference and any(v['text'] is not None and comparison_text(v['text']) != comparison_text(original_texts[0])
                                            for v in matrix[variant].values()):
                risks.append({'kind': 'original_engine_agreement_changed_after_transformation', 'variant': variant})
    for engine in engines:
        texts = [matrix[v][engine]['text'] for v in matrix]
        counts = Counter(comparison_text(t) for t in texts if t is not None)
        if len(counts) > 1 and sum(counts.values()) >= 3:
            for variant in matrix:
                text = matrix[variant][engine]['text']
                if text is not None and counts[comparison_text(text)] == 1:
                    risks.append({'kind': 'reading_unique_to_one_variant', 'engine': engine, 'variant': variant})
        if len(texts) > 1 and all(t is not None for t in texts) and len({comparison_text(t) for t in texts}) == 1:
            risks.append({'kind': 'shared_reading_unverified_across_variants', 'engine': engine,
                          'reason': 'Stability cannot rule out shared errors or omissions'})
    measured = [c for c in comparisons if c['exact_agreement'] is not None]
    summary = {'measured_pairs': len(measured), 'exact_agreement_pairs': sum(c['exact_agreement'] for c in measured),
        'exact_agreement_rate': sum(c['exact_agreement'] for c in measured)/len(measured) if measured else None,
        'changed_pairs': sum(c['exact_agreement'] is False for c in measured),
        'unresolved_alignments': sum(c['status'] in {'ambiguous_alignment', 'analysis_incomplete'} for c in comparisons),
        'missing_pairs': sum(c['status'] == 'missing_output' for c in comparisons)}
    return {'schema': 'vtm-variant-stability-v1', 'reference_variant': reference, 'matrix': matrix,
        'comparisons': comparisons, 'risks': risks, 'summary': summary, 'parameters': p,
        'vote_policy': 'within_engine_pairs_only; across_engine_consensus_separate_per_variant',
        'limitations': ['Stability and edit distance are not accuracy, CER or WER.',
                       'Dependent variants do not increase the number of independent engines.']}
