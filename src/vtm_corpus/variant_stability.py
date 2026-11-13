"""Within-engine perturbation diagnostics; variants never become model votes."""
from __future__ import annotations

from collections import Counter

from .text_alignment import comparison_text, align_characters, AlignmentLimitError
from .restoration_registry import variant_spec


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


def stability_pair_summary(pairs, *, rate=True):
    measured = [c for c in pairs if c.get('exact_agreement') is not None and c.get('eligible', True)]
    same = sum(c['exact_agreement'] for c in measured)
    return dict(measured_pairs=len(measured), exact_agreement_pairs=same,
        exact_agreement_rate=same/len(measured) if rate and measured else None,
        changed_pairs=sum(c['exact_agreement'] is False for c in measured),
        unresolved_alignments=sum(c['status'] in {'ambiguous_alignment', 'analysis_incomplete'} for c in pairs),
        missing_pairs=sum(c['status'] == 'missing_output' for c in pairs),
        ineligible_pairs=sum(not c.get('eligible', True) for c in pairs),
        denominator='available eligible reference-versus-variant pairs; dependent observations, not votes')


def stability_mechanism_groups(comparisons, engines):
    groups = {}
    for group in sorted({c['method_family'] for c in comparisons}):
        pairs = [c for c in comparisons if c['method_family'] == group]
        groups[group] = dict(variants=sorted({c['variant'] for c in pairs}),
            by_engine={engine: stability_pair_summary([c for c in pairs if c['engine'] == engine]) for engine in engines},
            by_interpretation={kind: {engine: stability_pair_summary([c for c in pairs if c['engine'] == engine
                and c['perturbation_class'] == kind]) for engine in engines}
                for kind in sorted({c['perturbation_class'] for c in pairs})},
            interpretation='sensitivity, not content-preserving invariance' if group in {
                'neural_restoration', 'generative_reconstruction', 'binarization', 'denoising_sharpening'}
                else 'inspect perturbation_class; even conservative probes may alter evidence')
    return groups


def stability_joint_changes(comparisons):
    readings = {}
    for pair in comparisons:
        if not pair.get('eligible', True):
            continue
        for item in pair['regions'] or [pair]:
            if item['exact_agreement'] is not False or item['status'] not in {'changed', 'ambiguous_alignment'}:
                continue
            key = (pair['variant'], item['region_id'], comparison_text(item['variant_text']))
            readings.setdefault(key, []).append(pair['engine'])
    return [dict(variant=v, region_id=r, alternative=text, engines=sorted(set(es)),
                 status='uncertain', interpretation='shared change, not independent confirmation or correctness')
            for (v, r, text), es in readings.items() if len(set(es)) > 1]


def stability_consensus_axes(records_by_variant, comparisons, engines, *, max_regions=256):
    """Region-level whole-reading agreement, separate from character consensus."""
    region_ids = sorted({r for records in records_by_variant.values() for rec in records.values()
                         for r in stability_regions(rec)})
    output = []
    for region_id in region_ids[:max_regions] or ['page_text']:
        for group in sorted({c['method_family'] for c in comparisons}):
            pairs = [r for c in comparisons if c['method_family'] == group for r in (
                [c] if region_id == 'page_text' else c['regions']) if r['region_id'] == region_id]
            expected = len(engines)*len({c['variant'] for c in comparisons if c['method_family'] == group})
            eligible = len(pairs) == expected and bool(pairs) and all(
                c.get('eligible', True) and c['status'] in {'stable', 'changed'} for c in pairs)
            stability = 'high' if eligible and all(c['exact_agreement'] for c in pairs) else 'low' if eligible else 'uncertain'
            agreement = {}
            variants = ['original', *sorted({c['variant'] for c in comparisons if c['method_family'] == group})]
            for variant in variants:
                texts = []
                for engine in engines:
                    rec = records_by_variant.get(variant, {}).get(engine)
                    raw = (rec or {}).get('raw', {})
                    region = stability_regions(rec).get(region_id)
                    text = rec.get('text') if rec and region_id == 'page_text' else (region or {}).get('text')
                    if rec and not rec.get('quality_issues') and not raw.get('hit_max') and text:
                        texts.append(comparison_text(text))
                counts = Counter(texts)
                agreement[variant] = ('uncertain' if len(texts) < len(engines) else
                    'high' if max(counts.values(), default=0) > len(engines)/2 else 'low')
            output.append(dict(region_id=region_id, method_family=group,
                within_engine_stability=stability, across_engine_agreement=agreement, status='uncertain',
                correctness='unverified', definition='high agreement: strict majority of configured engines has identical whole-region reading; high stability: all eligible pairs unchanged'))
    return output


def analyze_variant_stability(records_by_variant, engines, *, options=None, input_states=None,
                              configuration_identities=None):
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
                pair.update(variant_spec(variant), eligible=True)
                quality = any(r and (r.get('quality_issues') or r.get('raw', {}).get('hit_max') is True) for r in (ref, rec))
                processor_ids = [(r or {}).get('raw', {}).get('image_preparation', {}).get('processor_identity')
                                 for r in (ref, rec)]
                processor_mismatch = all(processor_ids) and processor_ids[0] != processor_ids[1]
                config_mismatch = configuration_identities is not None and (
                    not configuration_identities.get(reference, {}).get(engine) or
                    configuration_identities.get(reference, {}).get(engine) != configuration_identities.get(variant, {}).get(engine))
                if quality or config_mismatch or processor_mismatch:
                    pair.update(eligible=False, status='ineligible_pair', exact_agreement=None,
                        uncertainty=['quality_ineligible' if quality else 'inference_configuration_mismatch'
                            if config_mismatch else 'processor_runtime_identity_mismatch'])
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
                        diagnostic.update(status='source_geometry_mismatch', changes=[], exact_agreement=None, eligible=False)
                        pair.update(eligible=False)
                    if not pair['eligible']:
                        diagnostic.update(eligible=False, exact_agreement=None)
                    pair['regions'].append(diagnostic)
                pair['regions_complete'] = len(region_ids) <= p['max_regions']
                pair['confounds'] = []
                if not all(processor_ids):
                    pair['confounds'].append('processor_runtime_identity_unobserved; controlled_comparison_uncertain')
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
    summary = stability_pair_summary(comparisons, rate=False)
    summary['rate_scope'] = 'no pooled rate across mechanisms; see groups by_engine and by_interpretation'
    regional = [{**r, **{k: c[k] for k in ('engine', 'variant', 'method_family', 'perturbation_class')}}
                for c in comparisons for r in c['regions']]
    return {'schema': 'vtm-variant-stability-v2', 'reference_variant': reference, 'matrix': matrix,
        'comparisons': comparisons, 'risks': risks, 'summary': summary, 'parameters': p,
        'groups': stability_mechanism_groups(comparisons, engines),
        'region_groups': stability_mechanism_groups(regional, engines),
        'group_units': {'groups': 'page text pairs', 'region_groups': 'observed canonical region text pairs'},
        'engine_sensitivity_profiles': {e: {g: [c['variant'] for c in comparisons if c['engine'] == e
            and c['method_family'] == g and c.get('eligible') and c['exact_agreement'] is False]
            for g in sorted({c['method_family'] for c in comparisons})} for e in engines},
        'sensitive_text_positions': [{**change, 'engine': c['engine'], 'variant': c['variant'],
            'region_id': r['region_id'], 'status': 'uncertain; text offsets are not glyph coordinates'}
            for c in comparisons if c.get('eligible') for r in c['regions'] for change in r['changes']],
        'joint_changes': stability_joint_changes(comparisons),
        'consensus_stability_axes': stability_consensus_axes(records_by_variant, comparisons, engines, max_regions=p['max_regions']),
        'vote_policy': 'within_engine_pairs_only; across_engine_consensus_separate_per_variant',
        'limitations': ['Stability and edit distance are not accuracy, CER or WER.',
                       'Dependent variants do not increase the number of independent engines.']}
