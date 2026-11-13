"""Opt-in scheduling and evidence enrichment of the existing four-engine campaign."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import OCRConfig
from .corpus_ocr import prepare_multi_corpus, write_json
from .image_dataset import dataset_open, dataset_page, dataset_validate_page, dataset_load_image, dataset_file_hash, dataset_path, dataset_hash
from .prepared_variants import OCR_IMAGE_VARIANTS, variant_region_entry
from .multi_compare import compare_multi_corpus
from .ocr_records import load_run, read_checkpoint_campaign
from .pdf_images import acquire_pdf_image
from .spatial_coverage import analyze_spatial_coverage, spatial_settings
from .variant_stability import analyze_variant_stability, stability_settings, stability_mechanism_groups
from .restoration_evaluation import evaluate_restoration_page, restoration_evaluation_settings, restoration_comparison_config
from .restoration_registry import SURVEY_SOURCE_GROUPS


def prepare_variant_corpus(models, variants, *, profile_dir=None, max_books=1, cache_renders=True):
    if (not isinstance(variants, list) or not variants or len(set(variants)) != len(variants)
            or set(variants)-OCR_IMAGE_VARIANTS.keys() or 'original' not in variants):
        raise ValueError('OCR_VARIANTS requires original and unique supported OCR images; masks are excluded')
    if any(not c.image_dataset or c.image_dataset_fallback for c in models.values()):
        raise ValueError('Multi-variant OCR requires PREPARED_IMAGES and PREPARED_MISSING_FALLBACK=False')
    root = Path(next(iter(models.values())).workdir).resolve()
    combined = {'schema': 'vtm-multi-variant-campaign-v1', 'models': list(models),
        'variants': variants, 'variant_manifests': {}, 'pairs': [], 'total_pages': 0,
        'total_books': 0, 'cache_renders': cache_renders, 'render_cache': str(root/'render_cache'),
        'vote_policy': 'four_engines_per_variant; variants_are_dependent'}
    for variant in variants:
        configs = {engine: OCRConfig(**{**cfg.to_dict(), 'image_variant': variant,
            'workdir': str(root if variant == 'original' else root/'variant_runs'/variant)}) for engine, cfg in models.items()}
        manifest = prepare_multi_corpus(configs, profile_dir=profile_dir, max_books=max_books,
            cache_renders=cache_renders, allow_unavailable_variants=True)
        campaign = json.loads(manifest.read_text(encoding='utf8'))
        combined['variant_manifests'][variant] = str(manifest)
        combined['pairs'].extend({**pair, 'variant': variant} for pair in campaign['pairs'])
        combined['total_pages'] += campaign['total_pages']
        combined['total_books'] = max(combined['total_books'], campaign['total_books'])
    path = root/'multi_variant_manifest.json'
    write_json(path, combined)
    return path


def evidence_source_image(row, manifests):
    """Verified canonical original, or the validated visible-PDF acquisition path."""
    book, page = row['book'], row['page']-1
    info = next(iter(manifests.values()))
    source_info = info.get('books', {}).get(book, {})
    expected = source_info.get('sha256')
    if source_info.get('legacy_source_unverified') or not expected or dataset_file_hash(row['source']) != expected:
        raise ValueError('Source SHA-256/provenance not verified')
    for manifest in manifests.values():
        cfg = manifest['config']
        if cfg.get('image_dataset'):
            root, descriptor = dataset_open(cfg['image_dataset'])
            if descriptor['snapshot_id'] != cfg['image_dataset_key']:
                raise ValueError('Canonical snapshot changed')
            record = dataset_page(root, descriptor, expected, page)
            if record is None:
                raise ValueError('Canonical source page unavailable')
            layout = dataset_validate_page(root, record)
            image = dataset_load_image(root, record['source'])
            return image, {'book': book, 'page': page+1, 'source_sha256': expected,
                'image_sha256': record['source']['sha256'], 'size': list(image.size),
                'image_reference': str(dataset_path(root, record['source']['path'])),
                'snapshot_id': descriptor['snapshot_id'], 'provenance': 'source_observation',
                'legacy_frame_verified': False, 'reading_direction': layout['reading_direction']}
    import pypdfium2 as pdfium
    cfg = info['config']
    doc = pdfium.PdfDocument(row['source'])
    try:
        doc.init_forms()
        pdf_page = doc[page]
        try:
            image, acquisition = acquire_pdf_image(pdf_page, dpi=cfg.get('dpi', 200))
        finally:
            pdf_page.close()
    finally:
        doc.close()
    # Legacy renderer uses the same visible page frame, with proportional resizing.
    # Annotations/forms were not requested there; do not assert equivalence for them.
    equivalent = acquisition['annotations'] == 0
    return image, {'book': book, 'page': page+1, 'source_sha256': expected,
        'image_sha256': hashlib.sha256(image.tobytes()).hexdigest(), 'size': list(image.size),
        'image_hash_kind': 'rgb_pixels', 'provenance': 'source_observation', 'acquisition': acquisition,
        'legacy_frame_verified': equivalent, 'reading_direction': cfg.get('reading_direction', 'vertical_rtl')}


def evidence_save_image(directory, name, image):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/name
    temporary = path.with_suffix('.png.tmp')
    image.save(temporary, format='PNG')
    temporary.replace(path)
    return {'path': str(path.resolve()), 'sha256': dataset_file_hash(path), 'size': list(image.size)}


def spatial_artifacts(image, diagnostic, mask, directory):
    from PIL import Image, ImageDraw
    artifacts = {}
    p = diagnostic['parameters']
    if p['save_overlays']:
        for view in ('candidates', 'ocr_geometry', 'coverage'):
            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)
            width = max(1, image.width//600)
            if view != 'candidates':
                for box in diagnostic['ocr_evidence']:
                    polygon = [tuple(p) for p in box['polygon']]
                    draw.line(polygon+[polygon[0]], fill='#008b8b', width=width)
            if view != 'ocr_geometry':
                for c in diagnostic['findings']:
                    color = ('#1464f4' if view == 'candidates' else '#ef3030' if c['kind'] == 'potential_uncovered_text_region'
                             else '#d68b00' if c['kind'] in {'ambiguous_visual_candidate', 'geometry_unavailable', 'partially_covered_region'}
                             else '#1464f4')
                    draw.rectangle(c['bbox'], outline=color, width=width)
            artifacts[view] = evidence_save_image(directory, view+'.png', overlay)
            overlay.close()
    if p['save_candidate_masks']:
        artifacts['source_analysis_mask'] = evidence_save_image(directory, 'mask.png', Image.fromarray(mask*255))
    priority = [c for c in diagnostic['findings'] if c['kind'] in {
        'potential_uncovered_text_region', 'partially_covered_region', 'ambiguous_visual_candidate', 'geometry_unavailable',
        'source_geometry_mismatch'}]
    for c in priority[:p['max_detail_crops']]:
        box = c['bbox']
        import math
        crop = image.crop((math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])))
        c['detail_crop'] = evidence_save_image(directory, c['id']+'.png', crop)
        crop.close()
    return artifacts


def variant_difference_artifacts(image, stability, manifests_by_variant, row, directory):
    """Canonical crop comparisons only; never invent glyph coordinates for text offsets."""
    if not stability['parameters']['save_difference_crops']:
        return
    limit = stability['parameters']['max_difference_crops']
    saved = {}
    for pair in stability['comparisons']:
        for region in pair['regions']:
            if region['exact_agreement'] is not False or region['status'] == 'source_geometry_mismatch':
                continue
            key = (pair['variant'], region['region_id'])
            if key in saved:
                region['image_comparison'] = saved[key]
                continue
            if len(saved) >= limit:
                region['image_comparison_status'] = 'crop_limit'
                continue
            cfg = OCRConfig(**manifests_by_variant[pair['variant']][pair['engine']]['config'])
            if not cfg.image_dataset:
                continue
            root, descriptor = dataset_open(cfg.image_dataset)
            rec = dataset_page(root, descriptor, dataset_file_hash(row['source']), row['page']-1)
            canonical = next((r for r in rec['regions'] if r['id'] == region['region_id']), None)
            if not canonical or canonical['source_box'] != region['source_box']:
                continue
            entry, meta = variant_region_entry(root, descriptor, rec, canonical, pair['variant'],
                                               allow_generated=cfg.allow_generated_variants)
            processed = dataset_load_image(root, entry)
            original = image.crop(canonical['source_box'])
            name = hashlib.sha256(str(key).encode()).hexdigest()[:16]
            saved[key] = {'granularity': 'canonical_crop_context; not aligned_glyphs', 'source_box': canonical['source_box'],
                'original': evidence_save_image(directory, name+'_original.png', original),
                'variant': evidence_save_image(directory, name+'_variant.png', processed),
                'provenance': meta['provenance'], 'variant_sha256': entry['sha256']}
            region['image_comparison'] = saved[key]
            processed.close(); original.close()


def compare_evidence_corpus(manifest_path, *, enable_multi_variant=False, enable_spatial=False,
                             spatial_options=None, stability_options=None, enable_restoration_ablation=False,
                             enable_visual_fidelity=False, restoration_options=None, prepared_dataset=None,
                             evidence_dir=None, book_groups=None, **comparison_options):
    manifest_path = Path(manifest_path)
    campaign = read_checkpoint_campaign(manifest_path)
    p = stability_settings(stability_options)
    spatial_settings(spatial_options)
    restoration_evaluation_settings(restoration_options)
    if (enable_restoration_ablation or enable_visual_fidelity) and not enable_multi_variant:
        raise ValueError('Restoration evaluation requires an existing multi-variant campaign')
    if enable_multi_variant:
        if campaign.get('schema') != 'vtm-multi-variant-campaign-v1':
            raise ValueError('Multi-variant comparison requires multi_variant_manifest.json')
        paths = campaign['variant_manifests']
        if p['reference_variant'] not in paths:
            raise ValueError('Reference variant missing from saved campaign')
    else:
        paths = {'original': str(manifest_path)}
    results = {v: compare_multi_corpus(path, **comparison_options) for v, path in paths.items()}
    if prepared_dataset:
        # Read-only relocation for evaluation. Raw run manifests/checkpoints stay intact.
        root, descriptor = dataset_open(prepared_dataset)
        for res in results.values():
            for manifests in res['manifests'].values():
                for manifest in manifests.values():
                    manifest['config'] = {**manifest['config'], 'image_dataset': str(root/f"snapshots/{descriptor['snapshot_id']}.json"),
                        'image_dataset_key': descriptor['snapshot_id']}
    reference = p['reference_variant'] if enable_multi_variant else 'original'
    result = results[reference]
    loaded, states = {}, {}
    for v, path in paths.items():
        child = read_checkpoint_campaign(path)
        loaded[v], states[v] = {}, {}
        for pair in child['pairs']:
            loaded[v][pair['book']], states[v][pair['book']] = {}, {}
            for e in child['models']:
                loaded[v][pair['book']][e] = load_run(Path(pair[e]), e, [], reading_order=comparison_options.get('reading_order', 'auto'))
                tasks = Path(pair[e])/'tasks.jsonl'
                states[v][pair['book']][e] = {t['page']: t.get('prepared_input') for t in
                    (json.loads(line) for line in tasks.read_text(encoding='utf8').splitlines() if line)}
    result['evidence_schema'] = 'vtm-evidence-v1'
    if enable_multi_variant:
        result['variant_results'] = {v: r for v, r in results.items() if v != reference}
        result['selected_variants'] = list(paths)
    corpus_summaries = []
    for row in result['rows']:
        book, page = row['book'], row['page']-1
        records = {v: {e: run['records'].get((book, page)) for e, run in books[book].items()}
                   for v, books in loaded.items()}
        artifact_root = Path(evidence_dir) if evidence_dir else manifest_path.parent/'evidence'
        directory = artifact_root/hashlib.sha256(book.encode()).hexdigest()[:12]/f'p{page+1:05d}'
        if enable_multi_variant:
            row['variant_stability'] = analyze_variant_stability(records, result['models'], options=p,
                input_states={v: {e: pages.get(page) for e, pages in books[book].items()} for v, books in states.items()},
                configuration_identities={v: {e: dataset_hash(restoration_comparison_config(m['config'],
                    comparison_options.get('reading_order', 'auto'))) for e, m in res['manifests'][book].items()}
                    for v, res in results.items()})
            source_info = next(iter(result['manifests'][book].values())).get('books', {}).get(book, {})
            row['variant_stability']['source'] = {'book': book, 'page': page+1,
                'source_sha256': source_info.get('sha256'), 'source_verified': row['source_verified']}
            row['variant_stability']['book_group'] = ((book_groups or {}).get(book) or
                SURVEY_SOURCE_GROUPS.get(source_info.get('sha256'), 'unclassified'))
            if p['compare_across_engines']:
                row['variant_consensus'] = {v: next(r for r in res['rows'] if r['book'] == book and r['page'] == page+1)
                                          for v, res in results.items() if v != reference}
        if enable_restoration_ablation or enable_visual_fidelity:
            source_info = next(iter(result['manifests'][book].values())).get('books', {}).get(book, {})
            # Saved source hashes permit offline evaluation without re-reading PDF pixels.
            # Legacy caches without established provenance remain explicitly unverified.
            source_sha = source_info.get('sha256') if not source_info.get('legacy_source_unverified') else None
            try:
                row['restoration_evaluation'] = evaluate_restoration_page(records,
                    {v: res['manifests'][book] for v, res in results.items()}, result['models'],
                    book=book, page=page+1, source_sha256=source_sha,
                    input_states={v: {e: pages.get(page) for e, pages in books[book].items()} for v, books in states.items()},
                    options=restoration_options, stability_options=p,
                    reading_order=comparison_options.get('reading_order', 'auto'), enable_visual=enable_visual_fidelity)
            except (OSError, ValueError, KeyError, ImportError) as exc:
                row['restoration_evaluation'] = {'schema': 'vtm-restoration-evaluation-v1',
                    'status': 'analysis_incomplete', 'complete': False, 'reason': str(exc),
                    'regions': [], 'review_cases': []}
        if not enable_spatial and not (enable_multi_variant and p['save_difference_crops']):
            continue
        try:
            image, source = evidence_source_image(row, result['manifests'][book])
            try:
                if enable_spatial:
                    diagnostic, mask = analyze_spatial_coverage(image, records[reference], source,
                        options=spatial_options, reading_direction=source['reading_direction'])
                    diagnostic['artifacts'] = spatial_artifacts(image, diagnostic, mask, directory)
                    diagnostic['linked_evidence'] = {'page_consensus': '#page-consensus', 'character_consensus': '#character-consensus'}
                    row['spatial_coverage'] = diagnostic
                    corpus_summaries.append(diagnostic['summary'])
                if enable_multi_variant:
                    variant_difference_artifacts(image, row['variant_stability'],
                        {v: r['manifests'][book] for v, r in results.items()}, row, directory)
            finally:
                image.close()
        except (OSError, ValueError, KeyError, ImportError) as exc:
            if enable_spatial:
                row['spatial_coverage'] = {'schema': 'vtm-spatial-coverage-v1', 'status': 'analysis_incomplete',
                    'complete': False, 'findings': [], 'reason': str(exc), 'summary': {}}
            if enable_multi_variant:
                row['variant_stability']['image_evidence_status'] = 'unavailable: '+str(exc)
    if enable_spatial:
        denominator = sum(s['candidate_ink_pixels'] for s in corpus_summaries if s['overlapped_candidate_ink_pixels'] is not None)
        numerator = sum(s['overlapped_candidate_ink_pixels'] for s in corpus_summaries if s['overlapped_candidate_ink_pixels'] is not None)
        result['spatial_summary'] = {'pages_analyzed': len(corpus_summaries), 'pages_selected': len(result['rows']),
            'candidates': sum(s['candidates'] for s in corpus_summaries),
            'potential_uncovered': sum(s['potential_uncovered'] for s in corpus_summaries),
            'manual_review': sum(s['manual_review'] for s in corpus_summaries),
            'candidate_ink_overlap_ratio': numerator/denominator if denominator else None,
            'denominator': 'candidate analysis ink pixels on pages with usable geometry; varying analysis scales',
            'candidate_ink_pixels_with_geometry': denominator}
    for row in result['rows']:
        spatial = row.get('spatial_coverage', {})
        stability = row.get('variant_stability', {})
        if spatial and (not spatial.get('complete') or spatial.get('summary', {}).get('manual_review', 0)):
            row['reasons'].append('spatial_coverage_review')
        if stability and (stability['summary']['changed_pairs'] or stability['summary']['missing_pairs']
                          or stability['summary']['unresolved_alignments']):
            row['reasons'].append('variant_stability_review')
        if row['reasons']:
            row['route'] = 'review_page'
        restoration = row.get('restoration_evaluation')
        if restoration and (not restoration.get('complete') or not restoration.get('input_complete') or any(
                pair.get('exact_agreement') is False or pair.get('status') in {
                    'ineligible_pair', 'inference_configuration_mismatch', 'processor_runtime_identity_mismatch',
                    'ambiguous_alignment', 'analysis_incomplete'}
                for region in restoration['regions'] for pair in region['comparisons']) or any(
                v.get('visual_fidelity', {}).get('warnings') or (v.get('visual_fidelity') and
                    v['visual_fidelity']['status'] != 'measured') for region in restoration['regions']
                for v in region['variants'].values())):
            row['reasons'].append('restoration_evaluation_review')
            row['route'] = 'review_page'
    if enable_multi_variant:
        summaries = [r['variant_stability']['summary'] for r in result['rows']]
        result['stability_summary'] = {k: sum(s[k] for s in summaries) for k in
            ('measured_pairs', 'exact_agreement_pairs', 'changed_pairs', 'unresolved_alignments', 'missing_pairs')}
        s = result['stability_summary']
        s['exact_agreement_rate'] = None
        s['rate_scope'] = 'grouped_by_mechanism_and_engine_only'
        grouped, seen = {}, set()
        for row in result['rows']:
            d = row['variant_stability']; sha = d['source']['source_sha256']
            identity = (sha, row['page']) if sha else (row['book'], row['page'])
            if identity in seen:
                continue
            seen.add(identity)
            grouped.setdefault(d['book_group'], []).append(d)
        s['by_book_group'] = {g: dict(unique_source_pages=len(ds),
            groups=stability_mechanism_groups([c for d in ds for c in d['comparisons']], result['models']),
            region_groups=stability_mechanism_groups([{**r, **{k:c[k] for k in (
                'engine','variant','method_family','perturbation_class')}} for d in ds
                for c in d['comparisons'] for r in c['regions']], result['models']))
            for g, ds in grouped.items()}
        s['groups'] = stability_mechanism_groups([c for ds in grouped.values() for d in ds
            for c in d['comparisons']], result['models'])
        s['book_group_policy'] = 'analyst override or observed source SHA256; aliases deduplicated by source/page; unknown hashes unclassified'
    return result


def show_page_evidence(result, book, page, *, max_findings=12):
    """Optional in-notebook preview, using only already generated, hashed images."""
    from IPython.display import display
    from PIL import Image
    row = next(r for r in result['rows'] if r['book'] == book and r['page'] == page)
    spatial = row.get('spatial_coverage')
    if spatial:
        print('Spatial Coverage:', spatial['status'], spatial.get('summary', {}))
        ref = spatial.get('artifacts', {}).get('coverage')
        if ref:
            if dataset_file_hash(ref['path']) != ref['sha256']:
                raise ValueError('Coverage preview hash mismatch')
            with Image.open(ref['path']) as im:
                display(im.copy())
        for finding in spatial.get('findings', [])[:max_findings]:
            print(finding['id'], finding['kind'], finding['bbox'], finding['uncertainty'])
    stability = row.get('variant_stability')
    if stability:
        print('Variant Stability:', stability['summary'])
        for pair in stability['comparisons'][:max_findings]:
            print(pair['engine'], pair['variant'], pair['status'], pair['normalized_edit_distance'], pair['uncertainty'])
