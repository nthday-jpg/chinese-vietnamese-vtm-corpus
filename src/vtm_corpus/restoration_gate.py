"""Read original-only OCR checkpoints to measure six pairwise region distances."""
from __future__ import annotations

import json
from pathlib import Path

from .image_dataset import dataset_open, dataset_read
from .ocr_records import read_checkpoint_campaign, load_run
from .text_alignment import comparison_text


def restoration_edit_distance(left, right):
    """Exact unit-cost Levenshtein via bit vectors; no quadratic traceback."""
    if len(left) > len(right):
        left, right = right, left
    if not left:
        return len(right)
    masks = {}
    for i, char in enumerate(left):
        masks[char] = masks.get(char, 0) | (1 << i)
    positive, negative, score = ~0, 0, len(left)
    last = 1 << (len(left)-1)
    for char in right:
        equal = masks.get(char, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        add = negative | ~(horizontal | positive)
        remove = positive & horizontal
        score += bool(add & last) - bool(remove & last)
        add = (add << 1) | 1
        positive = (remove << 1) | ~(vertical | add)
        negative = add & vertical
    return score


def restoration_pairwise_scores(observations, *, min_characters=8):
    if type(min_characters) is not int or min_characters < 1:
        raise ValueError('min_characters must be a positive integer')
    engines = ('ppocr', 'tonggu', 'paddle', 'hunyuan')
    texts = {}
    for engine in engines:
        entry = observations.get(engine)
        if not entry or entry.get('raw', {}).get('hit_max') is True:
            return dict(state='incomplete_ocr', reason='missing, ambiguous or truncated original OCR')
        text = comparison_text(entry['text'])
        if len(text) < min_characters:
            return dict(state='insufficient_text', reason='each model needs at least %d visible characters' % min_characters)
        texts[engine] = text
    pairs = {}
    for i, left in enumerate(engines):
        for right in engines[i+1:]:
            distance = restoration_edit_distance(texts[left], texts[right])
            pairs[f'{left}:{right}'] = dict(distance=distance,
                denominator=max(len(texts[left]), len(texts[right])),
                rate=distance/max(len(texts[left]), len(texts[right])))
    return dict(state='ok', pairs=pairs, min_pairwise_rate=min(p['rate'] for p in pairs.values()),
                mean_pairwise_rate=sum(p['rate'] for p in pairs.values())/6,
                lengths={e:len(t) for e,t in texts.items()})


def collect_restoration_disagreements(manifest_path, prepared_dataset, *, min_characters=8):
    """Require matching PDF, canonical region, box and original PNG hash in all four runs.

    Aliases of an identical PDF share one decision. Conflicting alias observations
    are rejected rather than depending on filesystem iteration order.
    """
    root, descriptor = dataset_open(prepared_dataset)
    canonical = {}
    for page in descriptor['pages']:
        record = json.loads(dataset_read(root, page['record']))
        for region in record['regions']:
            key = f"{page['source_sha256']}:{page['page']}:{region['id']}"
            item = canonical.setdefault(key, dict(image_sha256=region['images']['original']['sha256'],
                source_box=region['source_box'], crop_type=region['crop_type'],
                size=region['images']['original']['size'], source_sha256=page['source_sha256'],
                page=page['page']+1, region_id=region['id'], aliases=[]))
            item['aliases'].append(dict(book=page['book'],page=page['page']+1,region_id=region['id']))
    observations, ambiguous, visited, warnings = {}, set(), set(), []

    def visit(path):
        path = Path(path).resolve()
        if path in visited:
            return
        visited.add(path)
        campaign = read_checkpoint_campaign(path)
        if set(campaign['models']) != {'ppocr', 'tonggu', 'paddle', 'hunyuan'}:
            raise ValueError('Restoration gating requires exactly four OCR models')
        if campaign.get('crop_manifests'):
            for ref in campaign['crop_manifests'].values():
                visit(ref)
            return
        if campaign.get('variant_manifests'):
            visit(campaign['variant_manifests']['original'])
            return
        for pair in campaign['pairs']:
            for engine in campaign['models']:
                run = load_run(Path(pair[engine]), engine, warnings)
                if run['manifest']['config'].get('image_variant') != 'original':
                    raise ValueError('Restoration selection must use original-only OCR')
                source = run['manifest']['books'][pair['book']]['sha256']
                for (book, page), reading in run['records'].items():
                    if book != pair['book'] or (book, page) not in run['selected']:
                        continue
                    raw = reading['raw']
                    prepared = raw.get('image_preparation', {})
                    identity = prepared.get('dataset', {})
                    if identity.get('source_sha256') != source or identity.get('variant') != 'original':
                        continue
                    for region in prepared.get('regions', []):
                        key = f"{source}:{page}:{region.get('canonical_region_id')}"
                        expected = canonical.get(key)
                        if not expected or region.get('variant_sha256') != expected['image_sha256'] or region.get('page_box') != expected['source_box']:
                            continue
                        bucket = observations.setdefault(key, {})
                        # Keep only the flag needed by the gate. Retaining raw
                        # detector polygons for the whole corpus can consume GBs.
                        item = dict(text=region['text'], raw={'hit_max':region.get('raw', {}).get('hit_max')})
                        if raw.get('hit_max') is True:
                            item['raw'] = {**item['raw'], 'hit_max': True}
                        previous = bucket.get(engine)
                        if previous is not None and (previous['text'] != item['text'] or
                                previous['raw'].get('hit_max') != item['raw'].get('hit_max')):
                            ambiguous.add(key)
                        bucket[engine] = item
    visit(manifest_path)
    scores = {}
    for key in canonical:
        scores[key] = (dict(state='ambiguous_alias_ocr', reason='identical source aliases disagree') if key in ambiguous else
                       restoration_pairwise_scores(observations.get(key, {}), min_characters=min_characters))
        scores[key]['region'] = canonical[key]
    return dict(schema='vtm-original-ocr-disagreement-v1', scores=scores, warnings=warnings,
                original_manifest=str(Path(manifest_path).resolve()),
                normalization='remove whitespace only; preserve glyph forms and punctuation',
                interpretation='OCR disagreement is a triage signal, not verified damage or accuracy')
