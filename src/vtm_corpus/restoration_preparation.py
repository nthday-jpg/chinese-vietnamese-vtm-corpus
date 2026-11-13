"""CPU-only planning for the existing notebook preparation engine."""
from __future__ import annotations

import json

from .image_dataset import dataset_read
from .restoration_registry import variant_spec, preparation_applicability
from .restoration_classical import EXTRA_CLASSICAL, classical_resource_metadata, classical_parameters


def plan_restoration_variants(root, base, config, *, applicability=None):
    applicability = applicability or preparation_applicability
    result = dict(schema='vtm-restoration-plan-v1', selected_regions=0, manifest_rows=0,
        planned_output_images=0, planned_output_pixels=0, by_variant={}, resource_warnings=[],
        storage_estimate_basis='PNG upper estimate from raw RGB plus row/deflate overhead; before deduplication',
        runtime_estimate='unmeasured; CPU crop timings are not full-page/GPU estimates')
    storage = 0
    for page in base['pages']:
        record = json.loads(dataset_read(root, page['record']))
        for region in record['regions']:
            selected = region['crop_type'] in config['crop_types'] and (
                not config.get('region_ids') or region['id'] in config['region_ids'])
            result['selected_regions'] += int(selected)
            w, h = region['images']['original']['size']
            for name in config['variants']:
                spec = variant_spec(name)
                item = result['by_variant'].setdefault(name, dict(planned_images=0, pixels=0, not_applicable=0,
                    deselected=0, method=spec, parameters=classical_parameters(name,
                        config.get('variant_options', {}).get(name, {})) if name in EXTRA_CLASSICAL else
                        config.get('model_options', {}).get(name, config.get('variant_options', {}).get(name, {}))))
                result['manifest_rows'] += 1
                if not selected:
                    item['deselected'] += 1
                    continue
                state, reason = applicability('tadisr_x4' if name == 'tadisr_text_mask' else name,
                    page['book'], page['page'], region, config)
                if state != 'ready':
                    item['not_applicable'] += 1
                    continue
                scale = spec['scale']
                ow, oh = (w*scale, h*scale) if scale else (512, 128)
                pixels = ow*oh
                item['planned_images'] += 1; item['pixels'] += pixels
                result['planned_output_images'] += 1; result['planned_output_pixels'] += pixels
                if name != 'original':
                    storage += (3*ow+1)*oh*1.01 + 1024
                limits = classical_resource_metadata(name, w*h) if name in EXTRA_CLASSICAL else {}
                if pixels > config['max_output_pixels'] or w*h > limits.get('method_max_pixels', float('inf')):
                    result['resource_warnings'].append(dict(book=page['book'], page=page['page']+1,
                        region=region['id'], variant=name, status='resource_exceeded', pixels=pixels,
                        method_max_pixels=limits.get('method_max_pixels'), reason='explicit pixel limit; no automatic downscale'))
    result['additional_storage_estimate_bytes'] = int(storage)
    result['deduplication'] = 'content-addressed PNG storage; identical outputs can share objects, method rows stay distinct'
    return result
