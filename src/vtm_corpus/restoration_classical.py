"""Orientation-independent image experiments. Original RGB is always retained separately."""
from __future__ import annotations

import math


EXTRA_CLASSICAL = (
    'grayscale_luma', 'grayscale_red', 'grayscale_green', 'grayscale_blue',
    'lab_l', 'lab_a', 'lab_b', 'red_ink', 'background_normalized',
    'sauvola', 'niblack', 'bilateral', 'unsharp',
)


def classical_parameters(name, options):
    defaults = {
        'background_normalized': dict(sigma=30., strength=.8, floor=32.),
        'red_ink': dict(min_red=80, dominance=1.25, difference=25),
        'sauvola': dict(window_size=31, k=.2, r=128.),
        'niblack': dict(window_size=31, k=.2),
        'bilateral': dict(diameter=5, sigma_color=20., sigma_space=3.),
        'unsharp': dict(sigma=1., amount=.35, threshold=3.),
    }
    params = {**defaults.get(name, {}), **options}
    if set(options) - set(defaults.get(name, {})):
        raise ValueError(f'{name}: unknown options {set(options)-set(defaults.get(name, {}))}')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in params.values()):
        raise ValueError(f'{name}: parameters must be finite numbers')
    if name in ('sauvola', 'niblack'):
        w = params['window_size']
        if not isinstance(w, int) or not 3 <= w <= 501 or w % 2 != 1 or not -1 <= params['k'] <= 1:
            raise ValueError('Binarization requires odd window_size 3..501 and k in [-1,1]')
        if name == 'sauvola' and not 1 <= params['r'] <= 255:
            raise ValueError('Sauvola r must be 1..255 on the uint8 luminance scale')
    if name == 'background_normalized' and not (
        1 <= params['sigma'] <= 500 and 0 <= params['strength'] <= 1 and 1 <= params['floor'] <= 255
    ):
        raise ValueError('Invalid background normalization parameters')
    if name == 'red_ink' and not (0 <= params['min_red'] <= 255 and
            1 <= params['dominance'] <= 4 and 0 <= params['difference'] <= 255):
        raise ValueError('Invalid red ink thresholds')
    if name == 'bilateral' and not (isinstance(params['diameter'], int) and
            params['diameter'] in (3, 5, 7, 9) and 0 < params['sigma_color'] <= 50 and
            0 < params['sigma_space'] <= 10):
        raise ValueError('Bilateral parameters exceed conservative limits')
    if name == 'unsharp' and not (0 < params['sigma'] <= 3 and
            0 <= params['amount'] <= .75 and 0 <= params['threshold'] <= 30):
        raise ValueError('Unsharp parameters exceed conservative limits')
    return params


def extra_classical_variant(image, name, options):
    import cv2
    import numpy as np
    from PIL import Image

    if name not in EXTRA_CLASSICAL:
        raise ValueError(f'Unknown classical variant: {name}')
    params = classical_parameters(name, options)
    rgb = np.asarray(image.convert('RGB'))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    warnings = ['experimental; inspect_against_original; ocr_accuracy_not_measured']
    details = {}
    if name == 'grayscale_luma':
        result = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    elif name.startswith('grayscale_'):
        result = rgb[:, :, {'red': 0, 'green': 1, 'blue': 2}[name.split('_')[1]]]
        warnings.append('channel_selection_may_hide_colored_ink; original_RGB_retained')
    elif name.startswith('lab_'):
        result = lab[:, :, {'l': 0, 'a': 1, 'b': 2}[name[-1]]]
        warnings.append('Lab_ab_are_chroma_diagnostics_not_luminance' if name != 'lab_l'
                        else 'grayscale_loses_color_information')
    elif name == 'red_ink':
        a = rgb.astype(np.float32)
        mask = ((a[:, :, 0] >= params['min_red']) &
                (a[:, :, 0] >= params['dominance'] * np.maximum(a[:, :, 1], a[:, :, 2])) &
                (a[:, :, 0] - np.maximum(a[:, :, 1], a[:, :, 2]) >= params['difference']))
        result = np.full_like(rgb, 255)
        result[mask] = rgb[mask]
        details['red_candidate_fraction'] = float(mask.mean())
        warnings.append('red_only_analysis; black_ink_omitted_in_this_variant_only; mask_is_not_ground_truth')
    elif name == 'background_normalized':
        luminance = lab[:, :, 0].astype(np.float32)
        background = cv2.GaussianBlur(luminance, (0, 0), params['sigma'], borderType=cv2.BORDER_REFLECT)
        corrected = luminance * float(np.median(background)) / np.maximum(background, params['floor'])
        lab[:, :, 0] = np.clip((1 - params['strength']) * luminance + params['strength'] * corrected, 0, 255).round().astype('uint8')
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        warnings.append('chroma_retained; may_create_halo_or_amplify_noise')
    elif name in ('sauvola', 'niblack'):
        # uint8 scale, population variance, reflected boundaries, no scikit-image dependency.
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
        window = (params['window_size'],) * 2
        mean = cv2.boxFilter(gray, -1, window, borderType=cv2.BORDER_REFLECT)
        variance = np.maximum(0, cv2.boxFilter(gray * gray, -1, window, borderType=cv2.BORDER_REFLECT) - mean * mean)
        std = np.sqrt(variance)
        threshold = mean * (1 + params['k'] * (std / params['r'] - 1)) if name == 'sauvola' else mean - params['k'] * std
        result = np.where(gray > threshold, 255, 0).astype('uint8')
        warnings.append('may_erase_weak_strokes_or_merge_strokes; color_lost')
        details.update(boundary='reflect', intensity_scale=[0, 255], geometry='identity')
    elif name == 'bilateral':
        lab[:, :, 0] = cv2.bilateralFilter(lab[:, :, 0], params['diameter'], params['sigma_color'], params['sigma_space'])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        warnings.append('chroma_retained; may_remove_weak_strokes')
    elif name == 'unsharp':
        l = lab[:, :, 0].astype(np.float32)
        delta = l - cv2.GaussianBlur(l, (0, 0), params['sigma'])
        delta[np.abs(delta) < params['threshold']] = 0
        lab[:, :, 0] = np.clip(l + params['amount'] * delta, 0, 255).round().astype('uint8')
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        warnings.append('chroma_retained; may_create_halo_or_false_edges')
    output = Image.fromarray(result).convert('RGB')
    if output.size != image.size:
        raise ValueError('Classical transformation unexpectedly changed geometry')
    return output, dict(method=name, parameters=params, opencv=cv2.__version__,
                        orientation='independent; no_rotation', warnings=warnings, **details)
