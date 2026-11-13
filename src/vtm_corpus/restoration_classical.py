"""Orientation-independent image experiments. Original RGB is always retained separately."""
from __future__ import annotations

import math

from .restoration_registry import variant_spec


EXTRA_CLASSICAL = (
    'grayscale_luma', 'grayscale_red', 'grayscale_green', 'grayscale_blue',
    'lab_l', 'lab_a', 'lab_b', 'red_ink', 'background_normalized',
    'sauvola', 'niblack', 'bilateral', 'unsharp',
    'retinex_msr', 'retinex_msrcp', 'wolf', 'nick', 'color_pca', 'color_gmm',
    'mean_shift', 'anisotropic_diffusion', 'background_morphological',
    'background_guided', 'background_otsu',
)


def classical_parameters(name, options):
    defaults = {
        'background_normalized': dict(sigma=30., strength=.8, floor=32.),
        'red_ink': dict(min_red=80, dominance=1.25, difference=25),
        'sauvola': dict(window_size=31, k=.2, r=128.),
        'niblack': dict(window_size=31, k=.2),
        'bilateral': dict(diameter=5, sigma_color=20., sigma_space=3.),
        'unsharp': dict(sigma=1., amount=.35, threshold=3.),
        'retinex_msr': dict(sigma_small=15., sigma_medium=40., sigma_large=100., clip_percent=1., strength=.6),
        'retinex_msrcp': dict(sigma_small=15., sigma_medium=40., sigma_large=100., clip_percent=1., strength=.6),
        'wolf': dict(window_size=31, k=.5),
        'nick': dict(window_size=31, k=-.1),
        'color_pca': dict(sample_pixels=8192, clip_percent=1.),
        'color_gmm': dict(sample_pixels=8192, components=3, iterations=15, variance_floor=9.),
        'mean_shift': dict(spatial_radius=3., color_radius=12.),
        'anisotropic_diffusion': dict(iterations=5, kappa=12., timestep=.15),
        'background_morphological': dict(kernel_size=31, strength=.8, floor=32.),
        'background_guided': dict(kernel_size=31, strength=.8, floor=32., radius=15, epsilon=.01),
        'background_otsu': dict(kernel_size=31, strength=1., floor=32.),
    }
    params = {**defaults.get(name, {}), **options}
    if set(options) - set(defaults.get(name, {})):
        raise ValueError(f'{name}: unknown options {set(options)-set(defaults.get(name, {}))}')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in params.values()):
        raise ValueError(f'{name}: parameters must be finite numbers')
    if name in ('sauvola', 'niblack', 'wolf', 'nick'):
        w = params['window_size']
        if not isinstance(w, int) or not 3 <= w <= 501 or w % 2 != 1 or not -1 <= params['k'] <= 1:
            raise ValueError('Binarization requires odd window_size 3..501 and k in [-1,1]')
        if name == 'sauvola' and not 1 <= params['r'] <= 255:
            raise ValueError('Sauvola r must be 1..255 on the uint8 luminance scale')
        if name == 'wolf' and not 0 <= params['k'] <= 1:
            raise ValueError('Wolf k must be in [0,1]')
        if name == 'nick' and not -.5 <= params['k'] < 0:
            raise ValueError('NICK k must be negative in [-.5,0)')
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
    if name.startswith('retinex_') and not (
            1 <= params['sigma_small'] < params['sigma_medium'] < params['sigma_large'] <= 500 and
            0 <= params['clip_percent'] <= 5 and 0 < params['strength'] <= 1):
        raise ValueError('Invalid Retinex scales/clipping/strength')
    if name in ('color_pca', 'color_gmm'):
        if type(params['sample_pixels']) is not int or not 64 <= params['sample_pixels'] <= 65536:
            raise ValueError('Color fit sample_pixels must be integer 64..65536')
        if name == 'color_pca' and not 0 <= params['clip_percent'] <= 5:
            raise ValueError('Invalid PCA clipping')
        if name == 'color_gmm' and not (type(params['components']) is int and 2 <= params['components'] <= 5 and
                type(params['iterations']) is int and 1 <= params['iterations'] <= 50 and
                1 <= params['variance_floor'] <= 100):
            raise ValueError('Invalid GMM fit budget/variance floor')
    if name == 'mean_shift' and not (1 <= params['spatial_radius'] <= 8 and 1 <= params['color_radius'] <= 30):
        raise ValueError('Mean Shift radii exceed conservative limits')
    if name == 'anisotropic_diffusion' and not (type(params['iterations']) is int and
            1 <= params['iterations'] <= 30 and 1 <= params['kappa'] <= 40 and 0 < params['timestep'] <= .25):
        raise ValueError('Diffusion requires bounded iterations and stable explicit timestep <= .25')
    if name in ('background_morphological', 'background_guided', 'background_otsu'):
        if not (type(params['kernel_size']) is int and 3 <= params['kernel_size'] <= 301 and
                params['kernel_size'] % 2 and 0 < params['strength'] <= 1 and 1 <= params['floor'] <= 255):
            raise ValueError('Invalid morphological background scale/strength/floor')
        if name == 'background_guided' and not (type(params['radius']) is int and
                1 <= params['radius'] <= 150 and 1e-5 <= params['epsilon'] <= .5):
            raise ValueError('Invalid guided-background radius/epsilon')
    return params


def classical_resource_metadata(name, pixels):
    """Conservative array-workspace estimate, not measured peak RSS or wall time."""
    limits = {'mean_shift': 4_000_000, 'color_gmm': 8_000_000,
              'anisotropic_diffusion': 8_000_000, 'retinex_msr': 12_000_000,
              'retinex_msrcp': 12_000_000, 'background_guided': 12_000_000}
    return dict(device='cpu', input_pixels=pixels, workspace_estimate_bytes=pixels*160 + 16*1024**2,
                workspace_estimate_status='conservative_arrays_only; library_allocator_RSS_unmeasured',
                method_max_pixels=limits.get(name, 25_000_000))


def classical_local_moments(gray, window_size):
    import cv2
    import numpy as np
    gray = gray.astype(np.float64)
    window = (window_size,)*2
    mean = cv2.boxFilter(gray, -1, window, borderType=cv2.BORDER_REFLECT)
    second = cv2.boxFilter(gray*gray, -1, window, borderType=cv2.BORDER_REFLECT)
    variance = np.maximum(0, second-mean*mean)
    return mean, second, variance


def classical_percentile_map(values, percent, *, flat_value):
    import numpy as np
    lo, hi = np.percentile(values, [percent, 100-percent])
    if hi-lo <= 1e-8:
        return np.full_like(values, flat_value), dict(flat_input=True, low=float(lo), high=float(hi))
    return np.clip((values-lo)*255/(hi-lo), 0, 255), dict(flat_input=False, low=float(lo), high=float(hi))


def classical_gmm(lab, params):
    import numpy as np
    pixels = lab.reshape(-1, 3).astype(np.float64)
    sample = pixels[np.linspace(0, len(pixels)-1, min(len(pixels), params['sample_pixels']), dtype=int)]
    # Farthest-point seeds avoid repeated white quantiles on monochrome pages.
    # Do not invent extra components when the input contains fewer distinct colors.
    centers = [sample[int(sample[:, 0].argmax())]]
    distance = np.full(len(sample), np.inf)
    while len(centers) < params['components']:
        distance = np.minimum(distance, ((sample-centers[-1])**2).sum(axis=1))
        index = int(distance.argmax())
        if distance[index] < 1e-8:
            break
        centers.append(sample[index])
    means = np.asarray(centers).copy()
    count = len(means)
    variances = np.tile(np.maximum(sample.var(0), params['variance_floor']), (count, 1))
    weights = np.full(count, 1/count)
    def posterior(x):
        logp = -.5*np.sum((x[:, None, :]-means)**2/variances + np.log(variances), axis=2) + np.log(weights)
        logp -= logp.max(axis=1, keepdims=True)
        prob = np.exp(logp)
        return prob/prob.sum(axis=1, keepdims=True)
    for _ in range(params['iterations']):
        prob = posterior(sample)
        mass = np.maximum(prob.sum(0), 1e-8)
        means = prob.T@sample/mass[:, None]
        variances = np.maximum(prob.T@(sample*sample)/mass[:, None]-means*means, params['variance_floor'])
        weights = np.maximum(mass/len(sample), 1e-8); weights /= weights.sum()
    paper = int(means[:, 0].argmax())
    result = np.empty(len(pixels), dtype='uint8')
    for start in range(0, len(pixels), 65536):
        # High paper posterior -> white. Other components include black AND red ink.
        result[start:start+65536] = (255*posterior(pixels[start:start+65536])[:, paper]).round().astype('uint8')
    return result.reshape(lab.shape[:2]), dict(component_means_lab=means.tolist(), paper_component=paper,
        fit_pixels=len(sample), initialization='deterministic_farthest_color_points',
        requested_components=params['components'], effective_components=count,
        warning='brightest_component_is_an_assumption; posterior_is_not_ink_ground_truth')


def new_classical_pixels(rgb, lab, name, params):
    import cv2
    import numpy as np
    luma = lab[:, :, 0].astype(np.float32)
    details, warnings = {}, []
    if name in ('retinex_msr', 'retinex_msrcp'):
        intensity = rgb.astype(np.float32).mean(axis=2)
        log_intensity = np.log1p(intensity)
        reflectance = np.zeros_like(intensity)
        for key in ('sigma_small', 'sigma_medium', 'sigma_large'):
            reflectance += (log_intensity-np.log1p(cv2.GaussianBlur(intensity, (0, 0), params[key],
                borderType=cv2.BORDER_REFLECT)))/3
        target, details = classical_percentile_map(reflectance, params['clip_percent'], flat_value=float(intensity.mean()))
        target = (1-params['strength'])*intensity + params['strength']*target
        if name == 'retinex_msrcp':
            gain = np.minimum(255/np.maximum(rgb.max(axis=2), 1), target/np.maximum(intensity, 1e-6))
            result = np.clip(rgb.astype(np.float32)*gain[:, :, None], 0, 255).round().astype('uint8')
            warnings.append('common_RGB_gain_preserves_ratios_before_quantization; not_colorimetric_fidelity')
        else:
            result = target.round().astype('uint8')
            warnings.append('grayscale_loses_red_ink_color')
        warnings.append('log_contrast_mapping_can_amplify_noise_and_create_halos')
    elif name in ('color_pca', 'color_gmm'):
        if name == 'color_gmm':
            result, details = classical_gmm(lab, params)
        else:
            pixels = rgb.reshape(-1, 3).astype(np.float32)
            sample = pixels[np.linspace(0, len(pixels)-1, min(len(pixels), params['sample_pixels']), dtype=int)]
            center = sample.mean(0)
            eigenvalues, axes = np.linalg.eigh((sample-center).T@(sample-center)/max(1, len(sample)))
            axis = axes[:, -1]
            if float(axis@np.array([.299, .587, .114])) < 0:
                axis = -axis
            projected = ((pixels-center)@axis).reshape(rgb.shape[:2])
            result, details = classical_percentile_map(projected, params['clip_percent'], flat_value=255.)
            result = result.round().astype('uint8')
            details.update(axis_rgb=axis.tolist(), eigenvalues=eigenvalues.tolist(), fit_pixels=len(sample))
        warnings.append('unsupervised_color_projection; may_hide_faint_or_red_ink; color_lost')
    elif name == 'mean_shift':
        result = cv2.pyrMeanShiftFiltering(rgb, params['spatial_radius'], params['color_radius'], maxLevel=0)
        details.update(pyramid_levels=0, geometry='identity')
        warnings.append('mode_seeking_can_merge_nearby_strokes_and_colors')
    elif name == 'anisotropic_diffusion':
        current = luma.copy()
        for _ in range(params['iterations']):
            padded = np.pad(current, 1, mode='edge')
            delta = np.zeros_like(current)
            for neighbor in (padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]):
                gradient = neighbor-current
                delta += np.exp(-(gradient/params['kappa'])**2)*gradient
            current += params['timestep']*delta
        lab[:, :, 0] = np.clip(current, 0, 255).round().astype('uint8')
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        details.update(boundary='no_flux', stencil='four_neighbors')
        warnings.append('weak_gradient_strokes_can_diffuse; no_missing_stroke_reconstruction')
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params['kernel_size'],)*2)
        surface = cv2.morphologyEx(lab[:, :, 0], cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REFLECT).astype(np.float32)
        if name == 'background_guided':
            guide = surface/255
            window = (2*params['radius']+1,)*2
            def average(a): return cv2.boxFilter(a, -1, window, borderType=cv2.BORDER_REFLECT)
            mean = average(guide)
            variance = np.maximum(0, average(guide*guide)-mean*mean)
            a = variance/(variance+params['epsilon'])
            b = mean-a*mean
            surface = np.clip((average(a)*guide+average(b))*255, 0, 255)
            details['guidance'] = 'estimated_background_only; never_original_strokes'
        corrected = luma*float(np.median(surface))/np.maximum(surface, params['floor'])
        normalized = np.clip((1-params['strength'])*luma+params['strength']*corrected, 0, 255).round().astype('uint8')
        if name == 'background_otsu':
            threshold, result = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            details.update(observed_otsu_threshold=float(threshold), hypothesis_components=['closing_background', 'division_L', 'Otsu'])
            warnings.append('binarization_loses_red_color_and_can_erase_weak_strokes')
        else:
            lab[:, :, 0] = normalized
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        warnings.append('morphology_only_estimates_background; large_letters_rules_may_contaminate_surface')
    return result, details, warnings


def extra_classical_variant(image, name, options):
    import cv2
    import numpy as np
    from PIL import Image

    if name not in EXTRA_CLASSICAL:
        raise ValueError(f'Unknown classical variant: {name}')
    params = classical_parameters(name, options)
    resources = classical_resource_metadata(name, image.width*image.height)
    if image.width*image.height > resources['method_max_pixels']:
        raise MemoryError(f'{name} exceeds max_method_pixels={resources["method_max_pixels"]}; choose smaller regions explicitly')
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
    elif name in ('sauvola', 'niblack', 'wolf', 'nick'):
        # uint8 scale, population variance, reflected boundaries, no scikit-image dependency.
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
        mean, second, variance = classical_local_moments(gray, params['window_size'])
        std = np.sqrt(variance)
        if name == 'sauvola':
            threshold = mean * (1 + params['k'] * (std / params['r'] - 1))
        elif name == 'niblack':
            threshold = mean-params['k']*std
        elif name == 'wolf':
            maximum = float(std.max())
            threshold = mean-params['k']*(1-std/max(maximum, 1e-8))*(mean-float(gray.min()))
            details['global_max_std'] = maximum
        else:
            threshold = mean+params['k']*np.sqrt(second+variance)
        # Constant white windows must remain paper, including equality to T.
        result = np.where(gray >= threshold-1e-7, 255, 0).astype('uint8')
        warnings.append('may_erase_weak_strokes_or_merge_strokes; color_lost')
        details.update(boundary='reflect', intensity_scale=[0, 255], geometry='identity', threshold_tie_epsilon=1e-7)
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
    else:
        result, details, extra_warnings = new_classical_pixels(rgb, lab, name, params)
        warnings.extend(extra_warnings)
    output = Image.fromarray(result).convert('RGB')
    if output.size != image.size:
        raise ValueError('Classical transformation unexpectedly changed geometry')
    return output, dict(method=name, parameters=params, opencv=cv2.__version__,
                        **variant_spec(name), resources=resources,
                        orientation='independent; no_rotation', warnings=warnings, **details)
