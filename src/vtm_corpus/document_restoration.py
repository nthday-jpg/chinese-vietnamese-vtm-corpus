"""Standalone worker adapters using pinned upstream inference architectures.

No project imports, downloads, implicit resizing, text rotation or dewarping.
Torch is imported only inside worker functions. Real GPU inference is a separate
verification step from checkpoint/contract tests.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from .learned_document_restoration import learned_document_load, learned_document_run


DOCUMENT_VARIANTS = {
    'docentr_binarization': ('docentr', 'binarization_base_2018_patch8', 'model_8_2018_base.pt'),
    'degan_binarization': ('degan', 'binarization', 'degan_weights.bundle'),
    'degan_deblurring': ('degan', 'deblurring', 'degan_weights.bundle'),
    'degan_watermark_removal': ('degan', 'watermark_removal', 'degan_weights.bundle'),
    'docres_appearance': ('docres', 'appearance', 'docres.pkl'),
    'docres_deblurring': ('docres', 'deblurring', 'docres.pkl'),
    'docres_deshadowing': ('docres', 'deshadowing', 'docres.pkl'),
    'docres_binarization': ('docres', 'binarization', 'docres.pkl'),
    'docdiff_deblurring': ('docdiff', 'deblurring', 'init.pth'),
    'restormer_motion_deblurring': ('restormer', 'motion', 'motion_deblurring.pth'),
    'restormer_defocus_deblurring': ('restormer', 'defocus', 'single_image_defocus_deblurring.pth'),
    'restormer_real_denoising': ('restormer', 'real', 'real_denoising.pth'),
    'restormer_gaussian_denoising': ('restormer', 'gaussian_color_blind', 'gaussian_color_denoising_blind.pth'),
}


def document_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def docres_prompt_module(vendor):
    """Use verbatim upstream prompt definitions without importing unused MBD.

    inference.py imports dewarp/MBD eagerly, although these four tasks do not
    use it. Selecting only the named definitions avoids its unrelated dependency
    graph. No function bodies are rewritten and source integrity is checked by
    preflight before this loader runs.
    """
    import cv2
    import numpy as np
    utils = document_module(Path(vendor) / 'utils.py', 'docres_upstream_utils')
    scope = dict(cv2=cv2, np=np, utils=utils)
    names = {'appearance_prompt', 'deshadow_prompt', 'deblur_prompt', 'binarization_promptv2'}
    tree = ast.parse((Path(vendor) / 'inference.py').read_text(encoding='utf8'))
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in definitions} != names:
        raise ValueError('Pinned DocRes DTSPrompt definitions missing')
    exec(compile(ast.Module(body=definitions, type_ignores=[]), 'docres-upstream-prompts', 'exec'), scope)
    return SimpleNamespace(**{n: scope[n] for n in names})


def document_load(model, assets, options):
    if model.startswith(('docentr_', 'degan_')):
        return learned_document_load(model, assets, options)
    import torch
    family, task, _ = DOCUMENT_VARIANTS[model]
    vendor = Path(assets['vendor'])
    device = options.get('device', 'cuda')
    if family == 'docres':
        arch = document_module(vendor / 'models/restormer_arch.py', 'docres_upstream_arch')
        net = arch.Restormer(inp_channels=6, out_channels=3, dim=48,
            num_blocks=[2, 3, 3, 4], num_refinement_blocks=4, heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', dual_pixel_task=True)
        state = torch.load(assets['checkpoint'], map_location='cpu', weights_only=True)['model_state']
        # Upstream saves DataParallel keys; never blindly truncate ordinary keys.
        if state and all(k.startswith('module.') for k in state):
            state = {k[7:]: v for k, v in state.items()}
        net.load_state_dict(state, strict=True)
        return SimpleNamespace(network=net.eval().to(device), prompts=docres_prompt_module(vendor), task=task)
    if family == 'docdiff':
        import yaml
        from model.DocDiff import DocDiff
        from schedule.schedule import Schedule
        from schedule.diffusionSample import GaussianDiffusion
        cfg = yaml.safe_load((vendor / 'conf.yml').read_text(encoding='utf8'))
        if (cfg['CHANNEL_X'], cfg['CHANNEL_Y'], cfg['TIMESTEPS'], cfg['SCHEDULE'], str(cfg['PRE_ORI'])) != (3, 3, 100, 'linear', 'True'):
            raise ValueError('DocDiff released deblurring checkpoint/config contract changed')
        if options.get('steps', 100) != 100:
            raise ValueError('Released DocDiff sampler uses 100 steps; changing T changes its schedule')
        net = DocDiff(input_channels=cfg['CHANNEL_X'] + cfg['CHANNEL_Y'], output_channels=cfg['CHANNEL_Y'],
                      n_channels=cfg['MODEL_CHANNELS'], ch_mults=cfg['CHANNEL_MULT'], n_blocks=cfg['NUM_RESBLOCKS'])
        net.init_predictor.load_state_dict(torch.load(assets['checkpoint'], map_location='cpu', weights_only=True), strict=True)
        net.denoiser.load_state_dict(torch.load(assets['denoiser'], map_location='cpu', weights_only=True), strict=True)
        net.eval().to(device)
        sampler = GaussianDiffusion(net.denoiser, 100, Schedule('linear', 100)).eval().to(device)
        return SimpleNamespace(network=net, sampler=sampler, task=task)
    arch = document_module(vendor / 'basicsr/models/archs/restormer_arch.py', 'restormer_upstream_arch')
    params = dict(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='BiasFree' if task in ('real', 'gaussian_color_blind') else 'WithBias',
        dual_pixel_task=False)
    net = arch.Restormer(**params)
    state = torch.load(assets['checkpoint'], map_location='cpu', weights_only=True)
    net.load_state_dict(state['params'], strict=True)
    return SimpleNamespace(network=net.eval().to(device), task=task)


def document_pad(tensor, multiple=8, *, top_left=False, mode='reflect'):
    import torch.nn.functional as F
    h, w = tensor.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    padding = (pw, 0, ph, 0) if top_left else (0, pw, 0, ph)
    # Reflect requires pad < dimension. Replication handles tiny/thin fixtures.
    actual = 'replicate' if mode == 'reflect' and (ph >= h or pw >= w) else mode
    return F.pad(tensor, padding, mode=actual), dict(padding=list(padding), mode=actual,
        original_size=[w, h], padded_size=[w + pw, h + ph], image_to_input=[1, 0, 0, 1, padding[0], padding[2]])


def document_tiled(tensor, predict, tile, overlap):
    """Independent axes, CPU accumulation; no resize/rotation, no uncovered border."""
    import torch
    h, w = tensor.shape[-2:]
    if not isinstance(tile, int) or tile < 8 or tile % 8 or not isinstance(overlap, int) or not 0 <= overlap < tile or overlap % 8:
        raise ValueError('Document tile/overlap must be multiples of 8; 0 <= overlap < tile')
    def positions(length):
        if length <= tile:
            return [0]
        return sorted(set([*range(0, length - tile + 1, tile - overlap), length - tile]))
    sums = torch.zeros(1, 3, h, w, dtype=torch.float32)
    weights = torch.zeros(1, 1, h, w, dtype=torch.float32)
    count = 0
    for y in positions(h):
        for x in positions(w):
            patch = tensor[:, :, y:min(y + tile, h), x:min(x + tile, w)]
            result = predict(patch).detach().float().cpu()
            if tuple(result.shape) != (1, 3, patch.shape[-2], patch.shape[-1]) or not torch.isfinite(result).all():
                raise ValueError('Document model violated output shape/finite-pixel contract')
            sums[:, :, y:y + result.shape[-2], x:x + result.shape[-1]] += result
            weights[:, :, y:y + result.shape[-2], x:x + result.shape[-1]] += 1
            count += 1
    if not (weights > 0).all():
        raise ValueError('Uncovered document pixels')
    return sums / weights, count


def document_run(model, backend, image, options):
    if model.startswith(('docentr_', 'degan_')):
        return learned_document_run(model, backend, image, options)
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    family, task, _ = DOCUMENT_VARIANTS[model]
    rgb = np.asarray(image.convert('RGB'))
    if family == 'docres':
        # DocRes checkpoint and DTSPrompt operate on BGR, unlike Restormer/DocDiff.
        bgr = rgb[:, :, ::-1].copy()
        if task in ('deblurring', 'binarization'):
            top, left = (-image.height) % 8, (-image.width) % 8
            padded = cv2.copyMakeBorder(bgr, top, 0, left, 0, cv2.BORDER_REPLICATE)
            prompt = (backend.prompts.deblur_prompt(padded) if task == 'deblurring'
                      else backend.prompts.binarization_promptv2(padded))
            array = np.concatenate((padded, prompt), axis=2)
            padding = dict(padding=[left, 0, top, 0], mode='replicate', original_size=list(image.size),
                           padded_size=[array.shape[1], array.shape[0]])
        else:
            prompt = (backend.prompts.appearance_prompt(bgr) if task == 'appearance'
                      else backend.prompts.deshadow_prompt(bgr))
            array = np.concatenate((bgr, prompt), axis=2)
        tensor = torch.from_numpy(array.astype(np.float32) / 255).permute(2, 0, 1)[None]
        if task in ('appearance', 'deshadowing'):
            tensor, padding = document_pad(tensor, top_left=True, mode='replicate')
        def predict(patch):
            value = backend.network(patch.to(options.get('device', 'cuda')))
            if task == 'binarization':
                value = value[:, :2].softmax(1).argmax(1, keepdim=True).float().repeat(1, 3, 1, 1)
            return value
    else:
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1)[None]
        tensor, padding = document_pad(tensor)
        def predict(patch):
            patch = patch.to(options.get('device', 'cuda'))
            if family == 'docdiff':
                coarse = backend.network.init_predictor(patch, 0)
                return coarse + backend.sampler(torch.randn_like(patch), coarse, 'True')
            return backend.network(patch)
    with torch.inference_mode():
        output, tiles = document_tiled(tensor, predict, options.get('tile', 512), options.get('overlap', 32))
    left, _, top, _ = padding['padding']
    output = output[:, :, top:top + image.height, left:left + image.width]
    pixels = (output[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype('uint8')
    if family == 'docres':
        pixels = pixels[:, :, ::-1].copy()
    return Image.fromarray(pixels), {}, dict(task=task, padding=padding, tiles=tiles,
        orientation='unchanged', geometry='padding_removed; identity',
        inference_mode='native_resolution_tiled; no_upstream_square_resize',
        fidelity='unverified; inspect_strokes_and_red_ink', seam_review='required')
