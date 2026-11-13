"""Real released DocEnTR and DE-GAN adapters, with native upright patch geometry.

No training, network access, TF installation, whole-page resizing or fallback.
DocEnTR uses the pinned upstream BinModel and vit-pytorch 0.24.3. DE-GAN uses
the verbatim upstream generator definition with Keras 3's Torch backend.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import zipfile


LEARNED_DOCUMENT_TASKS = {
    'docentr_binarization': ('docentr', None, .5),
    'degan_binarization': ('degan', 'weights/binarization_generator_weights.h5', .95),
    'degan_deblurring': ('degan', 'weights/deblur_weights.h5', None),
    'degan_watermark_removal': ('degan', 'weights/watermark_rem_weights.h5', None),
}


def learned_document_options(name, options):
    if name not in LEARNED_DOCUMENT_TASKS:
        raise ValueError(name)
    if options.get('patch_size', 256) != 256:
        raise ValueError('Released learned document checkpoints require 256x256 native patches; no resize')
    if options.get('device', 'cuda') not in ('cpu', 'cuda', 'cuda:0'):
        raise ValueError('Select one CPU/CUDA device; learned document models run sequentially')


def learned_document_load(name, assets, options):
    import torch
    learned_document_options(name, options)
    family, member, _ = LEARNED_DOCUMENT_TASKS[name]
    vendor = Path(assets['vendor'])
    device = options.get('device', 'cuda')
    if family == 'docentr':
        from vit_pytorch import ViT
        spec = importlib.util.spec_from_file_location('docentr_pinned_binae', vendor/'models/binae.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        encoder = ViT(image_size=256, patch_size=8, num_classes=1000, dim=768,
                      depth=6, heads=8, mlp_dim=2048)
        network = module.BinModel(encoder=encoder, decoder_dim=768, decoder_depth=6, decoder_heads=8)
        network.load_state_dict(torch.load(assets['checkpoint'], map_location='cpu', weights_only=True), strict=True)
        return SimpleNamespace(network=network.eval().to(device), family=family)
    # The upstream generator has no TensorFlow-specific operations in its body.
    # Use only that exact definition; unused training/discriminator imports are not executed.
    os.environ['KERAS_BACKEND'] = 'torch'
    import keras
    if keras.backend.backend() != 'torch':
        raise RuntimeError('DE-GAN requires a fresh worker with KERAS_BACKEND=torch; kernel backend is not changed')
    keras.backend.clear_session()
    tree = ast.parse((vendor/'models/models.py').read_text(encoding='utf8'))
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generator_model']
    if len(definitions) != 1:
        raise ValueError('Pinned DE-GAN generator definition missing')
    scope = {n: getattr(keras.layers, n) for n in (
        'Input', 'Conv2D', 'MaxPooling2D', 'Dropout', 'UpSampling2D', 'concatenate')}
    scope['Model'] = keras.Model
    exec(compile(ast.Module(body=definitions, type_ignores=[]), 'degan-upstream-generator', 'exec'), scope)
    network = scope['generator_model'](biggest_layer=512 if name == 'degan_watermark_removal' else 1024)
    record = assets.get('archive_members', {}).get(member)
    if not record or not record.get('sha256'):
        raise ValueError('DE-GAN archive member integrity manifest is missing; run preparation script')
    with tempfile.TemporaryDirectory(prefix='vtm_degan_') as temporary:
        target = Path(temporary)/Path(member).name
        digest = hashlib.sha256(); written = 0
        with zipfile.ZipFile(assets['checkpoint']) as archive:
            if archive.getinfo(member).file_size != record['bytes']:
                raise ValueError('DE-GAN archive member size differs from manifest')
            with archive.open(member) as source, target.open('wb') as output:
                for block in iter(lambda: source.read(4*1024**2), b''):
                    digest.update(block); written += len(block); output.write(block)
        if written != record['bytes'] or digest.hexdigest() != record['sha256']:
            raise ValueError('DE-GAN archive member checksum mismatch')
        # No by_name/skip_mismatch: all released generator weights must load.
        network.load_weights(target)
    return SimpleNamespace(network=network.to(device).eval(), family=family,
                           keras_version=keras.__version__, backend='torch')


def learned_document_run(name, backend, image, options):
    import numpy as np
    import torch
    from PIL import Image
    learned_document_options(name, options)
    family, _, threshold = LEARNED_DOCUMENT_TASKS[name]
    device = options.get('device', 'cuda')
    if family == 'docentr':
        # Preserve the released demo's cv2 BGR convention and ImageNet mean/std.
        array = np.asarray(image.convert('RGB'))[:, :, ::-1].copy().astype(np.float32)/255
        channels = 3
    else:
        array = np.asarray(image.convert('L'), dtype=np.float32)[:, :, None]/255
        channels = 1
    width, height = image.size
    pw, ph = (-width) % 256, (-height) % 256
    padded = np.ones((height+ph, width+pw, channels), dtype=np.float32)
    padded[:height, :width] = array
    output = np.empty_like(padded)
    mean = np.array([.485, .456, .406], dtype=np.float32)[:, None, None]
    std = np.array([.229, .224, .225], dtype=np.float32)[:, None, None]
    count = 0
    with torch.inference_mode():
        for y in range(0, height+ph, 256):
            for x in range(0, width+pw, 256):
                patch = padded[y:y+256, x:x+256]
                if family == 'docentr':
                    tensor = torch.from_numpy((patch.transpose(2, 0, 1)-mean)/std)[None].to(device)
                    _, _, tokens = backend.network(tensor, torch.zeros_like(tensor))
                    if tuple(tokens.shape) != (1, 1024, 192):
                        raise ValueError('DocEnTR released base/8 checkpoint output contract changed')
                    value = tokens.reshape(1, 32, 32, 8, 8, 3).permute(0, 5, 1, 3, 2, 4).reshape(3, 256, 256)
                    result = value.detach().cpu().numpy()*std+mean
                    result = result.transpose(1, 2, 0)
                else:
                    tensor = torch.from_numpy(patch[None]).to(device)
                    prediction = backend.network(tensor, training=False)
                    if tuple(prediction.shape) != (1, 256, 256, 1):
                        raise ValueError('DE-GAN released generator output contract changed')
                    result = prediction.detach().cpu().numpy()[0]
                if not np.isfinite(result).all():
                    raise ValueError('Learned document model produced non-finite pixels')
                output[y:y+256, x:x+256] = np.clip(result, 0, 1)
                count += 1
    output = output[:height, :width]
    if threshold is not None:
        output = (output > threshold).astype(np.float32)
    pixels = (output*255).round().astype('uint8')
    if channels == 1:
        pixels = np.repeat(pixels, 3, axis=2)
    else:
        pixels = pixels[:, :, ::-1].copy()
    return Image.fromarray(pixels), {}, dict(task=name, patches=count, patch_size=256,
        padding=[0, pw, 0, ph], padding_value=1., geometry='native_patches; padding_removed; identity',
        orientation='unchanged; no_rotation', threshold=threshold,
        channel_convention='upstream_BGR_ImageNet_normalization' if family == 'docentr' else 'upstream_Pillow_luminance',
        output_conversion='clip_01_to_uint8; no_matplotlib_display_autocontrast',
        keras_backend=getattr(backend, 'backend', None), keras_version=getattr(backend, 'keras_version', None),
        warnings=['learned_prior; may_erase_or_add_strokes; red_ink_color_not_preserved',
                  'nonoverlapping_native_256_patches_match_upstream; inspect_seams'])
