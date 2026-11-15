"""Placement adapters for the pinned, unmodified UniHIR and TADiSR sources.

CPU/disk spill is rejected. A device map verifies placement, not inference quality.
"""
from __future__ import annotations


def restoration_memory_limits(torch, options):
    cap = options.get('max_memory', '18GiB')
    if not isinstance(cap, str) or not cap.endswith('GiB'):
        raise ValueError('max_memory must be a per-GPU limit such as 18GiB')
    requested = int(float(cap[:-3]) * 1024**3)
    reserve = int(float(options.get('activation_reserve_gib', 4)) * 1024**3)
    if requested <= 0 or reserve < 1024**3:
        raise ValueError('Positive weight budget and >=1 GiB activation reserve required')
    limits = {}
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        limit = min(requested, free - reserve)
        # VAE, decoder and prompt live on the entry GPU outside the sharded model.
        if index == 0:
            limit -= int(options.get('entry_reserve_gib', 2) * 1024**3)
        if limit < 4 * 1024**3:
            raise MemoryError(f'GPU {index}: insufficient free VRAM after activation reserve')
        limits[index] = limit
    return limits


def verify_restoration_device_map(mapping, count, *, require_multiple=True):
    devices = set()
    for value in mapping.values():
        if isinstance(value, int):
            index = value
        else:
            label = str(value)
            if not label.startswith('cuda:'):
                raise MemoryError(f'Model would spill to {label}; reduce scope or change explicit placement budgets')
            index = int(label.split(':')[1])
        if not 0 <= index < count:
            raise ValueError(f'Device map refers to unavailable GPU {index}')
        devices.add(index)
    if require_multiple and len(devices) < 2:
        raise ValueError('Sharded placement did not distribute the model across multiple GPUs')
    return mapping


def unihir_weight_limits(torch, options):
    """Reserve a whole visible GPU for the full-resolution VAE encode/decode."""
    limits=restoration_memory_limits(torch,options)
    vae=options.get('vae_gpu_id')
    if vae is None:return limits,None
    if vae=='auto':vae=torch.cuda.device_count()-1
    if isinstance(vae,bool) or not isinstance(vae,int) or vae not in limits:
        raise ValueError('UniHIR vae_gpu_id must be auto or a visible logical GPU index')
    if len(limits)<3:
        raise MemoryError('Dedicated UniHIR VAE requires >=3 visible GPUs so BAGEL remains sharded across >=2 GPUs')
    if torch.cuda.mem_get_info(vae)[0]<16*1024**3:
        raise MemoryError(f'UniHIR VAE GPU {vae} needs >=16 GiB free before loading; no automatic resize')
    del limits[vae]
    return limits,vae


def restoration_gpu_probe(torch, model, options):
    """Validate every visible GPU; aggregate capacity is only a placement precheck."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required but unavailable')
    devices = []
    for index in range(torch.cuda.device_count()):
        (torch.ones(8, device=f'cuda:{index}') * 2).sum().item()
        torch.cuda.synchronize(index)
        devices.append(dict(index=index, name=torch.cuda.get_device_name(index),
            total_bytes=torch.cuda.get_device_properties(index).total_memory,
            free_bytes=torch.cuda.mem_get_info(index)[0], capability=list(torch.cuda.get_device_capability(index))))
    sharded = options.get('placement') == 'sharded'
    if sharded and len(devices) < 2:
        raise RuntimeError(f'{model}: sharded placement requires >=2 visible GPUs')
    if model in ('unihir_restoration', 'tadisr_x4') and sharded:
        limits = unihir_weight_limits(torch,options)[0] if model=='unihir_restoration' else restoration_memory_limits(torch, options)
        minimum = 32 if model == 'unihir_restoration' else 28
        if sum(limits.values()) < minimum * 1024**3:
            raise MemoryError(f'{model}: insufficient aggregate weight budgets; inference workspace is reserved separately')
    elif model == 'unihir_restoration' and devices[0]['total_bytes'] < 75 * 1024**3:
        raise RuntimeError('Single-GPU UniHIR needs >=75 GiB; use placement=sharded for multiple L4 GPUs')
    elif model == 'tadisr_x4' and devices[0]['total_bytes'] < 24 * 1024**3:
        raise RuntimeError('Single-GPU float32 TADiSR needs >=24 GiB; use placement=sharded on L4')
    if model == 'unihir_restoration' and any(d['capability'][0] < 8 for d in devices):
        raise RuntimeError('UniHIR BF16/FlashAttention requires Ampere-or-newer GPUs')
    report=dict(devices=devices, placement=options.get('placement', 'single'),
        cuda_visible_devices=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
        weight_capacity_check_only=True)
    if model=='unihir_restoration' and sharded:
        weight_limits,vae_gpu=unihir_weight_limits(torch,options)
        report.update(weight_gpu_ids=sorted(weight_limits),vae_gpu_id=vae_gpu)
    return report


def cpu_staged_checkpoint_dispatch(dispatch, model, **kwargs):
    """Stage checkpoint tensors on CPU before Accelerate casts/places them.

    The pinned Accelerate safetensors loader otherwise holds the checkpoint's
    original dtype on GPU alongside the converted model. Clone function globals
    for this call only; neither Accelerate nor the upstream module is patched.
    Final model placement still follows the validated GPU-only device map.
    """
    import types

    def clone(function, replacements):
        result = types.FunctionType(function.__code__,
            {**function.__globals__, **replacements}, function.__name__,
            function.__defaults__, function.__closure__)
        result.__kwdefaults__ = function.__kwdefaults__
        return result

    load = dispatch.__globals__['load_checkpoint_in_model']
    read = load.__globals__['load_state_dict']

    def cpu_read(checkpoint_file, device_map=None):
        return read(checkpoint_file, device_map=None)

    staged_load = clone(load, {'load_state_dict': cpu_read})
    return clone(dispatch, {'load_checkpoint_in_model': staged_load})(model, **kwargs)


def verify_unihir_vae(vae):
    """Check the real codec on a tiny tensor, restoring all touched RNG states."""
    import time
    import torch
    parameter=next(vae.parameters());device=parameter.device
    devices=[device.index] if device.type=='cuda' else []
    started=time.monotonic()
    with torch.random.fork_rng(devices=devices),torch.inference_mode():
        image=torch.zeros((1,3,64,64),device=device,dtype=parameter.dtype)
        latent=vae.encode(image);decoded=vae.decode(latent)
        if decoded.shape!=image.shape:raise ValueError('UniHIR VAE roundtrip dimensions differ')
        if not torch.isfinite(latent).all().item() or not torch.isfinite(decoded).all().item():
            raise ValueError('UniHIR VAE produced non-finite values in codec precheck')
    return dict(device=str(device),dtype=str(parameter.dtype),input_size=[64,64],
                seconds=round(time.monotonic()-started,3),status='passed',scope='codec_only; not full model inference')


def load_unihir_multi_gpu(assets, options):
    import torch
    import types
    import utils as upstream
    from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch
    limits,vae_gpu=unihir_weight_limits(torch,options)

    def validate(mapping):
        verify_restoration_device_map(mapping,torch.cuda.device_count())
        for device in mapping.values():
            index=device if isinstance(device,int) else int(str(device).split(':')[1])
            if index not in limits:raise MemoryError('UniHIR weight map uses the GPU reserved for VAE')
        return mapping

    def infer(model, **kwargs):
        # The upstream estimate otherwise assumes float32 although loading is BF16.
        kwargs.update(max_memory=limits, dtype=torch.bfloat16,
                      no_split_module_classes=['Qwen2MoTDecoderLayer'])
        return validate(infer_auto_device_map(model, **kwargs))

    def dispatch(model, **kwargs):
        # Upstream co-locates its embedding/connector modules after inference.
        validate(kwargs['device_map'])
        return cpu_staged_checkpoint_dispatch(load_checkpoint_and_dispatch, model, **kwargs)

    # Bind two placement functions without changing the pinned upstream files/globals.
    loader = types.FunctionType(upstream.load_model.__code__, {
        **upstream.load_model.__globals__, 'infer_auto_device_map': infer,
        'load_checkpoint_and_dispatch': dispatch}, argdefs=upstream.load_model.__defaults__)
    result = loader(assets['base_model'], max_mem_per_gpu=options.get('max_memory', '18GiB'))
    result[0]._vtm_device_map = {k: str(v) for k, v in result[0].hf_device_map.items()}
    if vae_gpu is not None:
        result[2].to(f'cuda:{vae_gpu}')
        result[0]._vtm_device_map['vae_model']=f'cuda:{vae_gpu}'
    if options.get('verify_vae',False):result[0]._vtm_vae_smoke=verify_unihir_vae(result[2])
    return result


def load_tadisr_multi_gpu(assets, options):
    import pathlib
    import torch
    from accelerate import dispatch_model, infer_auto_device_map
    from tadisr.inference import CogView4Config, load_cogview4
    # Construct/load on host RAM; never materialize the complete float32 stack on L4 0.
    net = load_cogview4(CogView4Config(pathlib.Path(assets['base_model']),
        pathlib.Path(assets['checkpoint']), pathlib.Path(assets['prompt_embeddings']), 'cpu'))
    limits = restoration_memory_limits(torch, options)
    no_split = getattr(net.transformer, '_no_split_modules', None) or ['CogView4Block']
    mapping = infer_auto_device_map(net.transformer, max_memory=limits,
        no_split_module_classes=no_split, dtype=torch.float32)
    verify_restoration_device_map(mapping, len(limits))
    net.transformer = dispatch_model(net.transformer, device_map=mapping, main_device=0, force_hooks=True)
    net.execution_device = torch.device('cuda:0')
    net.vae.to(net.execution_device)
    net.js_decoder.to(net.execution_device)
    net.prompt_embeds = net.prompt_embeds.to(net.execution_device)
    net._vtm_device_map = {k: str(v) for k, v in mapping.items()}
    return net


def restoration_cuda_metrics(torch, *, reset=False):
    records = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)
            if reset:
                torch.cuda.reset_peak_memory_stats(index)
            records.append(dict(index=index, peak_allocated_bytes=torch.cuda.max_memory_allocated(index),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved(index)))
    return records
