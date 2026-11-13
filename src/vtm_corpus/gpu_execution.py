"""GPU placement for isolated inference workers; no CUDA import in the notebook."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event


def gpu_worker_environment(indices, environment=None):
    """Select logical devices, respecting a mask inherited from the platform."""
    env = dict(os.environ if environment is None else environment)
    indices = list(indices)
    if not indices or any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices):
        raise ValueError('GPU IDs must be distinct nonnegative logical indices')
    inherited = env.get('CUDA_VISIBLE_DEVICES')
    visible = inherited.split(',') if inherited is not None else None
    if visible is not None:
        visible = [v.strip() for v in visible if v.strip() and v.strip() != '-1']
        if any(i >= len(visible) for i in indices):
            raise ValueError(f'GPU IDs {indices} exceed inherited CUDA_VISIBLE_DEVICES={inherited!r}')
    env['CUDA_VISIBLE_DEVICES'] = ','.join(visible[i] if visible is not None else str(i) for i in indices)
    return env


def validate_ocr_gpu_map(engines, mapping):
    if set(mapping) != set(engines):
        raise ValueError('GPU_MAP must assign exactly one GPU to every OCR engine')
    ids = list(mapping.values())
    if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Parallel OCR requires one distinct nonnegative GPU ID per engine')
    return dict(mapping)


def probe_gpu_inventory():
    """Probe in a child so importing Torch cannot freeze the parent's CUDA mask."""
    source = '''import json, torch
devices=[]
for i in range(torch.cuda.device_count()):
    x=torch.ones(2,device=f'cuda:{i}'); value=float((x*x).sum().item())
    devices.append(dict(index=i,name=torch.cuda.get_device_name(i),
        total_bytes=torch.cuda.get_device_properties(i).total_memory,
        free_bytes=torch.cuda.mem_get_info(i)[0],capability=list(torch.cuda.get_device_capability(i)),
        tensor_sum=value))
print(json.dumps(devices))
'''
    result = subprocess.run([sys.executable, '-u', '-c', source], capture_output=True,
                            text=True, encoding='utf8', timeout=90, check=True)
    return json.loads(result.stdout.splitlines()[-1])


def run_gpu_task_waves(tasks, function, gpu_ids):
    """Run one task per device per wave; multi-GPU tasks reserve the entire set.

    tasks is a sequence of (name, reserved_gpu_ids). The caller owns publication.
    The event lets process supervisors stop their children on notebook interrupt.
    """
    allowed = set(gpu_ids)
    wave, used, results = [], set(), {}
    cancel = Event()

    def drain():
        if not wave:
            return
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            futures = {pool.submit(function, name, cancel): name for name, _ in wave}
            try:
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
            except BaseException:
                cancel.set()
                for future in futures:
                    future.cancel()
                raise
        wave.clear()
        used.clear()

    for name, ids in tasks:
        reserved = set(ids)
        if not reserved <= allowed or len(reserved) != len(ids):
            raise ValueError(f'{name}: invalid GPU reservation {ids}')
        if used & reserved or (wave and not reserved):
            drain()
        # CPU tasks run alone to bound host RAM and CPU use.
        if not reserved:
            results[name] = function(name, cancel)
            continue
        wave.append((name, ids))
        used.update(reserved)
    drain()
    return results
