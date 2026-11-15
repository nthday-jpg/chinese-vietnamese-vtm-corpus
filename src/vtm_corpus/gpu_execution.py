"""GPU placement for isolated inference workers; no CUDA import in the notebook."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from threading import Event, Lock
from collections import deque
import time


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


def choose_gpu_job(pending, active, timings, *, preferred=None, failed=()):
    """Start untouched models first, then help the longest estimated backlog.

    A helper is worthwhile only when its share of remaining inference exceeds
    the measured model load cost. Without measurements require two queued pages.
    Running pages are never moved between workers.
    """
    candidates = [n for n, count in pending.items() if count and n not in failed]
    fresh = [n for n in candidates if not active.get(n, 0)]
    if fresh:
        return preferred if preferred in fresh else max(fresh, key=lambda n: pending[n])
    def worthwhile(name):
        timing = timings.get(name, {})
        if not timing.get('page_seconds'):
            return pending[name] >= 2
        return (pending[name] * timing['page_seconds'] / (active.get(name, 0) + 1)
                > timing.get('load_seconds', 0))
    candidates = [n for n in candidates if worthwhile(n)]
    return max(candidates, key=lambda n: pending[n] * timings.get(n, {}).get(
        'page_seconds', 1) / (active.get(n, 0) + 1), default=None)


def run_gpu_page_jobs(jobs, function, gpu_ids, *, preferred=None, cancel_event=None, reserved_jobs=None):
    """Persistent worker per GPU/model; idle cards steal unclaimed pages.

    function(name, page_iterator, gpu_ids, slot, cancel) owns one model load.
    jobs maps single-GPU models to pages. reserved_jobs maps sharded models to
    (pages, GPU IDs); one copy holds its reservation while spare cards do other
    work. No framework is imported.
    """
    queues = {name: deque(pages) for name, pages in jobs.items()}
    reserved_jobs = reserved_jobs or {}
    pending_heavy = list(reserved_jobs)
    for name,(pages,ids) in reserved_jobs.items():
        if name in queues or not ids or not set(ids) <= set(gpu_ids) or len(set(ids)) != len(ids):
            raise ValueError(f'{name}: invalid shared-model reservation')
        queues[name] = deque(pages)
    active, timings, results = {}, {}, {}
    lock, cancel = Lock(), cancel_event if cancel_event is not None else Event()

    class PageStream:
        def __init__(self, name):
            self.name = name
        def __iter__(self):
            first = True
            while not cancel.is_set():
                with lock:
                    if not queues[self.name]:
                        return
                    page = queues[self.name].popleft()
                started = time.monotonic()
                yield page
                with lock:
                    old = timings.setdefault(self.name, {})
                    seconds = time.monotonic() - started
                    if first:
                        seconds = max(0.001, seconds - old.get('load_seconds', 0))
                    old['page_seconds'] = seconds
                    first = False
        def model_loaded(self, seconds):
            with lock:
                timings.setdefault(self.name, {})['load_seconds'] = seconds or 0

    free, running = set(gpu_ids), {}
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        def launch(name, ids):
            active[name] = active.get(name, 0) + 1
            free.difference_update(ids)
            slot = '-'.join(map(str, ids))
            running[pool.submit(function, name, PageStream(name), list(ids), slot, cancel)] = (name, ids, slot)
        try:
            while not cancel.is_set():
                with lock:
                    for name in pending_heavy[:]:
                        ids = reserved_jobs[name][1]
                        if set(ids) <= free:
                            launch(name, ids)
                            pending_heavy.remove(name)
                    for gpu in sorted(free):
                        name = choose_gpu_job({n: len(queues[n]) for n in jobs}, active,
                                              timings, preferred=(preferred or {}).get(gpu))
                        if name is not None:
                            launch(name, [gpu])
                if not running:
                    break
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    name, ids, slot = running.pop(future)
                    results[(name, slot)] = future.result()
                    active[name] -= 1
                    free.update(ids)
        except BaseException:
            cancel.set()
            raise
    return results
