"""Small shared contracts for preparation status and explicit variant selection."""
from __future__ import annotations


PREPARATION_FAILURE_STATES = {'failed', 'missing_assets', 'runtime_incompatible',
                              'inference_failed', 'timeout', 'resource_exceeded'}


def restoration_failure_state(error, *, phase='inference'):
    message = str(error).lower()
    if isinstance(error, (TimeoutError,)) or 'timeout' in message or 'timed out' in message:
        return 'timeout'
    if isinstance(error, MemoryError) or any(s in message for s in (
            'out of memory', 'exceeds max_', 'storage budget', 'no space left', 'disk full')):
        return 'resource_exceeded'
    if isinstance(error, FileNotFoundError):
        return 'missing_assets'
    return 'runtime_incompatible' if phase == 'preflight' else 'inference_failed'


def select_preparation_variants(mode, selected):
    classical = ['original', 'clahe', 'denoised', 'bicubic_x2', 'bicubic_x4']
    neural = ['realesrgan_x4', 'tadisr_x4', 'difftsr', 'unihir_restoration']
    if mode == 'selected':
        return list(selected)
    if mode == 'classical':
        return classical
    if mode == 'all_eligible':
        return classical + neural
    raise ValueError('VARIANT_MODE must be classical, selected or all_eligible')
