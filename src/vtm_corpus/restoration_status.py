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
    from .restoration_registry import VARIANT_SPECS, UNAVAILABLE_RESTORATION_METHODS
    classical = [name for name, spec in VARIANT_SPECS.items() if spec[1] in {
        'source_observation', 'deterministic_transformation'}]
    neural = [name for name, spec in VARIANT_SPECS.items() if spec[1] in {
        'neural_restoration', 'generative_reconstruction'} and name != 'difftsr']
    if mode == 'selected':
        unavailable = {n:UNAVAILABLE_RESTORATION_METHODS[n] for n in selected if n in UNAVAILABLE_RESTORATION_METHODS}
        if unavailable:
            raise ValueError('No runnable upstream/checkpoint for requested methods: '+str(unavailable))
        return list(selected)
    if mode == 'classical':
        return classical
    if mode == 'all_eligible':
        return classical + neural
    raise ValueError('VARIANT_MODE must be classical, selected or all_eligible')
