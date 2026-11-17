"""Small shared contracts for preparation status and explicit variant selection."""
from __future__ import annotations


PREPARATION_FAILURE_STATES = {'failed', 'missing_assets', 'runtime_incompatible',
                              'inference_failed', 'timeout', 'resource_exceeded'}


def restoration_failure_state(error, *, phase='inference'):
    message = str(error).lower()
    if isinstance(error, (TimeoutError,)) or 'timeout' in message or 'timed out' in message:
        return 'timeout'
    if isinstance(error, MemoryError) or any(s in message for s in (
            'out of memory', 'exceeds max_', 'max_storage_bytes', 'storage reserve',
            'storage budget', 'no space left', 'disk full')):
        return 'resource_exceeded'
    if isinstance(error, FileNotFoundError):
        return 'missing_assets'
    return 'runtime_incompatible' if phase == 'preflight' else 'inference_failed'


def preparation_policy_variants():
    """Active methods; historical registry identities remain readable."""
    return ('original', 'tadisr_x4', 'difftsr', 'unihir_restoration')


def validate_preparation_selection(selected):
    allowed = set(preparation_policy_variants()) | {'tadisr_text_mask'}
    rejected = sorted(set(selected) - allowed)
    if rejected:
        raise ValueError('Methods removed from active restoration policy: ' + ', '.join(rejected))
    return list(selected)


def preparation_run_groups():
    """Source baseline only; text SR must be selected explicitly."""
    return {'baseline': ['original'], 'light': ['original']}


def select_preparation_variants(mode, selected):
    from .restoration_registry import VARIANT_SPECS, UNAVAILABLE_RESTORATION_METHODS
    groups=preparation_run_groups()
    if mode in groups:return list(groups[mode])
    if mode == 'full':return list(preparation_policy_variants())
    if mode == 'selective':return ['original','unihir_restoration']
    if mode in VARIANT_SPECS:return validate_preparation_selection(list(dict.fromkeys(['original',mode])))
    if mode == 'selected':
        unavailable = {n:UNAVAILABLE_RESTORATION_METHODS[n] for n in selected if n in UNAVAILABLE_RESTORATION_METHODS}
        if unavailable:
            raise ValueError('No runnable upstream/checkpoint for requested methods: '+str(unavailable))
        return validate_preparation_selection(selected)
    if mode == 'classical':
        return ['original']
    if mode == 'all_eligible':
        # DiffTSR still requires explicit selection and reviewed horizontal regions.
        return [name for name in preparation_policy_variants() if name != 'difftsr']
    raise ValueError('VARIANT_MODE must be full, selective, a run group, method name, classical, selected or all_eligible')
