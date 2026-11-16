"""Storage accounting and resumable stops for offline image preparation."""
from __future__ import annotations

import errno
from pathlib import Path
import shutil


class PreparationStorageLimit(RuntimeError):
    """The requested run needs more space; completed objects remain reusable."""


def preparation_storage_usage(root):
    seen = set()
    total = 0
    for path in Path(root).rglob('*'):
        if path.is_file():
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino) if stat.st_ino else str(path.resolve())
            if identity not in seen:
                seen.add(identity)
                total += stat.st_size
    return total


def preparation_storage_budget(root, max_bytes=None, reserve_bytes=2 * 1024**3):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
        raise ValueError('MAX_STORAGE_BYTES must be None or a positive integer')
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError('STORAGE_RESERVE_BYTES must be a nonnegative integer')
    used = preparation_storage_usage(root)
    free = shutil.disk_usage(root).free
    return dict(root=root, used=used,
                limit=used + max(0, free - reserve_bytes) if max_bytes is None else max_bytes,
                reserve_bytes=reserve_bytes, automatic=max_bytes is None)


def preparation_storage_check(budget, path, byte_count):
    """Check both net dataset growth and the space needed by the atomic temporary."""
    if budget.get('stop_reason'):
        raise PreparationStorageLimit(budget['stop_reason'])
    path = Path(path)
    previous = path.stat().st_size if path.is_file() else 0
    growth = byte_count - previous
    free = shutil.disk_usage(budget['root']).free
    reason = None
    if budget['used'] + growth > budget['limit']:
        reason = (f"max_storage_bytes reached: used={budget['used']}, "
                  f"additional={max(0, growth)}, limit={budget['limit']}")
    elif free < byte_count + budget['reserve_bytes']:
        reason = (f"Storage reserve reached: free={free}, atomic_write={byte_count}, "
                  f"reserve={budget['reserve_bytes']}")
    if reason:
        budget['stop_reason'] = reason
        raise PreparationStorageLimit(reason)
    return growth


def is_preparation_storage_error(error):
    if isinstance(error, PreparationStorageLimit):
        return True
    if isinstance(error, OSError) and error.errno in (errno.ENOSPC, errno.EDQUOT):
        return True
    message = str(error).lower()
    return any(text in message for text in (
        'max_storage_bytes', 'no space left on device', 'disk full', 'disk quota exceeded',
        'storage reserve reached'))
