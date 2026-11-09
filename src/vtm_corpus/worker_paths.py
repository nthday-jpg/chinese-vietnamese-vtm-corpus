"""Path translation for a Linux WSL worker reading a Windows campaign."""
from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
import re


def to_wsl_path(value: str | Path) -> str:
    value = str(value)
    if value.startswith("/"):
        return value
    path = PureWindowsPath(value)
    if not re.fullmatch(r"[A-Za-z]:", path.drive) or not path.is_absolute():
        raise ValueError(f"WSL requires an absolute drive path or Linux path: {value}")
    return "/mnt/" + path.drive[0].lower() + "/" + "/".join(path.parts[1:])


def runtime_path(value: str | Path) -> Path:
    value = str(value)
    return Path(to_wsl_path(value)) if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", value) else Path(value)
