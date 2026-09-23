"""Storage safety checks for experiments that invoke external runtimes."""

from __future__ import annotations

import ctypes
import os
from collections.abc import Callable, Iterable
from pathlib import Path

DRIVE_REMOVABLE = 2
PathLike = str | os.PathLike[str]


class StorageSafetyError(RuntimeError):
    """Raised before a benchmark would access a removable storage volume."""


def _windows_drive_type(path: PathLike) -> int | None:
    if os.name != "nt":
        return None
    anchor = Path(path).anchor
    if not anchor:
        return None
    try:
        return int(ctypes.windll.kernel32.GetDriveTypeW(anchor))
    except (AttributeError, OSError):
        return None


def assert_storage_safe(
    paths: Iterable[PathLike],
    *,
    allow_removable: bool = False,
    drive_type_fn: Callable[[str], int | None] | None = None,
) -> None:
    """Reject removable paths unless the caller explicitly opts in.

    ``drive_type_fn`` is injectable so the policy is testable without touching a real
    volume. Relative paths and non-Windows paths are left unchanged.
    """

    if allow_removable:
        return
    resolver = drive_type_fn or _windows_drive_type
    removable = []
    for path in paths:
        value = os.fspath(path)
        if resolver(value) == DRIVE_REMOVABLE:
            removable.append(value)
    if removable:
        joined = ", ".join(removable)
        raise StorageSafetyError(
            f"refusing benchmark access to removable storage: {joined}; "
            "copy artifacts to a stable volume or pass --allow-removable-storage explicitly"
        )
