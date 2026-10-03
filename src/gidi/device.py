"""Single place for compute-device selection (mps > cuda > cpu)."""

from __future__ import annotations

from typing import Literal

import torch

DevicePreference = Literal["auto", "mps", "cuda", "cpu"]


def available_devices() -> list[str]:
    """Return usable device types in preference order."""
    devices: list[str] = []
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")
    return devices


def resolve_device(preference: DevicePreference | str = "auto") -> torch.device:
    """Resolve a device preference to a ``torch.device``.

    ``"auto"`` picks the first available of mps, cuda, cpu. An explicit device that is
    not available raises ``RuntimeError`` instead of silently falling back.
    """
    preference = preference.lower()
    available = available_devices()
    if preference == "auto":
        return torch.device(available[0])
    if preference not in ("mps", "cuda", "cpu"):
        raise ValueError(f"unknown device preference: {preference!r}")
    if preference not in available:
        raise RuntimeError(f"device {preference!r} requested but not available (have {available})")
    return torch.device(preference)
