"""Central device resolution for every torch-backed component (embedding
backbone, projection-head training loop). Single source of truth for the
`FLORALENS_DEVICE` environment variable so the backbone, the trainer, and the
API config all agree on which device a model/tensor lives on.

`FLORALENS_DEVICE` accepts:
  - "auto" (default) -> cuda if `torch.cuda.is_available()`, else cpu.
  - "cuda"            -> cuda, or raise if no CUDA device is visible (fail
                          loud rather than silently falling back when the
                          caller explicitly asked for a GPU).
  - "cpu"             -> always cpu, regardless of GPU availability.
"""
from __future__ import annotations

import os

import torch

DEVICE_ENV_VAR = "FLORALENS_DEVICE"
_VALID_PREFERENCES = ("auto", "cuda", "cpu")


def resolve_device(preference: str | None = None) -> torch.device:
    """Resolve a device preference to a concrete `torch.device`.

    `preference` defaults to the `FLORALENS_DEVICE` environment variable
    (itself defaulting to "auto") when not given explicitly.
    """
    pref = (preference if preference is not None else os.environ.get(DEVICE_ENV_VAR, "auto"))
    pref = pref.strip().lower()
    if pref not in _VALID_PREFERENCES:
        raise ValueError(
            f"unknown device preference {pref!r}; expected one of {_VALID_PREFERENCES}"
        )
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "FLORALENS_DEVICE=cuda requested but torch.cuda.is_available() is False"
            )
        return torch.device("cuda")
    # "auto"
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
