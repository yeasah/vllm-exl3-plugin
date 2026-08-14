"""Environment variables, read off the `EXL3_` prefix.

vLLM claims the `VLLM_` namespace: it validates the environment at startup and
warns about anything beginning with `VLLM_` that it does not recognize, so a
plugin putting its own settings there produces `Unknown vLLM environment
variable detected` for every one of them. `EXL3_` is the prefix exllamav3
already uses for its own switches (`EXL3_SM90_BARRIER`).

The old `VLLM_EXL3_*` spellings still work, with a warning, since two of them
were documented.
"""

from __future__ import annotations

import os

from .log import init_logger

logger = init_logger(__name__)

_WARNED: set[str] = set()


def get(name: str, default: str) -> str:
    """Read `EXL3_<name>`, falling back to the deprecated `VLLM_EXL3_<name>`."""
    value = os.environ.get(f"EXL3_{name}")
    if value is not None:
        return value
    legacy = os.environ.get(f"VLLM_EXL3_{name}")
    if legacy is None:
        return default
    if name not in _WARNED:
        _WARNED.add(name)
        logger.warning(
            "VLLM_EXL3_%s is deprecated and will stop being read: vLLM reserves "
            "the VLLM_ prefix and warns about anything in it that it does not "
            "recognize. Use EXL3_%s instead.",
            name,
            name,
        )
    return legacy


def flag(name: str) -> bool:
    return get(name, "0") == "1"
