"""Loggers that vLLM will actually print.

vLLM configures logging for the `vllm` logger only (`vllm/logger.py`'s
dictConfig names a single logger, with `propagate: False`). A plugin module
called `vllm_exl3_plugin.*` therefore sits outside that hierarchy: its records
propagate to a root logger with no handler, where `logging.lastResort` emits
WARNING and above unformatted and drops INFO entirely.

Naming our loggers under `vllm.` puts them back inside the configured
hierarchy, so they pick up vLLM's handler, its formatting, and
`VLLM_LOGGING_LEVEL`.
"""

from __future__ import annotations

from vllm.logger import init_logger as _vllm_init_logger


def init_logger(name: str):
    """`vllm.logger.init_logger`, but attached to the tree vLLM configures."""
    return _vllm_init_logger(name if name.startswith("vllm.") else f"vllm.{name}")
