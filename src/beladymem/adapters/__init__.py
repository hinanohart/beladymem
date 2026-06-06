"""Adapters that turn external memory logs into a :class:`beladymem.MemoryTrace`.

Every adapter emits the same intermediate representation -- a sequence of events
with stable item ids -- so the oracle and metrics never see adapter specifics.
"""

from __future__ import annotations

from .jsonl import load_jsonl
from .longmemeval import longmemeval_to_trace

__all__ = ["load_jsonl", "longmemeval_to_trace"]
