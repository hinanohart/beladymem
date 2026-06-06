"""Pass-through adapter for the native JSONL trace format."""

from __future__ import annotations

from ..trace import MemoryTrace


def load_jsonl(path) -> MemoryTrace:
    """Load a native beladymem JSONL trace (see :meth:`MemoryTrace.from_jsonl`)."""
    return MemoryTrace.from_jsonl(path)
