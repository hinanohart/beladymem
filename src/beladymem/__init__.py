"""beladymem -- score agent-memory eviction policies by competitive ratio
against the Belady MIN offline-optimal oracle (CPU-only, synthetic-validated).

This is a *measurement instrument*, not a memory system. See ``NON_CLAIMS``.
"""

from __future__ import annotations

from .report import NON_CLAIMS, ScoreReport
from .score import score
from .trace import Determinism, EventType, MemoryTrace, TraceEvent

__version__ = "0.1.0a1"

__all__ = [
    "__version__",
    "Determinism",
    "EventType",
    "MemoryTrace",
    "TraceEvent",
    "ScoreReport",
    "NON_CLAIMS",
    "score",
]
