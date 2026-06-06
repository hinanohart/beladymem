"""Adapter for LongMemEval-style oracle annotations -> a useful-USE stream.

Honesty note (read this): LongMemEval ships **session-** and **turn-level** gold
annotations, not item/fact-level ones. This adapter therefore builds a
*session-as-item* reference stream: each evidence session id is one cache item,
and a question that depends on it is one useful retrieval.
That is enough to score eviction policies with **no LLM judge**, but it does NOT
score the item-level contents of a real mem0/Letta store -- mapping a store's
fact ids onto LongMemEval gold would need a heuristic or an LLM, which beladymem
deliberately does not claim. The data is not redistributed; pass a path to a
locally downloaded ``longmemeval_*`` JSON.

Expected record shape (a list of question dicts), tolerant to extra keys::

    {"question_id": "...",
     "haystack_session_ids": ["s1", "s2", ...],   # all sessions present
     "answer_session_ids":   ["s2", ...]}          # the evidence (useful) ones
"""

from __future__ import annotations

import json
from pathlib import Path

from ..trace import Determinism, EventType, MemoryTrace, TraceEvent


def longmemeval_records_to_trace(records, granularity: str = "session") -> MemoryTrace:
    """Convert a list of LongMemEval question records into a useful-USE trace.

    Each ``answer_session_ids`` entry becomes a useful retrieval of that session
    item, in question order. Only ``granularity="session"`` is supported: without
    item-level gold a turn-level stream would mislabel session items, so anything
    else is rejected fail-closed rather than emitting a provenance lie.
    """
    if granularity != "session":
        raise ValueError(
            f"granularity={granularity!r} is not supported; LongMemEval gold is "
            "session/turn level and only session-as-item is implemented without "
            "item-level gold. Pass granularity='session'."
        )
    events: list[TraceEvent] = []
    t = 0
    for rec in records:
        evidence = rec.get("answer_session_ids") or rec.get("evidence_session_ids") or []
        for sid in evidence:
            events.append(
                TraceEvent(
                    t=t,
                    op=EventType.USE,
                    item_id=f"session:{sid}",
                    useful=True,
                    meta={"granularity": granularity, "q": rec.get("question_id")},
                )
            )
            t += 1
    # Semantic, because session-as-item collapses many facts -> oracle is a lower bound.
    return MemoryTrace(events=events, determinism=Determinism.SEMANTIC)


def longmemeval_to_trace(path, granularity: str = "session") -> MemoryTrace:
    """Load a local LongMemEval JSON file and convert it to a trace."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data if isinstance(data, list) else data.get("questions", [])
    return longmemeval_records_to_trace(records, granularity=granularity)
