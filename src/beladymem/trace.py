"""Memory-access trace contract for beladymem.

A trace is a time-ordered list of events on an external agent-memory store.
The canonical oracle reads only the **useful retrievals**; the other event kinds
are kept for adapter fidelity and provenance:

* ``USE`` -- a retrieval. ``useful=True`` marks a retrieval that materially
  helped: these form the classical cache *reference stream* that the competitive
  ratio scores. ``useful=False`` marks a noise retrieval, excluded entirely.
* ``WRITE`` -- the moment an item is stored. Recorded for fidelity, but the
  canonical oracle is defined on the useful-retrieval references and does not
  read WRITE events (see :mod:`beladymem.oracle`).
* ``EVICT`` -- provenance only; the scorer re-simulates evictions under each
  policy, so recorded EVICT events are ignored during scoring.

Determinism of item identity decides the oracle's standing:

* ``EXACT_KEY`` -- stable ids, count budget -> Belady MIN is the *true optimum*
  (competitive ratio in ``[0, 1]``).
* ``SEMANTIC`` -- fuzzy match / merge -> Belady MIN is a *lower bound only*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class EventType(str, Enum):
    WRITE = "write"
    USE = "use"
    EVICT = "evict"


class Determinism(str, Enum):
    EXACT_KEY = "exact_key"
    SEMANTIC = "semantic"


@dataclass
class TraceEvent:
    t: int
    op: EventType
    item_id: str
    size: int = 1
    useful: bool = True
    ts: float | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.op, EventType):
            self.op = EventType(self.op)
        if self.size < 1:
            raise ValueError(f"size must be >= 1, got {self.size} for item {self.item_id!r}")
        if not self.item_id:
            raise ValueError("item_id must be a non-empty stable identifier")

    def to_dict(self) -> dict:
        d: dict = {
            "t": self.t,
            "op": self.op.value,
            "item_id": self.item_id,
            "size": self.size,
        }
        if self.op is EventType.USE:
            d["useful"] = self.useful
        if self.ts is not None:
            d["ts"] = self.ts
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TraceEvent:
        return cls(
            t=int(d["t"]),
            op=EventType(d["op"]),
            item_id=str(d["item_id"]),
            size=int(d.get("size", 1)),
            useful=bool(d.get("useful", True)),
            ts=d.get("ts"),
            meta=dict(d.get("meta", {})),
        )


@dataclass
class MemoryTrace:
    events: list[TraceEvent]
    determinism: Determinism = Determinism.EXACT_KEY

    def __post_init__(self) -> None:
        if not isinstance(self.determinism, Determinism):
            self.determinism = Determinism(self.determinism)
        last_t: int | None = None
        for e in self.events:
            if last_t is not None and e.t < last_t:
                raise ValueError(f"trace time must be non-decreasing; saw t={e.t} after t={last_t}")
            last_t = e.t

    def __len__(self) -> int:
        return len(self.events)

    @property
    def n_useful(self) -> int:
        return sum(1 for e in self.events if e.op is EventType.USE and e.useful)

    def to_jsonl(self, path) -> None:
        p = Path(path)
        with p.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"determinism": self.determinism.value}) + "\n")
            for e in self.events:
                f.write(json.dumps(e.to_dict()) + "\n")

    @classmethod
    def from_jsonl(cls, path) -> MemoryTrace:
        p = Path(path)
        events: list[TraceEvent] = []
        determinism = Determinism.EXACT_KEY
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if "determinism" in d and "op" not in d:
                    determinism = Determinism(d["determinism"])
                    continue
                events.append(TraceEvent.from_dict(d))
        return cls(events=events, determinism=determinism)
