"""Synthetic trace generators with planted structure (useful-USE streams).

This module intentionally uses a seeded RNG to *manufacture ground truth* for
the sensitivity gates. It is excluded from the honesty grep (which forbids
randomness on the metric path) precisely because manufacturing ground truth --
not measuring -- is its job. The canonical oracle reads only the useful-USE
references, so these generators emit USE events.
"""

from __future__ import annotations

import numpy as np

from .trace import Determinism, EventType, MemoryTrace, TraceEvent


def _use(t: int, item) -> TraceEvent:
    return TraceEvent(t=t, op=EventType.USE, item_id=str(item), useful=True)


def from_use_sequence(seq) -> MemoryTrace:
    """Build an EXACT_KEY trace from a flat sequence of item ids (all useful)."""
    return MemoryTrace(
        events=[_use(t, it) for t, it in enumerate(seq)],
        determinism=Determinism.EXACT_KEY,
    )


def cyclic_thrash(n_items: int = 5, cycles: int = 20) -> MemoryTrace:
    """The textbook LRU pathology: a working set of ``n_items`` referenced
    cyclically. With a budget below ``n_items`` LRU evicts exactly the item it is
    about to need (near-zero hits) while Belady keeps the soon-needed ones."""
    seq = []
    for _ in range(cycles):
        seq.extend(range(n_items))
    return from_use_sequence(seq)


def short_reuse_trace(seed: int = 0) -> MemoryTrace:
    """LRU-favouring (small OPT). Two "stale" items are hammered early to give
    them a very high frequency, then abandoned; a fresh working set is then
    referenced *cyclically* (so retention across the cycle matters). LFU clings
    to the stale high-frequency items, leaving too few slots for the cycle and
    thrashing it; LRU evicts the stale items and tracks the optimum."""
    rng = np.random.default_rng(seed)
    seq: list[str] = []
    # phase 1: inflate frequency of two "stale" items far above anything later
    stale = ["A", "B"]
    reps = int(rng.integers(16, 24))
    for _ in range(reps):
        seq.extend(stale)
    # phase 2: a fresh working set referenced cyclically (no immediate repeat)
    fresh = [f"n{k}" for k in range(3)]
    rounds = int(rng.integers(4, 7))
    for _ in range(rounds):
        seq.extend(fresh)
    return from_use_sequence(seq)


def freq_dominated_trace(seed: int = 0) -> MemoryTrace:
    """LFU-favouring (large OPT). A small hot set is referenced very often,
    interleaved with many cold singletons. LFU keeps the hot set (near OPT); LRU
    lets the cold scan evict hot items by recency (far from OPT)."""
    rng = np.random.default_rng(seed)
    hot = [f"H{k}" for k in range(4)]
    seq: list[str] = []
    cold = 0
    steps = int(rng.integers(380, 420))
    for _ in range(steps):
        # several hot references then one cold singleton
        for _ in range(int(rng.integers(2, 4))):
            seq.append(hot[int(rng.integers(0, len(hot)))])
        seq.append(f"K{cold}")
        cold += 1
    return from_use_sequence(seq)


# Budgets at which the divergence family is scored (see gates.G9). Tuned so LRU
# tracks the optimum on the short-reuse group while LFU dominates absolute hits
# on the frequency-dominated group -- the asymmetry that makes competitive ratio
# and FiFA regret disagree.
SHORT_REUSE_BUDGET = 3
FREQ_DOMINATED_BUDGET = 7


def divergence_family(n_each: int = 60, seed: int = 0):
    """Return ``(short_traces, freq_traces)`` -- the constructed family on which
    competitive ratio and FiFA regret rank LRU vs LFU in *opposite* order."""
    short = [short_reuse_trace(seed + i) for i in range(n_each)]
    freq = [freq_dominated_trace(seed + 1000 + i) for i in range(n_each)]
    return short, freq
