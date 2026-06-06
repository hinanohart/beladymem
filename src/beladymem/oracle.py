"""Belady MIN offline-optimal eviction oracle and the demand-paging replay.

Canonical model (this is what the competitive ratio and the NON-CLAIMs mean):

* The sequence of **useful retrievals** (``USE`` events with ``useful=True``) is
  treated as a classical cache *reference stream*, and the memory budget ``B`` is
  a cache of size ``B``.
* Replay is classical **demand paging**: on a reference, a hit is scored if the
  item is resident; on a miss the item is fetched (admitted), evicting a victim
  first if the cache is full. The first reference to an item is a compulsory
  miss, exactly as in standard cache analysis.
* The **Belady MIN** oracle evicts the resident item whose next useful retrieval
  is farthest in the future (Belady, 1966). It is the offline optimum for this
  stream, so ``competitive_ratio = policy_hits / belady_hits`` lies in ``[0, 1]``
  for every exact-key, count-budget trace.
* ``WRITE`` events are recorded for adapter fidelity but do **not** change the
  canonical oracle, which is defined on the useful-retrieval references. ``EVICT``
  events are provenance and are ignored.

Under semantic (fuzzy) matching, or a byte budget, Belady MIN is only a *lower
bound* on the true optimum (see ``beladymem.score``).
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from math import inf

from .trace import EventType, MemoryTrace, TraceEvent


def _events(trace) -> list[TraceEvent]:
    return trace.events if isinstance(trace, MemoryTrace) else trace


class BeladyPolicy:
    """Offline-optimal eviction for the demand-paging model: evict the resident
    item whose next useful retrieval is farthest in the future (ties broken by
    item_id, for determinism)."""

    name = "belady-min"

    def __init__(self, events: list[TraceEvent]):
        self.uses: dict[str, list[int]] = defaultdict(list)
        for i, e in enumerate(events):
            if e.op is EventType.USE and e.useful:
                self.uses[e.item_id].append(i)

    def reset(self) -> None:
        pass

    def next_use(self, item: str, i: int) -> float:
        lst = self.uses.get(item)
        if not lst:
            return inf
        k = bisect_right(lst, i)
        return lst[k] if k < len(lst) else inf

    def on_use(self, i: int, item: str, useful: bool) -> None:
        pass

    def evict_victim(self, i: int, cache: dict) -> str:
        return max(cache, key=lambda it: (self.next_use(it, i), it))


def _over(cache_size: int, cache_bytes: int, incoming: int, budget: int, mode: str) -> bool:
    """True while the cache cannot yet hold one more incoming item of the given
    size (so a victim must be evicted)."""
    if mode == "count":
        return cache_size >= budget
    return cache_bytes + incoming > budget


def replay(trace, budget: int, policy, budget_mode: str = "count") -> dict:
    """Demand-paging replay of ``policy`` on the useful-retrieval stream.

    Returns ``{"useful_hits", "total_useful", "premature"}`` where ``premature``
    lists ``(item, evicted_at_ref, missed_at_ref)`` -- items the policy evicted
    and then missed on a later retrieval.
    """
    events = _events(trace)
    policy.reset()
    cache: dict[str, int] = {}
    cache_bytes = 0
    useful_hits = 0
    total_useful = 0
    premature: list[tuple[str, int, int]] = []
    last_evict: dict[str, int] = {}
    for i, e in enumerate(events):
        if not (e.op is EventType.USE and e.useful):
            continue  # WRITE / EVICT / noise USE: not part of the canonical stream
        total_useful += 1
        if e.item_id in cache:
            useful_hits += 1
            policy.on_use(i, e.item_id, True)
            continue
        # miss -> fetch (demand paging)
        if e.item_id in last_evict:
            premature.append((e.item_id, last_evict[e.item_id], i))
        policy.on_use(i, e.item_id, True)
        while cache and _over(len(cache), cache_bytes, e.size, budget, budget_mode):
            victim = policy.evict_victim(i, cache)
            if victim is None or victim not in cache:
                # Fail closed: an invalid victim would let the cache grow past the
                # budget and silently produce a competitive ratio > 1. Refuse
                # rather than report a wrong number.
                raise ValueError(
                    f"policy {getattr(policy, 'name', policy)!r} returned an "
                    f"invalid eviction victim {victim!r}; the budget invariant "
                    f"would be violated"
                )
            cache_bytes -= cache[victim]
            del cache[victim]
            last_evict[victim] = i
        if budget_mode == "count" and budget < 1:
            # B=0: never admit. The eviction loop above has already drained the
            # cache (it is provably empty here), so the cache stays empty and the
            # ratio is a well-defined 0/0 -> nan (G8), never > 1.
            continue
        cache[e.item_id] = e.size
        cache_bytes += e.size
    return {
        "useful_hits": useful_hits,
        "total_useful": total_useful,
        "premature": premature,
    }


def belady_min(trace, budget: int, budget_mode: str = "count") -> dict:
    events = _events(trace)
    return replay(events, budget, BeladyPolicy(events), budget_mode)


def belady_min_hits_bruteforce(trace, budget: int) -> int:
    """Independent reference: classical Belady (evict farthest next reference)
    on the useful-USE stream, with a forward scan instead of binary search.
    Used only to cross-check :class:`BeladyPolicy` in the test-suite."""
    events = _events(trace)
    refs = [e.item_id for e in events if e.op is EventType.USE and e.useful]

    def next_ref(item: str, after_pos: int) -> float:
        for j in range(after_pos + 1, len(refs)):
            if refs[j] == item:
                return j
        return inf

    cache: set[str] = set()
    hits = 0
    for pos, item in enumerate(refs):
        if item in cache:
            hits += 1
            continue
        while len(cache) >= budget and cache:
            victim = max(cache, key=lambda it: (next_ref(it, pos), it))
            cache.discard(victim)
        if budget >= 1:
            cache.add(item)
    return hits
