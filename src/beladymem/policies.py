"""Reference online eviction policies for the demand-paging replay.

Each policy implements three hooks used by :func:`beladymem.oracle.replay`:

* ``reset()``                 -- clear internal state before a replay.
* ``on_use(i, item, useful)`` -- observe a reference (called on hit and on the
  miss that triggers a fetch, before the eviction).
* ``evict_victim(i, cache)``  -- choose the item_id to evict on a full-cache miss.

None of these may look into the future (that is the oracle's privilege).
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict


class FIFO:
    """Evict the item that was fetched earliest. The replay's cache preserves
    fetch order, so the oldest key is the first-in item."""

    name = "fifo"

    def reset(self) -> None:
        pass

    def on_use(self, i: int, item: str, useful: bool) -> None:
        pass

    def evict_victim(self, i: int, cache: dict) -> str:
        return next(iter(cache))


class LRU:
    name = "lru"

    def __init__(self) -> None:
        self.recency: OrderedDict[str, int] = OrderedDict()

    def reset(self) -> None:
        self.recency = OrderedDict()

    def on_use(self, i: int, item: str, useful: bool) -> None:
        self.recency[item] = i
        self.recency.move_to_end(item)

    def evict_victim(self, i: int, cache: dict) -> str:
        for it in self.recency:
            if it in cache:
                return it
        return next(iter(cache))


class LFU:
    name = "lfu"

    def __init__(self) -> None:
        self.freq: dict[str, int] = defaultdict(int)
        self.tie: dict[str, int] = {}

    def reset(self) -> None:
        self.freq = defaultdict(int)
        self.tie = {}

    def on_use(self, i: int, item: str, useful: bool) -> None:
        self.freq[item] += 1
        self.tie[item] = i

    def evict_victim(self, i: int, cache: dict) -> str:
        # Lowest frequency; tie -> least-recently-referenced; then id.
        return min(cache, key=lambda it: (self.freq.get(it, 0), self.tie.get(it, 0), it))


class BlindOracle:
    """Evicts by a *predicted* next-use score (farthest predicted = victim).

    With true predictions it reproduces Belady (consistency); with corrupted
    predictions its worst case measures robustness. ``pred(item, i) -> float``.
    """

    name = "blind-oracle"

    def __init__(self, pred):
        self.pred = pred

    def reset(self) -> None:
        pass

    def on_use(self, i: int, item: str, useful: bool) -> None:
        pass

    def evict_victim(self, i: int, cache: dict) -> str:
        return max(cache, key=lambda it: (self.pred(it, i), it))


REFERENCE_POLICIES = {"lru": LRU, "lfu": LFU, "fifo": FIFO}
