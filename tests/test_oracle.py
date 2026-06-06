from functools import cache

import numpy as np
import pytest

from beladymem.metrics import competitive_ratio
from beladymem.oracle import (
    BeladyPolicy,
    belady_min,
    belady_min_hits_bruteforce,
)
from beladymem.policies import FIFO, LFU, LRU
from beladymem.synth import cyclic_thrash, from_use_sequence


def _rand_trace(rng, max_items=6, max_len=40):
    nit = int(rng.integers(2, max_items))
    length = int(rng.integers(5, max_len))
    return from_use_sequence([f"x{int(rng.integers(0, nit))}" for _ in range(length)])


def _true_optimum(use_seq, B):
    n = len(use_seq)

    @cache
    def best(i, cache):
        if i == n:
            return 0
        item = use_seq[i]
        if item in cache:
            return 1 + best(i + 1, cache)
        c = set(cache)
        if len(c) < B:
            c.add(item)
            return best(i + 1, frozenset(c))
        if B == 0:
            return best(i + 1, cache)
        bestv = 0
        for v in c:
            c2 = set(c)
            c2.discard(v)
            c2.add(item)
            bestv = max(bestv, best(i + 1, frozenset(c2)))
        return bestv

    return best(0, frozenset())


def test_brute_matches_bisect_belady():
    rng = np.random.default_rng(0)
    for _ in range(500):
        tr = _rand_trace(rng)
        B = int(rng.integers(1, 7))
        assert belady_min(tr, B)["useful_hits"] == belady_min_hits_bruteforce(tr, B)


def test_belady_is_true_optimum_small():
    rng = np.random.default_rng(1)
    for _ in range(800):
        nit = int(rng.integers(2, 5))
        seq = [f"x{int(rng.integers(0, nit))}" for _ in range(int(rng.integers(3, 12)))]
        B = int(rng.integers(1, nit + 1))
        assert belady_min(from_use_sequence(seq), B)["useful_hits"] == _true_optimum(tuple(seq), B)


def test_no_policy_beats_belady_exact_key():
    rng = np.random.default_rng(2)
    for _ in range(500):
        tr = _rand_trace(rng)
        B = int(rng.integers(1, 7))
        for pol in (LRU(), LFU(), FIFO()):
            r, _, _ = competitive_ratio(tr, B, pol)
            assert np.isnan(r) or r <= 1.0 + 1e-9


def test_belady_self_ratio_is_one():
    tr = cyclic_thrash(5, 10)
    r, _, _ = competitive_ratio(tr, 3, BeladyPolicy(tr.events))
    assert abs(r - 1.0) < 1e-12


def test_lru_thrashes_on_cyclic():
    tr = cyclic_thrash(5, 20)
    r, _, opt = competitive_ratio(tr, 3, LRU())
    assert opt["useful_hits"] > 0
    assert r < 0.05


@pytest.mark.parametrize("budget_mode", ["count", "bytes"])
def test_replay_runs_in_both_budget_modes(budget_mode):
    tr = from_use_sequence(["a", "b", "a", "c", "a", "b"])
    out = belady_min(tr, 2, budget_mode=budget_mode)
    assert 0 <= out["useful_hits"] <= out["total_useful"]
