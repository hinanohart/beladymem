"""Competitive ratio, robustness/consistency, FiFA-style best-fixed-policy
regret, and cluster bootstrap confidence intervals.

The competitive ratio normalises each trace by its *own* clairvoyant optimum
(Belady MIN). FiFA-style regret instead normalises by the best *fixed* policy in
hindsight. These are different yardsticks: between two policies A and B the
best-fixed comparator cancels, so regret ranks them by *absolute* hits, whereas
the competitive ratio ranks them by *OPT-normalised* hits. The two rankings can
disagree -- gate G9 demonstrates exactly that on a constructed family.
"""

from __future__ import annotations

from math import inf, isnan, nan

import numpy as np

from .oracle import BeladyPolicy, _events, replay
from .policies import FIFO, LFU, LRU, BlindOracle


def policy_hits(trace, budget: int, policy, budget_mode: str = "count") -> int:
    return replay(trace, budget, policy, budget_mode)["useful_hits"]


def competitive_ratio(trace, budget: int, policy, budget_mode: str = "count"):
    """Return ``(ratio, policy_result, opt_result)``. ``ratio`` is ``nan`` when
    the oracle itself scores zero useful hits (degenerate, excluded by G8)."""
    events = _events(trace)
    opt = replay(events, budget, BeladyPolicy(events), budget_mode)
    pol = replay(events, budget, policy, budget_mode)
    if opt["useful_hits"] == 0:
        return nan, pol, opt
    return pol["useful_hits"] / opt["useful_hits"], pol, opt


DEFAULT_FIXED = (LRU, LFU, FIFO)


def best_fixed_policy_hits(
    trace, budget: int, policy_factories=DEFAULT_FIXED, budget_mode: str = "count"
) -> int:
    return max(policy_hits(trace, budget, f(), budget_mode) for f in policy_factories)


def fifa_regret(
    trace,
    budget: int,
    policy,
    policy_factories=DEFAULT_FIXED,
    budget_mode: str = "count",
) -> int:
    """Regret of ``policy`` against the best fixed policy in hindsight (the FiFA
    comparator). Lower is better."""
    best = best_fixed_policy_hits(trace, budget, policy_factories, budget_mode)
    return best - policy_hits(trace, budget, policy, budget_mode)


def consistency_ratio(trace, budget: int, budget_mode: str = "count") -> float:
    """BlindOracle fed the *true* next-use -> reproduces Belady -> ~1.0."""
    events = _events(trace)
    bp = BeladyPolicy(events)
    bo = BlindOracle(lambda it, i: bp.next_use(it, i))
    opt = replay(events, budget, BeladyPolicy(events), budget_mode)
    pol = replay(events, budget, bo, budget_mode)
    if opt["useful_hits"] == 0:
        return nan
    return pol["useful_hits"] / opt["useful_hits"]


def robustness_ratio(
    trace,
    budget: int,
    corruption: str = "invert",
    seed: int = 0,
    budget_mode: str = "count",
) -> float:
    """BlindOracle fed *corrupted* predictions (worst-case proxy). ``invert`` is
    deterministic and is the canonical robustness corruption."""
    events = _events(trace)
    bp = BeladyPolicy(events)
    items = sorted({e.item_id for e in events})
    rng = np.random.default_rng(seed)

    if corruption == "invert":

        def pred(it: str, i: int) -> float:
            nu = bp.next_use(it, i)
            return -nu if nu != inf else -1e18
    elif corruption == "shuffle":
        perm = {it: float(v) for it, v in zip(items, rng.permutation(len(items)), strict=True)}

        def pred(it: str, i: int) -> float:
            return perm[it]
    else:  # gaussian noise on the true next-use

        def pred(it: str, i: int) -> float:
            nu = bp.next_use(it, i)
            base = 0.0 if nu == inf else float(nu)
            return base + float(rng.normal(0, 5))

    bo = BlindOracle(pred)
    opt = replay(events, budget, BeladyPolicy(events), budget_mode)
    pol = replay(events, budget, bo, budget_mode)
    if opt["useful_hits"] == 0:
        return nan
    return pol["useful_hits"] / opt["useful_hits"]


def cluster_bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of the mean over independent trace-level values
    (each trace = one cluster). Each cluster contributes a single scalar, so this
    is an ordinary nonparametric bootstrap of the mean -- the "cluster" framing is
    only to make the resampling unit (one whole trace) explicit; there is no
    within-cluster resampling. NaNs are dropped (G8). Returns ``(mean, lo, hi)``.
    """
    vals = np.array([v for v in values if not (isinstance(v, float) and isnan(v))], dtype=float)
    if vals.size == 0:
        return (nan, nan, nan)
    rng = np.random.default_rng(seed)
    n = vals.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = vals[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (float(vals.mean()), lo, hi)


def paired_diff_ci(values_a, values_b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Bootstrap CI of the paired mean difference ``mean(a) - mean(b)`` over the
    common (non-NaN) clusters. Used by G9 to test rank reversal."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    if a.size == 0:
        return (nan, nan, nan)
    rng = np.random.default_rng(seed)
    n = a.size
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return (float((a - b).mean()), lo, hi)
