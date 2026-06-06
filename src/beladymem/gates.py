"""Pre-registered sensitivity gates G1-G9.

These run on *synthetic ground truth* and prove the instrument discriminates
*before* it is ever pointed at real data. They are the structural defence
against the failure mode of a metric that looks rigorous but cannot actually
tell policies apart. ``beladymem gate`` runs them all and exits non-zero on any
failure. Thresholds are fixed here and must not be relaxed to force a pass.

* G1 -- optimality & ratio bound: Belady scores 1.0 against itself; no online
  policy ever exceeds 1.0; LRU < 1.0 on the cyclic pathology.
* G2 -- textbook LRU thrash: LRU collapses to ~0 on a cyclic stream below budget.
* G3 -- budget monotonicity: Belady hits are non-decreasing in the budget.
* G4 -- robustness < consistency: a corrupted predictor scores below a perfect one.
* G5 -- consistency == 1.0: a perfectly-predicting BlindOracle reproduces Belady.
* G6 -- lower-bound soundness: semantic / byte modes are labelled lower_bound,
  exact-key count mode is labelled optimal.
* G7 -- CI coverage: the bootstrap CI of the mean is well calibrated.
* G8 -- degeneracy: a zero-useful-hit trace yields NaN (not a crash) and is
  excluded from aggregates.
* G9 -- G-C differentiation (the load-bearing one): on a constructed family the
  Belady competitive ratio and FiFA-style regret rank LRU vs LFU in *opposite*
  order, with non-overlapping bootstrap CIs -- proving the competitive ratio is
  not a re-expression of best-fixed-policy regret.
"""

from __future__ import annotations

from math import isnan

import numpy as np

from .metrics import (
    cluster_bootstrap_ci,
    competitive_ratio,
    consistency_ratio,
    paired_diff_ci,
    policy_hits,
    robustness_ratio,
)
from .oracle import BeladyPolicy, belady_min
from .policies import LFU, LRU
from .synth import (
    FREQ_DOMINATED_BUDGET,
    SHORT_REUSE_BUDGET,
    cyclic_thrash,
    divergence_family,
    freq_dominated_trace,
    from_use_sequence,
    short_reuse_trace,
)
from .trace import Determinism, EventType, MemoryTrace, TraceEvent


def _random_use_trace(rng, max_items=6, max_len=40) -> MemoryTrace:
    nit = int(rng.integers(2, max_items))
    length = int(rng.integers(5, max_len))
    seq = [f"x{int(rng.integers(0, nit))}" for _ in range(length)]
    return from_use_sequence(seq)


def g1_optimality_and_bound() -> tuple[bool, str]:
    tr = cyclic_thrash(5, 20)
    self_ratio, _, _ = competitive_ratio(tr, 3, BeladyPolicy(tr.events))
    lru_ratio, _, _ = competitive_ratio(tr, 3, LRU())
    rng = np.random.default_rng(1)
    max_ratio = 0.0
    for _ in range(400):
        t = _random_use_trace(rng)
        b = int(rng.integers(1, 6))
        for pol in (LRU(), LFU()):
            r, _, _ = competitive_ratio(t, b, pol)
            if not isnan(r):
                max_ratio = max(max_ratio, r)
    ok = abs(self_ratio - 1.0) < 1e-9 and lru_ratio < 1.0 and max_ratio <= 1.0 + 1e-9
    return (
        ok,
        f"belady_self={self_ratio:.3f} lru_cyclic={lru_ratio:.3f} max_online_ratio={max_ratio:.6f}",
    )


def g2_textbook_thrash() -> tuple[bool, str]:
    tr = cyclic_thrash(5, 30)
    opt = belady_min(tr, 3)["useful_hits"]
    lru_ratio, _, _ = competitive_ratio(tr, 3, LRU())
    ok = opt > 0 and lru_ratio < 0.10
    return ok, f"opt_hits={opt} lru_ratio={lru_ratio:.3f} (<0.10)"


def g3_budget_monotonicity() -> tuple[bool, str]:
    rng = np.random.default_rng(2)
    ok = True
    detail = ""
    for _ in range(50):
        tr = _random_use_trace(rng, max_items=8, max_len=60)
        hits = [belady_min(tr, b)["useful_hits"] for b in range(1, 9)]
        if any(hits[k + 1] < hits[k] for k in range(len(hits) - 1)):
            ok = False
            detail = f"non-monotonic: {hits}"
            break
    return ok, detail or "belady hits non-decreasing in budget over 50 traces"


def g4_robustness_below_consistency() -> tuple[bool, str]:
    worst = 1.0
    for gen, b in ((lambda: cyclic_thrash(6, 25), 3), (lambda: freq_dominated_trace(0), 5)):
        tr = gen()
        cons = consistency_ratio(tr, b)
        rob = robustness_ratio(tr, b, corruption="invert")
        worst = min(worst, cons - rob)
    ok = worst > 0.10
    return ok, f"min(consistency - robustness)={worst:.3f} (>0.10)"


def g5_consistency_is_one() -> tuple[bool, str]:
    worst = 0.0
    for gen, b in ((lambda: cyclic_thrash(5, 20), 3), (lambda: freq_dominated_trace(1), 5)):
        tr = gen()
        cons = consistency_ratio(tr, b)
        worst = max(worst, abs(cons - 1.0))
    ok = worst < 1e-9
    return ok, f"max|consistency - 1.0|={worst:.2e}"


def g6_lower_bound_soundness() -> tuple[bool, str]:
    from .score import score

    exact = from_use_sequence(["a", "b", "a", "c", "a"])
    semantic = MemoryTrace(
        events=[
            TraceEvent(t=i, op=EventType.USE, item_id=it)
            for i, it in enumerate(["a", "b", "a", "c", "a"])
        ],
        determinism=Determinism.SEMANTIC,
    )
    r_exact = score(exact, 2, "lru")
    r_sem = score(semantic, 2, "lru")
    r_byte = score(exact, 2, "lru", budget_mode="bytes")
    ok = (
        r_exact.optimal_label == "optimal"
        and r_sem.optimal_label == "lower_bound"
        and r_byte.optimal_label == "lower_bound"
    )
    return (
        ok,
        f"exact={r_exact.optimal_label} semantic={r_sem.optimal_label} byte={r_byte.optimal_label}",
    )


def g7_ci_coverage() -> tuple[bool, str]:
    rng = np.random.default_rng(3)
    pop = []
    for s in range(400):
        tr = freq_dominated_trace(s) if s % 2 else short_reuse_trace(s)
        r, _, _ = competitive_ratio(tr, 5, LRU())
        if not isnan(r):
            pop.append(r)
    pop = np.array(pop)
    true_mean = float(pop.mean())
    reps, n, cover = 300, 30, 0
    for b in range(reps):
        idx = rng.integers(0, len(pop), n)
        _, lo, hi = cluster_bootstrap_ci(pop[idx], n_boot=600, seed=b)
        if lo <= true_mean <= hi:
            cover += 1
    cov = cover / reps
    ok = 0.88 <= cov <= 0.99
    return ok, f"bootstrap CI coverage={cov:.3f} (target 0.88-0.99)"


def g8_degeneracy() -> tuple[bool, str]:
    # All-distinct singletons with budget 1: every reference is a compulsory miss
    # for both policy and oracle -> oracle useful_hits == 0 -> ratio is NaN. The
    # NaN must be excluded from aggregation while a valid value survives.
    tr_deg = from_use_sequence([f"u{k}" for k in range(8)])
    r_deg, _, opt = competitive_ratio(tr_deg, 1, LRU())
    r_ok, _, _ = competitive_ratio(cyclic_thrash(5, 10), 3, LRU())
    agg_mean, _, _ = cluster_bootstrap_ci([r_ok, r_deg], n_boot=200)
    ok = isnan(r_deg) and opt["useful_hits"] == 0 and abs(agg_mean - r_ok) < 1e-9
    return (
        ok,
        f"degenerate_ratio={'nan' if isnan(r_deg) else r_deg} "
        f"opt_hits={opt['useful_hits']} aggregate_excludes_nan={abs(agg_mean - r_ok) < 1e-9}",
    )


def g9_gc_differentiation() -> tuple[bool, str]:
    """The G-C gate: prove competitive ratio ranks LRU vs LFU differently from
    FiFA-style best-fixed-policy regret. Between two policies the best-fixed
    comparator cancels, so regret ranking is the *absolute hits* ranking."""
    short, freq = divergence_family(n_each=60, seed=0)
    rl, rf, hl, hf = [], [], [], []
    for tr in short:
        a, _, _ = competitive_ratio(tr, SHORT_REUSE_BUDGET, LRU())
        c, _, _ = competitive_ratio(tr, SHORT_REUSE_BUDGET, LFU())
        rl.append(a)
        rf.append(c)
        hl.append(policy_hits(tr, SHORT_REUSE_BUDGET, LRU()))
        hf.append(policy_hits(tr, SHORT_REUSE_BUDGET, LFU()))
    for tr in freq:
        a, _, _ = competitive_ratio(tr, FREQ_DOMINATED_BUDGET, LRU())
        c, _, _ = competitive_ratio(tr, FREQ_DOMINATED_BUDGET, LFU())
        rl.append(a)
        rf.append(c)
        hl.append(policy_hits(tr, FREQ_DOMINATED_BUDGET, LRU()))
        hf.append(policy_hits(tr, FREQ_DOMINATED_BUDGET, LFU()))
    ratio_lru, ratio_lfu = np.array(rl), np.array(rf)
    hits_lru, hits_lfu = np.array(hl, float), np.array(hf, float)
    mrl, mrf = np.nanmean(ratio_lru), np.nanmean(ratio_lfu)
    mhl, mhf = hits_lru.mean(), hits_lfu.mean()
    dr = paired_diff_ci(ratio_lru, ratio_lfu, seed=1)
    dh = paired_diff_ci(hits_lru, hits_lfu, seed=2)
    flip = (mrl > mrf) != (mhl > mhf)
    ratio_favours_lru = dr[1] > 0  # CI lower bound positive
    regret_favours_lfu = dh[2] < 0  # CI upper bound negative (LFU has more hits)
    ok = flip and ratio_favours_lru and regret_favours_lfu
    return ok, (
        f"ratio LRU={mrl:.3f}>LFU={mrf:.3f} CI_diff[{dr[1]:.3f},{dr[2]:.3f}] | "
        f"hits LFU={mhf:.0f}>LRU={mhl:.0f} CI_diff[{dh[1]:.0f},{dh[2]:.0f}] | flip={flip}"
    )


ALL_GATES = [
    ("G1", "optimality & ratio bound", g1_optimality_and_bound),
    ("G2", "textbook LRU thrash", g2_textbook_thrash),
    ("G3", "budget monotonicity", g3_budget_monotonicity),
    ("G4", "robustness < consistency", g4_robustness_below_consistency),
    ("G5", "consistency == 1.0", g5_consistency_is_one),
    ("G6", "lower-bound soundness", g6_lower_bound_soundness),
    ("G7", "CI coverage", g7_ci_coverage),
    ("G8", "degeneracy handling", g8_degeneracy),
    ("G9", "G-C differentiation", g9_gc_differentiation),
]


def run_all_gates(verbose: bool = True) -> bool:
    all_ok = True
    for key, title, fn in ALL_GATES:
        ok, detail = fn()
        all_ok = all_ok and ok
        if verbose:
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {key} {title}: {detail}")
    if verbose:
        print(f"\n{'ALL GATES PASS' if all_ok else 'GATE FAILURE'}")
    return all_ok
