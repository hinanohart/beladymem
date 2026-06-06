import numpy as np

from beladymem.metrics import (
    cluster_bootstrap_ci,
    competitive_ratio,
    consistency_ratio,
    fifa_regret,
    paired_diff_ci,
    policy_hits,
    robustness_ratio,
)
from beladymem.policies import FIFO, LFU, LRU
from beladymem.synth import cyclic_thrash, freq_dominated_trace, from_use_sequence


def test_consistency_is_exactly_one():
    tr = cyclic_thrash(5, 12)
    assert abs(consistency_ratio(tr, 3) - 1.0) < 1e-12


def test_robustness_below_consistency():
    tr = cyclic_thrash(6, 20)
    assert robustness_ratio(tr, 3, corruption="invert") < consistency_ratio(tr, 3)


def test_fifa_regret_nonnegative_against_self_class():
    tr = freq_dominated_trace(0)
    # regret of the best fixed policy in the comparator is 0
    best_is_lru = policy_hits(tr, 5, LRU()) >= max(
        policy_hits(tr, 5, LFU()), policy_hits(tr, 5, FIFO())
    )
    r = fifa_regret(tr, 5, LRU() if best_is_lru else LFU())
    assert r >= 0


def test_competitive_ratio_in_unit_interval():
    tr = freq_dominated_trace(3)
    for pol in (LRU(), LFU(), FIFO()):
        r, _, _ = competitive_ratio(tr, 5, pol)
        assert 0.0 <= r <= 1.0


def test_cluster_bootstrap_excludes_nan():
    mean, lo, hi = cluster_bootstrap_ci([0.5, float("nan"), 0.5], n_boot=200)
    assert abs(mean - 0.5) < 1e-9
    assert lo <= mean <= hi


def test_cluster_bootstrap_all_nan_is_nan():
    mean, lo, hi = cluster_bootstrap_ci([float("nan"), float("nan")], n_boot=50)
    assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)


def test_paired_diff_ci_detects_positive_difference():
    a = [1.0] * 30
    b = [0.0] * 30
    mean, lo, hi = paired_diff_ci(a, b, n_boot=500)
    assert mean == 1.0 and lo > 0


def test_paired_diff_ci_drops_unpaired_nan():
    a = [1.0, float("nan"), 1.0]
    b = [0.0, 0.0, float("nan")]
    mean, lo, hi = paired_diff_ci(a, b, n_boot=100)
    assert abs(mean - 1.0) < 1e-9


def test_policy_hits_matches_competitive_numerator():
    tr = from_use_sequence(["a", "b", "a", "c", "a", "b", "c"])
    r, pol, opt = competitive_ratio(tr, 2, LRU())
    assert pol["useful_hits"] == policy_hits(tr, 2, LRU())
