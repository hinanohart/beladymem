import pytest

from beladymem.policies import LRU
from beladymem.report import NON_CLAIMS, ScoreReport
from beladymem.score import score, score_many
from beladymem.synth import cyclic_thrash, from_use_sequence
from beladymem.trace import Determinism, EventType, MemoryTrace, TraceEvent


def test_score_by_name_class_and_instance_agree():
    tr = cyclic_thrash(5, 10)
    a = score(tr, 3, "lru").competitive_ratio
    b = score(tr, 3, LRU).competitive_ratio
    c = score(tr, 3, LRU()).competitive_ratio
    assert a == b == c


def test_score_unknown_policy_raises():
    tr = from_use_sequence(["a", "b", "a"])
    with pytest.raises(ValueError):
        score(tr, 2, "nope")


def test_optimal_label_exact_key_count():
    tr = from_use_sequence(["a", "b", "a", "c", "a"])
    assert score(tr, 2, "lru").optimal_label == "optimal"


def test_lower_bound_label_for_semantic():
    tr = MemoryTrace(
        events=[TraceEvent(t=i, op=EventType.USE, item_id=x) for i, x in enumerate("abaca")],
        determinism=Determinism.SEMANTIC,
    )
    assert score(tr, 2, "lru").optimal_label == "lower_bound"


def test_lower_bound_label_for_bytes():
    tr = from_use_sequence(["a", "b", "a", "c", "a"])
    assert score(tr, 2, "lru", budget_mode="bytes").optimal_label == "lower_bound"


def test_score_report_roundtrip_and_summary():
    tr = cyclic_thrash(4, 8)
    rep = score(tr, 2, "lru")
    assert isinstance(rep, ScoreReport)
    d = rep.to_dict()
    assert d["policy"] == "lru"
    assert len(d["non_claims"]) == 3
    assert "competitive_ratio" in rep.summary()


def test_non_claims_are_three_and_verbatim_available():
    assert len(NON_CLAIMS) == 3
    assert all(isinstance(s, str) and s for s in NON_CLAIMS)


def test_score_many_returns_one_report_per_policy():
    tr = cyclic_thrash(5, 10)
    reports = score_many(tr, 3, ["lru", "lfu", "fifo"])
    assert [r.policy for r in reports] == ["lru", "lfu", "fifo"]
