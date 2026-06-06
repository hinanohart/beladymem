"""Top-level scoring entry point: ``score(trace, budget, policy, budget_mode)``."""

from __future__ import annotations

from .metrics import competitive_ratio
from .policies import REFERENCE_POLICIES
from .report import ScoreReport
from .trace import Determinism, MemoryTrace


def _resolve_policy(policy):
    """Accept a policy instance, a policy class, or a reference name string."""
    if isinstance(policy, str):
        key = policy.lower()
        if key not in REFERENCE_POLICIES:
            raise ValueError(f"unknown policy {policy!r}; known: {sorted(REFERENCE_POLICIES)}")
        return REFERENCE_POLICIES[key]()
    if isinstance(policy, type):
        return policy()
    return policy


def _optimal_label(trace, budget_mode: str) -> str:
    """Belady MIN is the *true* optimum (ratio in ``[0, 1]``) for exact-key
    identity under a count budget. With a byte budget or semantic matching it is
    only a lower bound on the true optimum."""
    determinism = trace.determinism if isinstance(trace, MemoryTrace) else Determinism.EXACT_KEY
    if determinism is Determinism.EXACT_KEY and budget_mode == "count":
        return "optimal"
    return "lower_bound"


def score(trace, budget: int, policy, budget_mode: str = "count") -> ScoreReport:
    """Score one eviction ``policy`` on ``trace`` at a fixed ``budget``.

    ``policy`` may be a reference name (``"lru"``/``"lfu"``/``"fifo"``), a policy
    class, or an object implementing the policy hooks. The competitive ratio is
    ``policy_useful_hits / oracle_useful_hits`` and is labelled ``optimal`` only
    under exact-key identity with a count budget; otherwise it is a lower bound.
    """
    pol = _resolve_policy(policy)
    name = getattr(pol, "name", pol.__class__.__name__)
    ratio, pol_res, opt_res = competitive_ratio(trace, budget, pol, budget_mode)
    determinism = trace.determinism.value if isinstance(trace, MemoryTrace) else "exact_key"
    return ScoreReport(
        policy=name,
        budget=budget,
        budget_mode=budget_mode,
        determinism=determinism,
        competitive_ratio=ratio,
        policy_hits=pol_res["useful_hits"],
        oracle_hits=opt_res["useful_hits"],
        total_useful=opt_res["total_useful"],
        optimal_label=_optimal_label(trace, budget_mode),
        premature_forgets=pol_res["premature"],
    )


def score_many(trace, budget: int, policies, budget_mode: str = "count"):
    """Score several policies on the same trace; returns ``list[ScoreReport]``."""
    return [score(trace, budget, p, budget_mode) for p in policies]


__all__ = ["score", "score_many"]
