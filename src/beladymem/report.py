"""Score report dataclass and the machine-enforced NON-CLAIMs.

The three NON-CLAIMs below are reproduced verbatim in the README and grepped in
CI. They fence what the competitive ratio does and does not say.
"""

from __future__ import annotations

from dataclasses import dataclass, field

NON_CLAIMS = (
    "The competitive ratio does NOT predict downstream task accuracy or F1.",
    "Admission/write decisions are NOT scored; the oracle is eviction-only.",
    "Under semantic (fuzzy) matching the oracle is a LOWER BOUND, not the optimum.",
)


@dataclass
class ScoreReport:
    policy: str
    budget: int
    budget_mode: str
    determinism: str
    competitive_ratio: float
    policy_hits: int
    oracle_hits: int
    total_useful: int
    optimal_label: str  # "optimal" (exact_key+count) or "lower_bound"
    premature_forgets: list = field(default_factory=list)
    non_claims: tuple = NON_CLAIMS

    def summary(self) -> str:
        cr = (
            "nan"
            if self.competitive_ratio != self.competitive_ratio
            else f"{self.competitive_ratio:.4f}"
        )
        return (
            f"policy={self.policy} budget={self.budget}({self.budget_mode}) "
            f"competitive_ratio={cr} "
            f"hits={self.policy_hits}/{self.oracle_hits} "
            f"[{self.optimal_label}] premature_forgets={len(self.premature_forgets)}"
        )

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "budget": self.budget,
            "budget_mode": self.budget_mode,
            "determinism": self.determinism,
            "competitive_ratio": self.competitive_ratio,
            "policy_hits": self.policy_hits,
            "oracle_hits": self.oracle_hits,
            "total_useful": self.total_useful,
            "optimal_label": self.optimal_label,
            "premature_forgets": self.premature_forgets,
            "non_claims": list(self.non_claims),
        }
