# beladymem

Score an agent-memory **eviction / forgetting policy** by its *competitive
ratio* against the **Belady MIN** offline-optimal oracle. CPU-only, dependency
light (numpy), synthetic-validated.

> **Status: `0.1.0a1` (alpha).** Validated on synthetic ground truth only — see
> [Validation](#validation) and [What beladymem does not claim](#what-beladymem-does-not-claim).

beladymem is a **measurement instrument**, not a memory system. It takes a trace
of *useful retrievals* from a long-term memory store (mem0 / Letta / a custom
agent memory), treats it as a classical cache reference stream at a fixed budget
`B`, and reports — for each eviction policy — one falsifiable number: how close
that policy comes to the clairvoyant optimum.

```
competitive_ratio(policy) = useful_hits(policy) / useful_hits(Belady-MIN)   ∈ [0, 1]
```

## The model (why the ratio is in `[0, 1]`)

The sequence of useful retrievals is a cache **reference stream**; the memory
budget `B` is a cache of size `B`; replay is classical **demand paging** (a
forgotten-then-requested item is a miss). The **Belady MIN** oracle evicts the
resident item whose next useful retrieval is farthest in the future. By Belady's
1966 theorem this is the offline optimum for demand paging, so the competitive
ratio of any online policy lies in `[0, 1]` for exact-key identity under a count
budget. The optimality is verified two ways in the test-suite: against an
independent brute-force Belady, and against an exhaustive true optimum on small
traces.

`WRITE` events are recorded for adapter fidelity but the canonical oracle is
defined on the useful-retrieval references. Under **semantic** (fuzzy) matching
or a **byte** budget, Belady MIN is a documented *lower bound*, not the optimum.

## Why a competitive ratio and not regret

Recent work scores memory forgetting policies by **regret against the best fixed
policy in hindsight** (e.g. *Forgetful but Faithful*, arXiv:2512.12856) or learns
a Belady-style policy for the context window (*Neural Paging*, arXiv:2603.02228),
or benchmarks downstream staleness (*Memora*, arXiv:2604.20006). beladymem
measures a different thing: the ratio to the **unconstrained clairvoyant
optimum**, not the gap to the best *fixed* policy.

These are not the same yardstick, and they do not always agree. Gate **G9**
constructs a trace family on which the Belady competitive ratio ranks `LRU`
above `LFU` while best-fixed-policy regret ranks `LFU` above `LRU`, with
non-overlapping bootstrap confidence intervals — a reproducible demonstration
that the competitive ratio carries information that regret does not. Run it
yourself with `beladymem gate`.

## Install

```bash
pip install -e .          # numpy only
pip install -e ".[dev]"   # + pytest, ruff
```

## Quickstart

```python
from beladymem import score
from beladymem.synth import freq_dominated_trace

trace = freq_dominated_trace(seed=0)
report = score(trace, budget=5, policy="lru")
print(report.summary())
# policy=lru budget=5(count) competitive_ratio=0.73xx hits=.../... [optimal] premature_forgets=...
```

Bring your own policy by implementing three hooks (`reset`, `on_use`,
`evict_victim`) and pass the instance to `score`. Compare several at once with
`beladymem.score.score_many`.

### CLI

```bash
beladymem gate                                   # run sensitivity gates G1-G9
beladymem score trace.jsonl --budget 512 --policy lru,lfu,fifo
beladymem score trace.jsonl --budget 512 --json
```

### Trace format

A trace is JSON Lines: an optional header `{"determinism": "exact_key"}` then one
event per line, e.g. `{"t": 0, "op": "use", "item_id": "fact:42", "useful": true}`.
Adapters are provided for the native JSONL format and for LongMemEval-style
oracle annotations (`beladymem.adapters`).

## Validation

beladymem ships **pre-registered sensitivity gates** that run on synthetic
ground truth and prove the instrument discriminates *before* it is pointed at any
real data. `beladymem gate` runs all nine and exits non-zero on any failure:

| gate | checks |
|------|--------|
| G1 | Belady scores 1.0 against itself; no online policy exceeds 1.0; LRU < 1.0 on the cyclic pathology |
| G2 | LRU collapses to ~0 on a cyclic stream below budget (textbook thrash) |
| G3 | Belady hits are non-decreasing in the budget |
| G4 | a corrupted predictor (robustness) scores below a perfect one (consistency) |
| G5 | a perfectly-predicting oracle reproduces Belady exactly (consistency == 1.0) |
| G6 | semantic / byte modes are labelled `lower_bound`, exact-key count mode `optimal` |
| G7 | the bootstrap CI of the mean is well calibrated |
| G8 | a zero-useful-hit trace yields `NaN` (not a crash) and is excluded from aggregates |
| G9 | competitive ratio and best-fixed-policy regret rank LRU vs LFU in opposite order, CIs disjoint |

Thresholds are fixed in `gates.py` and are not relaxed to force a pass.

## What beladymem does **not** claim

These three NON-CLAIMs are enforced verbatim in the test-suite and CI:

- *The competitive ratio does NOT predict downstream task accuracy or F1.*
- *Admission/write decisions are NOT scored; the oracle is eviction-only.*
- *Under semantic (fuzzy) matching the oracle is a LOWER BOUND, not the optimum.*

Further scope notes: results in this repository are from **synthetic** traces
with planted ground truth. The LongMemEval adapter operates at
**session/turn granularity** (LongMemEval's gold is session/turn level, not
item level), so it scores sessions-as-items, not the fact-level contents of a
real store. beladymem is an offline diagnostic — Belady MIN is non-causal and
cannot be used as a runtime policy.

## License

MIT © 2026 hinanohart
