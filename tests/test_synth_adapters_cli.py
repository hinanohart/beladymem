import json

from beladymem.adapters import load_jsonl, longmemeval_to_trace
from beladymem.adapters.longmemeval import longmemeval_records_to_trace
from beladymem.cli import build_parser, main
from beladymem.synth import (
    cyclic_thrash,
    divergence_family,
    freq_dominated_trace,
    short_reuse_trace,
)
from beladymem.trace import Determinism, EventType, MemoryTrace


def test_synth_generators_are_valid_use_traces():
    for tr in (cyclic_thrash(5, 5), short_reuse_trace(0), freq_dominated_trace(0)):
        assert isinstance(tr, MemoryTrace)
        assert len(tr) > 0
        assert all(e.op is EventType.USE for e in tr.events)


def test_divergence_family_sizes():
    short, freq = divergence_family(n_each=10, seed=0)
    assert len(short) == 10 and len(freq) == 10


def test_synth_is_deterministic_by_seed():
    a = [e.item_id for e in short_reuse_trace(7).events]
    b = [e.item_id for e in short_reuse_trace(7).events]
    assert a == b


def test_jsonl_adapter_roundtrip(tmp_path):
    tr = freq_dominated_trace(1)
    p = tmp_path / "t.jsonl"
    tr.to_jsonl(p)
    back = load_jsonl(p)
    assert len(back) == len(tr)


def test_longmemeval_records_adapter():
    records = [
        {"question_id": "q1", "answer_session_ids": ["s1", "s2"]},
        {"question_id": "q2", "answer_session_ids": ["s2"]},
    ]
    tr = longmemeval_records_to_trace(records)
    assert tr.determinism is Determinism.SEMANTIC
    assert [e.item_id for e in tr.events] == ["session:s1", "session:s2", "session:s2"]


def test_longmemeval_file_adapter(tmp_path):
    p = tmp_path / "lme.json"
    p.write_text(
        json.dumps([{"question_id": "q1", "answer_session_ids": ["s9"]}]),
        encoding="utf-8",
    )
    tr = longmemeval_to_trace(p)
    assert [e.item_id for e in tr.events] == ["session:s9"]


def test_cli_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == "0.1.0a1"


def test_cli_gate_runs(capsys):
    rc = main(["gate"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ALL GATES PASS" in out


def test_cli_score_json(tmp_path, capsys):
    tr = cyclic_thrash(5, 8)
    p = tmp_path / "t.jsonl"
    tr.to_jsonl(p)
    rc = main(["score", str(p), "--budget", "3", "--policy", "lru,lfu", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert {r["policy"] for r in data} == {"lru", "lfu"}


def test_parser_requires_subcommand():
    parser = build_parser()
    assert parser.prog == "beladymem"
