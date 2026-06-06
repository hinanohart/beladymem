import json

import pytest

from beladymem.trace import Determinism, EventType, MemoryTrace, TraceEvent


def test_event_rejects_bad_size():
    with pytest.raises(ValueError):
        TraceEvent(t=0, op=EventType.USE, item_id="a", size=0)


def test_event_rejects_empty_id():
    with pytest.raises(ValueError):
        TraceEvent(t=0, op=EventType.USE, item_id="")


def test_event_coerces_str_op():
    e = TraceEvent(t=0, op="use", item_id="a")
    assert e.op is EventType.USE


def test_trace_rejects_non_monotonic_time():
    with pytest.raises(ValueError):
        MemoryTrace(
            events=[
                TraceEvent(t=5, op=EventType.USE, item_id="a"),
                TraceEvent(t=3, op=EventType.USE, item_id="b"),
            ]
        )


def test_n_useful_counts_only_useful_uses():
    tr = MemoryTrace(
        events=[
            TraceEvent(t=0, op=EventType.USE, item_id="a", useful=True),
            TraceEvent(t=1, op=EventType.USE, item_id="b", useful=False),
            TraceEvent(t=2, op=EventType.WRITE, item_id="c"),
        ]
    )
    assert tr.n_useful == 1


def test_jsonl_roundtrip(tmp_path):
    tr = MemoryTrace(
        events=[
            TraceEvent(t=0, op=EventType.USE, item_id="a"),
            TraceEvent(t=1, op=EventType.USE, item_id="b", useful=False),
            TraceEvent(t=2, op=EventType.WRITE, item_id="a"),
        ],
        determinism=Determinism.SEMANTIC,
    )
    p = tmp_path / "t.jsonl"
    tr.to_jsonl(p)
    back = MemoryTrace.from_jsonl(p)
    assert back.determinism is Determinism.SEMANTIC
    assert len(back) == 3
    assert [e.item_id for e in back.events] == ["a", "b", "a"]
    assert back.events[1].useful is False


def test_jsonl_is_utf8(tmp_path):
    tr = MemoryTrace(events=[TraceEvent(t=0, op=EventType.USE, item_id="memo-α")])
    p = tmp_path / "u.jsonl"
    tr.to_jsonl(p)
    # round-trips a non-ascii id through the utf-8 path
    back = MemoryTrace.from_jsonl(p)
    assert back.events[0].item_id == "memo-α"
    # and the file really is valid json lines
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0])["determinism"] == "exact_key"
