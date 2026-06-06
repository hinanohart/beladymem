import pytest

from beladymem.gates import ALL_GATES, run_all_gates


@pytest.mark.parametrize("key,title,fn", ALL_GATES, ids=[g[0] for g in ALL_GATES])
def test_each_sensitivity_gate_passes(key, title, fn):
    ok, detail = fn()
    assert ok, f"{key} ({title}) failed: {detail}"


def test_run_all_gates_returns_true():
    assert run_all_gates(verbose=False) is True
