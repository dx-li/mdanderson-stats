import json
import math
from pathlib import Path

import pytest

from mdanderson_stats.misclib_maximum import fun_max, rc_fun_max, set_fun_max


def test_compiled_native_maxima_and_analytic_locations():
    functions = (
        lambda x: -((x - 0.37) ** 2),
        lambda x: -((x + 2.3) ** 2),
        lambda x: math.exp(-x),
        lambda x: -((x - 0.3) ** 4),
        lambda x: 3 * math.log(x) + 7 * math.log1p(-x),
        math.sin,
    )
    expected = (0.37, -2.3, 0, 0.3, 0.3, math.pi / 2)
    cases = json.loads((Path(__file__).parent / "fixtures/misclib-maximum-native.json").read_text())
    for c, function, answer in zip(cases, functions, expected, strict=True):
        result = fun_max(
            function,
            local=set_fun_max(c["low"], c["high"], abs_tol=1e-10, rel_tol=1e-10),
        )
        assert result.status == 0
        assert result.x == pytest.approx(c["x"], abs=2e-8, rel=0)
        assert result.x == pytest.approx(answer, abs=2e-8, rel=0)
        assert c["low"] <= result.lower <= result.x <= result.upper <= c["high"]
        assert result.value == function(result.x)
        assert result.evaluations < 100


def test_reverse_communication_is_independent_and_budget_is_explicit():
    states = [set_fun_max(-5, 5), set_fun_max(-5, 5)]
    results = [rc_fun_max(state) for state in states]
    for _ in range(100):
        for i in range(2):
            if results[i].status == 1:
                results[i] = rc_fun_max(states[i], -((results[i].x - i) ** 2))
        if all(r.status == 0 for r in results):
            break
    assert results[0].x == pytest.approx(0, abs=1e-5, rel=0)
    assert results[1].x == pytest.approx(1, abs=1e-5, rel=0)
    with pytest.raises(ValueError, match="finished"):
        rc_fun_max(states[0], 0)
    state = set_fun_max(0, 1, max_evaluations=1)
    with pytest.raises(ArithmeticError, match="exhausted"):
        fun_max(lambda x: x, local=state)
    assert state.result is not None and state.result.status == -2
    state = set_fun_max(0, 1)
    rc_fun_max(state)
    with pytest.raises(ValueError):
        rc_fun_max(state, float("nan"))
    assert rc_fun_max(state, 0).status == 1


def test_extreme_scales_remain_finite_and_inside_bounds():
    for scale in (1e-200, 1.0, 1e200, 1e300):
        seen = []

        def function(x):
            assert 0 <= x <= scale
            seen.append(x)
            return -1e300 * (x / scale - 0.3) ** 2

        result = fun_max(
            function,
            local=set_fun_max(0, scale, abs_tol=scale * 1e-12, rel_tol=1e-10),
        )
        assert result.x / scale == pytest.approx(0.3, abs=2e-8, rel=0)
        assert math.isfinite(result.value)
        assert len(seen) == result.evaluations
    small = math.ulp(0.0)
    result = fun_max(lambda x: x, local=set_fun_max(0, small, abs_tol=small, rel_tol=0))
    assert result.status == 0
