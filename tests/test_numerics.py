import json
import math
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.numerics import invert_monotone, normal_tails


def test_normal_reference_routines():
    cases = json.loads((Path(__file__).parent / "fixtures/numerics.json").read_text())["normal"]
    for case in cases:
        actual = normal_tails(case["x"], log=True)
        np.testing.assert_allclose(
            actual, [case["log_cdf"], case["log_sf"]], rtol=2e-14, atol=1e-14
        )


def test_normal_against_independent_erfc():
    x = np.linspace(-30, 30, 301).reshape(7, 43)
    cdf, sf = normal_tails(x)
    expected = np.array([math.erfc(-v / math.sqrt(2)) / 2 for v in x.flat]).reshape(x.shape)
    np.testing.assert_allclose(cdf, expected, rtol=3e-13, atol=0)
    np.testing.assert_allclose(cdf + sf, 1, rtol=0, atol=2e-16)
    logcdf, logsf = normal_tails(x, log=True)
    np.testing.assert_allclose(np.exp(logcdf), cdf, rtol=3e-13, atol=0)
    np.testing.assert_allclose(np.exp(logsf), sf, rtol=3e-13, atol=0)


@pytest.mark.parametrize("x", [30, 100, 1e4, 1e7, 67861400])
def test_extreme_normal_log_tail_satisfies_mills_bounds(x):
    logdensity = -x * x / 2 - math.log(2 * math.pi) / 2
    lower = logdensity + math.log(x) - math.log1p(x * x)
    upper = logdensity - math.log(x)
    logtail = float(normal_tails(x, log=True)[1])
    tolerance = 8 * abs(np.spacing(upper))
    assert lower - tolerance <= logtail <= upper + tolerance
    assert np.isfinite(logtail)
    assert logtail == normal_tails(-x, log=True)[0]


def test_normal_endpoints_and_invalid_input():
    np.testing.assert_array_equal(normal_tails([-np.inf, 0, np.inf]), [[0, 0.5, 1], [1, 0.5, 0]])
    assert normal_tails(-np.inf, log=True)[0] == -np.inf
    with pytest.raises(ValueError, match="NaN"):
        normal_tails([0, np.nan])


def test_inversion_reference_original():
    cases = json.loads((Path(__file__).parent / "fixtures/numerics.json").read_text())["inverse"]
    functions = {1: math.sqrt, 2: math.exp, 3: lambda x: 7 - 3 * x, 4: lambda x: x**3}
    for case in cases:
        actual = invert_monotone(
            functions[case["kind"]],
            case["initial"],
            case["target"],
            lower=case["lower"],
            upper=case["upper"],
        )
        assert actual == pytest.approx(case["root"], rel=2e-6, abs=2e-8)
        assert functions[case["kind"]](actual) == pytest.approx(case["target"], rel=5e-6, abs=1e-7)


def test_inversion_named_argument_and_no_mutation():
    arguments = {"slope": -3, "intercept": 7}

    def function(slope, intercept, probability):
        return slope * probability + intercept

    root = invert_monotone(
        function, 0.5, 5.5, lower=0, upper=1, parameter="probability", function_kwargs=arguments
    )
    assert root == pytest.approx(0.5)
    assert arguments == {"slope": -3, "intercept": 7}


def test_inversion_nested_reentrant_calls():
    def outer(x):
        return invert_monotone(lambda y: y**3, 0, x, lower=-10, upper=10)

    result = invert_monotone(outer, 0, 2, lower=-100, upper=100)
    assert result == pytest.approx(8, rel=1e-5)


@pytest.mark.parametrize("initial", [-10, 0, 10])
def test_inversion_negative_roots_and_endpoints(initial):
    assert invert_monotone(lambda x: x, initial, -3, lower=-10, upper=10) == pytest.approx(-3)
    assert invert_monotone(lambda x: -x, initial, 10, lower=-10, upper=10) == -10
    assert invert_monotone(lambda x: x, initial, 10, lower=-10, upper=10) == 10


def test_inversion_direct_bracket_and_tight_tolerance():
    root = invert_monotone(
        lambda x: x**3,
        0,
        2,
        lower=0,
        upper=10,
        absolute_step=10,
        absolute_tolerance=1e-14,
        relative_tolerance=1e-14,
    )
    assert root == pytest.approx(2 ** (1 / 3), abs=2e-14)


def test_inversion_steps_below_float_spacing():
    root = invert_monotone(
        lambda x: x,
        1e10,
        1e10 + 10,
        lower=1e10 - 100,
        upper=1e10 + 100,
        absolute_step=1e-20,
        relative_step=0,
        relative_tolerance=1e-15,
    )
    assert root == pytest.approx(1e10 + 10, abs=1e-5, rel=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"absolute_step": 0, "relative_step": 0},
        {"relative_step": -1},
        {"step_multiplier": 1},
        {"absolute_tolerance": 0},
        {"relative_tolerance": 0},
        {"max_iterations": 0},
        {"max_iterations": 1.5},
        {"parameter": ""},
        {"lower": 1},
        {"upper": -1},
        {"target": math.inf},
    ],
)
def test_inversion_invalid_parameters(kwargs):
    args = {"initial": 0, "target": 1, "lower": -10, "upper": 10} | kwargs
    with pytest.raises(ValueError):
        invert_monotone(lambda x: x, **args)


def test_inversion_rejects_bad_evaluations_and_unbracketed_target():
    with pytest.raises(ValueError, match="bracket"):
        invert_monotone(lambda x: x, 0, 20, lower=-10, upper=10)
    with pytest.raises(ValueError, match="finite"):
        invert_monotone(lambda x: math.nan, 0, 1, lower=-10, upper=10)
    with pytest.raises(ValueError, match="scalar"):
        invert_monotone(lambda x: [x], 0, 1, lower=-10, upper=10)
    with pytest.raises(ValueError, match="monotonicity"):
        invert_monotone(lambda x: 100 if x == 0 else x, 0, 1, lower=-10, upper=10)


def test_inversion_nonconvergence_raises():
    with pytest.raises(RuntimeError, match="Bracket expansion"):
        invert_monotone(lambda x: x, 0, 99, lower=-100, upper=100, max_iterations=1)
    with pytest.raises(RuntimeError, match="converge"):
        invert_monotone(
            lambda x: x**3, 0, 2, lower=-100, upper=100, absolute_step=200, max_iterations=1
        )
