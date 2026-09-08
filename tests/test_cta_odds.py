"""Native RELRISK comparisons and independent log-Wald interval checks."""

import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import odds_ratio

CASES = json.loads((Path(__file__).parent / "fixtures/cta_odds.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_native_orientations_response_and_alpha(case):
    fit = odds_ratio(
        case["observed"],
        risk_factor=case["risk_factor"],
        response_index=case["response_index"],
        alpha=case["alpha"],
        legacy=True,
    )
    assert_allclose(
        [fit.odds_ratio, fit.standard_error, fit.lower, fit.upper], case["result"], rtol=2e-6
    )


def test_independent_wald_limits_and_source_cdf_bug():
    fit = odds_ratio([[80, 10], [20, 90]])
    se = math.sqrt(1 / 80 + 1 / 10 + 1 / 20 + 1 / 90)
    z = NormalDist().inv_cdf(0.975)
    assert_allclose(fit.odds, [4, 1 / 9])
    assert_allclose(fit.odds_ratio, 36)
    assert_allclose(fit.standard_error, se)
    assert_allclose(
        [fit.lower, fit.upper], [36 * math.exp(-z * se), 36 * math.exp(z * se)], rtol=1e-14
    )
    source = odds_ratio([[80, 10], [20, 90]], legacy=True)
    assert_allclose(source.multiplier, (1 + math.erf(0.025 / math.sqrt(2))) / 2, rtol=1e-14)
    assert source.lower > fit.lower and source.upper < fit.upper


def test_batch_scaling_orientation_and_reversal():
    a = np.array([[3.0, 7], [13, 17]])
    fit = odds_ratio(np.stack([a, a * 100]).reshape(1, 2, 2, 2))
    assert fit.odds.shape == (1, 2, 2)
    assert_allclose(fit.odds_ratio[0], [51 / 91] * 2)
    assert_allclose(fit.standard_error[0, 0], 10 * fit.standard_error[0, 1])
    transposed = odds_ratio(a.T, risk_factor="rows")
    assert_allclose(transposed.odds, fit.odds[0, 0])
    reverse = odds_ratio(a, response_index=1)
    assert_allclose(reverse.odds_ratio, 1 / fit.odds_ratio[0, 0])
    assert_allclose([reverse.lower, reverse.upper], [1 / fit.upper[0, 0], 1 / fit.lower[0, 0]])
    swap_groups = odds_ratio(a[:, ::-1])
    assert_allclose(swap_groups.odds_ratio, reverse.odds_ratio)
    for value in (
        fit.odds,
        fit.odds_ratio,
        fit.log_odds_ratio,
        fit.standard_error,
        fit.lower,
        fit.upper,
        fit.log_lower,
        fit.log_upper,
    ):
        assert not value.flags.writeable
    assert_array_equal(a, [[3, 7], [13, 17]])


def test_extreme_counts_and_alpha_keep_finite_log_results():
    fit = odds_ratio([[1e300, 1e-300], [1e-300, 1e300]])
    assert np.isinf(fit.odds_ratio)
    assert_allclose(fit.log_odds_ratio, 1200 * math.log(10))
    assert_allclose(fit.standard_error, math.sqrt(2) * 1e150)
    assert fit.lower == 0 and np.isinf(fit.upper)
    assert np.isfinite(fit.log_lower) and np.isfinite(fit.log_upper)
    smallest = np.nextafter(0.0, 1.0)
    tiny = odds_ratio(np.full((2, 2), smallest), alpha=smallest)
    assert_allclose(tiny.standard_error, 2 / math.sqrt(smallest))
    assert np.isfinite(tiny.multiplier) and tiny.multiplier > 38
    huge = odds_ratio(np.full((2, 2), 1e308))
    assert huge.odds_ratio == 1
    assert_allclose(huge.standard_error, 2e-154, atol=0)


@pytest.mark.parametrize(
    "table",
    [
        [[0, 1], [2, 3]],
        [[-1, 2], [3, 4]],
        [[1, np.nan], [2, 3]],
        [[1, np.inf], [2, 3]],
        [[1]],
        [[1, 2, 3], [4, 5, 6]],
    ],
)
def test_invalid_tables(table):
    with pytest.raises(ValueError):
        odds_ratio(table)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"risk_factor": "bad"},
        {"response_index": 2},
        {"response_index": True},
        {"response_index": 0.5},
        {"legacy": 1},
        {"alpha": 0},
        {"alpha": 1},
        {"alpha": np.nan},
        {"alpha": [0.05]},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        odds_ratio([[1, 2], [3, 4]], **kwargs)
