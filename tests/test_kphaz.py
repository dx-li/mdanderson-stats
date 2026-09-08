"""Source hazard/variance comparisons and independent grouped-risk formulas."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import kphaz

CASES = json.loads((Path(__file__).parent / "fixtures/kphaz.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_archived_s_numerics(case):
    r = kphaz(
        case["time"],
        case["status"],
        strata=case["strata"],
        q=case["q"],
        method=case["method"],
        legacy=True,
    )
    rows = np.column_stack([r.time, r.hazard, r.variance, r.stratum_index + 1])
    expected = np.array(case["rows"], dtype=float).reshape(-1, 4)
    assert_allclose(rows, expected, rtol=2e-13, atol=2e-14)


def test_nelson_all_events_has_reciprocal_remaining_risk_hazards():
    r = kphaz(range(1, 6), [1] * 5)
    assert_array_equal(r.time, [1.5, 2.5, 3.5, 4.5])
    assert_allclose(r.hazard, 1 / np.arange(4, 0, -1))
    assert_allclose(r.variance, 1 / np.arange(4, 0, -1) ** 2)


def test_product_limit_equals_log_survival_ratio_and_greenwood_increment():
    r = kphaz([1, 2, 3, 4], [1, 1, 1, 0], method="product-limit", q=2)
    # Survival at first failure is 3/4, at third it is 1/4.
    assert_allclose(r.hazard, [np.log(3) / 2])
    assert_allclose(r.variance, [(1 / (3 * 2) + 1 / (2 * 1)) / 4])


def test_tied_censoring_remains_in_common_risk_set_and_permutation_is_irrelevant():
    t = np.array([1, 2, 2, 2, 3, 4])
    d = np.array([1, 1, 1, 0, 1, 0])
    r = kphaz(t, d)
    assert_allclose(r.hazard, [2 / 5, 1 / 2])
    assert_allclose(r.variance, [2 / 25, 1 / 4])
    p = np.array([0, 3, 2, 1, 4, 5])
    other = kphaz(t[p], d[p])
    assert_array_equal(other.hazard, r.hazard)
    product = kphaz(t, d, method="product-limit")
    assert_allclose(product.hazard, [-np.log(3 / 5), np.log(2)])
    assert_allclose(product.variance, [2 / (5 * 3), 1 / 2])


def test_terminal_censor_source_nan_is_not_used_by_default():
    r = kphaz([1, 2, 3, 3], [1, 1, 1, 0], method="product-limit")
    assert np.all(np.isfinite(r.variance))
    legacy = kphaz([1, 2, 3, 3], [1, 1, 1, 0], method="product-limit", legacy=True)
    assert np.isnan(legacy.variance[-1])


def test_terminal_failure_infinite_estimate_survives_window_sum():
    r = kphaz([1, 2, 3, 4], [1] * 4, method="product-limit", q=2)
    assert np.isfinite(r.hazard[0])
    assert r.hazard[-1] == r.variance[-1] == np.inf
    assert not np.any(np.isnan(r.variance))


def test_stratum_labels_empty_strata_and_time_scaling():
    r = kphaz([1, 2, 3, 1, 2, 3], [1, 1, 1, 0, 0, 0], strata=["A"] * 3 + ["B"] * 3)
    assert r.strata == ("A", "B")
    assert_array_equal(r.stratum_index, [0, 0])
    scaled = kphaz([2, 4, 6], [1, 1, 1])
    assert_allclose(scaled.time, 2 * r.time)
    assert_allclose(scaled.hazard, r.hazard / 2)
    assert_allclose(scaled.variance, r.variance / 4)
    with pytest.raises(ValueError):
        r.hazard[0] = 3
    empty = kphaz([1, 2], [1, 1], q=3)
    assert empty.time.shape == (0,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time": []},
        {"time": [-1, 2]},
        {"time": [np.nan, 2]},
        {"status": [1]},
        {"status": [1, 2]},
        {"status": [0, 0]},
        {"q": 0},
        {"q": 1.5},
        {"q": True},
        {"method": "wrong"},
        {"legacy": 1},
        {"strata": [1]},
        {"strata": [1, None]},
    ],
)
def test_invalid_inputs(kwargs):
    args = dict(time=[1, 2], status=[1, 1])
    args.update(kwargs)
    with pytest.raises(ValueError):
        kphaz(**args)
