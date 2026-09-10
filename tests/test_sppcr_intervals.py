"""Interior source agreement, repaired support limits and workflow integration."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import sppcr_bootstrap, sppcr_bootstrap_intervals, sppcr_intervals

FIXTURE = json.loads((Path(__file__).parent / "fixtures/sppcr_intervals.json").read_text())


def intervals(mu, calibration_sd=0.1, frequency_sd=0.02, mutant_sd=0.02, **kwargs):
    return sppcr_intervals(
        mu,
        progenitor=(0, 0),
        calibration_sd=calibration_sd,
        frequency_transformed_sd=frequency_sd,
        mutant_transformed_sd=mutant_sd,
        **kwargs,
    )


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_report_expressions_or_disclosed_boundary_repair(case):
    p, c, s, cs = (case[k] for k in ["p", "calibration", "transformed_sd", "calibration_sd"])
    r = intervals([c * (1 - p), c * p], calibration_sd=cs, frequency_sd=s, mutant_sd=s)
    expected = case["values"]
    if case["interior"]:
        np.testing.assert_allclose(
            [
                r.mutant.lower,
                r.mutant.upper,
                r.calibration.lower,
                r.calibration.upper,
                r.inverse_calibration.lower,
                r.inverse_calibration.upper,
            ],
            expected,
            rtol=2e-13,
        )
    else:
        assert r.calibration.lower == 0
        assert r.inverse_calibration.upper == np.inf
        assert r.calibration.lower_clipped
        if p < 0.5:
            assert r.mutant.lower == 0 and r.mutant.lower_clipped
            assert expected[0] > p  # Source wraps a negative angle into a positive bound.
        else:
            assert r.mutant.upper == 1 and r.mutant.upper_clipped
            assert expected[1] < p


@pytest.mark.parametrize("p", [0, 1e-300, 0.1, 0.5, 0.9, 1])
@pytest.mark.parametrize("sd", [0, 1e-10, 0.1, 10, 1e308])
def test_frequency_limits_are_ordered_contain_estimate_and_stay_in_support(p, sd):
    r = intervals([1 - p, p], frequency_sd=sd, mutant_sd=sd)
    for f in [r.frequency, r.mutant]:
        assert np.all(f.available)
        assert np.all(f.lower >= 0)
        assert np.all(f.lower <= f.estimate)
        assert np.all(f.estimate <= f.upper)
        assert np.all(f.upper <= 1)
        if sd == 0:
            np.testing.assert_array_equal(f.lower, f.estimate)
            np.testing.assert_array_equal(f.upper, f.estimate)


def test_reciprocal_interval_reverses_endpoints_and_zero_is_unbounded():
    r = intervals([1, 2], calibration_sd=0.1)
    assert r.inverse_calibration.lower == pytest.approx(1 / r.calibration.upper)
    assert r.inverse_calibration.upper == pytest.approx(1 / r.calibration.lower)
    r = intervals([1, 1], calibration_sd=1, multiplier=2)
    assert r.calibration.lower == 0
    assert r.inverse_calibration.upper == np.inf
    assert r.inverse_calibration.available
    assert not r.calibration.lower_clipped  # Exactly zero, not outside the support.


def test_unavailable_sd_zero_means_and_independent_batch_availability():
    r = intervals(
        [[1, 2], [0, 0]],
        calibration_sd=[np.nan, 0],
        frequency_sd=[[np.nan, 0.1], [0.1, 0.1]],
        mutant_sd=0.1,
    )
    np.testing.assert_array_equal(r.calibration.available, [False, True])
    assert np.isnan(r.calibration.lower[0])
    assert r.calibration.lower[1] == r.calibration.upper[1] == 0
    assert not r.inverse_calibration.available.any()
    assert np.isnan(r.inverse_calibration.estimate[1])
    np.testing.assert_array_equal(r.frequency.available, [[False, True], [False, False]])
    assert np.isnan(r.frequency.lower[1]).all()


def test_bootstrap_intervals_use_observed_centers_and_retained_progenitor():
    b = sppcr_bootstrap(
        [1, 2],
        [[4, 8, 2], [10, 15, 5]],
        [20, 30],
        progenitor=(0, 1),
        rng=np.random.default_rng(91),
        replicates=30,
    )
    r = sppcr_bootstrap_intervals(b)
    assert b.summary.progenitor == (0, 1) == r.progenitor
    assert r.calibration.estimate == b.observed_fit.mu.sum()
    assert r.calibration.estimate != b.summary.calibration.mean
    assert r.calibration.upper == pytest.approx(
        b.observed_fit.mu.sum() + 1.959964 * b.summary.calibration.standard_deviation
    )
    assert r.mutant.estimate == pytest.approx(b.observed_fit.mu[2] / b.observed_fit.mu.sum())


def test_all_zero_bootstrap_has_calibration_but_no_frequency_intervals():
    b = sppcr_bootstrap(
        [1], [[0, 0]], 10, progenitor=(0, 1), rng=np.random.default_rng(2), replicates=3
    )
    r = sppcr_bootstrap_intervals(b)
    assert r.calibration.available and r.calibration.upper == 0
    assert not np.any(r.frequency.available)
    assert not r.inverse_calibration.available


def test_empty_batch_and_immutable_arrays():
    r = intervals(np.empty((0, 2)))
    assert r.frequency.lower.shape == (0, 2)
    mu = np.array([1.0, 2.0])
    r = intervals(mu)
    mu[:] = 0
    assert r.calibration.estimate == 3
    for x in [
        r.frequency.lower,
        r.frequency.estimate,
        r.calibration.available,
        r.mutant.lower_clipped,
    ]:
        with pytest.raises(ValueError):
            x.setflags(write=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(calibration_sd=-1),
        dict(frequency_sd=np.inf),
        dict(mutant_sd=-1),
        dict(multiplier=0),
        dict(multiplier=np.nan),
        dict(frequency_sd=[1, 2, 3]),
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        intervals([1, 2], **kwargs)


def test_unrepresentable_calibration_or_reciprocal_fails():
    with pytest.raises(ArithmeticError):
        intervals([1, 2], calibration_sd=1e308)
    with pytest.raises(ArithmeticError):
        intervals([1e-320], calibration_sd=0)
