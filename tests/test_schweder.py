import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.schweder import SchwederFitError, schweder_bootstrap, schweder_fit

REFERENCE = json.loads((Path(__file__).parent / "fixtures/multi.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["schweder"])
def test_original_desktop_and_splus_line_fits(case):
    desktop = case["reference"]["desktop"]
    splus = case["reference"]["splus"]
    if desktop["status"]:
        assert splus["status"] != 0
        with pytest.raises(SchwederFitError):
            schweder_fit(case["pvalues"], alpha=case["alpha"])
        return
    fitted = schweder_fit(case["pvalues"], alpha=case["alpha"])
    assert fitted.null_estimate == pytest.approx(desktop["null_estimate"], rel=2e-12)
    assert fitted.null_estimate == pytest.approx(splus["null_estimate"], rel=2e-12)
    assert fitted.prediction_variance == pytest.approx(
        splus["prediction_variance"], rel=2e-11, abs=1e-16
    )
    np.testing.assert_allclose(fitted.one_minus_p, desktop["one_minus_p"], atol=1e-15)
    np.testing.assert_array_equal(fitted.upper_counts, desktop["upper_counts"])


def test_exact_line_recovers_known_null_count():
    pvalues = [0.2, 0.4, 0.6, 0.8]
    fitted = schweder_fit(pvalues)
    assert fitted.null_estimate == pytest.approx(5)
    assert fitted.fitted_points == 4
    assert fitted.mean_squared_error < 1e-30
    # An unconstrained estimate may legitimately exceed the number of tests.
    np.testing.assert_allclose(fitted.one_minus_p * fitted.null_estimate, fitted.upper_counts)


def test_combines_ties_and_counts_observations_at_or_above_each_pvalue():
    values = [0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 0.8]
    result = schweder_fit(values)
    np.testing.assert_allclose(result.one_minus_p, [0.2, 0.4, 0.6, 0.8, 0.9])
    np.testing.assert_array_equal(result.upper_counts, [2, 3, 4, 5, 7])


def test_bootstrap_reproducibility_variance_and_original_plot():
    values = np.linspace(0.02, 0.98, 40)
    result = schweder_bootstrap(values, 50, rng=126)
    duplicate = schweder_bootstrap(values, 50, rng=126)
    np.testing.assert_array_equal(result.estimates, duplicate.estimates)
    assert result.mean == pytest.approx(np.mean(result.estimates))
    assert result.variance == pytest.approx(np.var(result.estimates, ddof=1))
    assert result.estimates.size == 50 and 50 <= result.attempts <= 100
    np.testing.assert_allclose(result.fit.one_minus_p, 1 - values[::-1])
    assert schweder_bootstrap(values, 1, rng=126).variance == 0


def test_bootstrap_draws_fitted_by_original_fortran():
    case = REFERENCE["bootstrap"]
    result = schweder_bootstrap(case["pvalues"], case["samples"], rng=case["seed"])
    np.testing.assert_allclose(result.estimates, case["estimates"], rtol=2e-12)
    assert result.attempts == case["attempts"]
    assert result.mean == pytest.approx(np.mean(case["estimates"]), rel=2e-12)
    assert result.variance == pytest.approx(np.var(case["estimates"], ddof=1), rel=2e-11)


def test_bootstrap_generator_state_is_explicit():
    rng = np.random.default_rng(11)
    values = np.linspace(0.05, 0.95, 20)
    first = schweder_bootstrap(values, 10, rng=rng)
    second = schweder_bootstrap(values, 10, rng=rng)
    assert not np.array_equal(first.estimates, second.estimates)


def test_bootstrap_failed_fits_are_bounded_and_do_not_become_zero_estimates():
    # Four unique values; most size-four bootstrap samples have fewer than four.
    with pytest.raises(SchwederFitError, match="successful fits"):
        schweder_bootstrap([0.2, 0.4, 0.6, 0.8], 50, rng=1)


@pytest.mark.parametrize("values", [[0.1] * 10, [0.1, 0.2, 0.3], [1e-100, 2e-100, 3e-100, 4e-100]])
def test_degenerate_data_rejected(values):
    with pytest.raises(SchwederFitError):
        schweder_fit(values)


@pytest.mark.parametrize("alpha", [0, 1, np.nan])
def test_invalid_fit_alpha(alpha):
    with pytest.raises(ValueError):
        schweder_fit([0.1, 0.2, 0.3, 0.4], alpha=alpha)


@pytest.mark.parametrize("samples", [0, -1, 1.5, True])
def test_invalid_bootstrap_samples(samples):
    with pytest.raises(ValueError):
        schweder_bootstrap([0.1, 0.2, 0.3, 0.4], samples=samples)
