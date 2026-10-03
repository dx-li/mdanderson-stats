"""Source-equation checks for the private generalized odds-rate core."""

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats._log_odds_rate import log_odds_rate_components


def test_c_one_is_the_log_logistic_distribution() -> None:
    relative_log_times = np.log([0.4, 0.8, 1.6, 3.2])
    log_shape = np.log(1.7)
    centered_log_scale = np.log(1.2)
    log_density, log_survival, cdf = log_odds_rate_components(
        relative_log_times, log_shape, centered_log_scale, log_c=0.0
    )

    log_ratio = relative_log_times - centered_log_scale
    shape = np.exp(log_shape)
    log_product = shape * log_ratio
    expected_log_survival = -np.logaddexp(0.0, log_product)
    expected_cdf = -np.expm1(expected_log_survival)
    expected_log_density = (
        log_shape
        - centered_log_scale
        + (shape - 1.0) * log_ratio
        - 2.0 * np.logaddexp(0.0, log_product)
    )
    np.testing.assert_allclose(log_survival, expected_log_survival, rtol=0, atol=2e-15)
    np.testing.assert_allclose(cdf, expected_cdf, rtol=0, atol=2e-15)
    np.testing.assert_allclose(log_density, expected_log_density, rtol=0, atol=2e-15)


def test_interior_c_matches_the_primary_survival_and_derivative() -> None:
    relative_log_times = np.log([0.3, 0.9, 2.4])
    log_shape = np.log(1.45)
    centered_log_scale = np.log(1.2)
    log_c = np.log(0.65)
    log_density, log_survival, cdf = log_odds_rate_components(
        relative_log_times, log_shape, centered_log_scale, log_c
    )

    shape = np.exp(log_shape)
    scale = np.exp(centered_log_scale)
    c = np.exp(log_c)
    ratio = relative_log_times - centered_log_scale
    power = np.exp(shape * ratio)
    expected_survival = (1.0 + c * power) ** (-1.0 / c)
    expected_density = (
        shape / scale * np.exp((shape - 1.0) * ratio) * (1.0 + c * power) ** (-1.0 / c - 1.0)
    )
    np.testing.assert_allclose(np.exp(log_survival), expected_survival, rtol=2e-14)
    np.testing.assert_allclose(cdf, 1.0 - expected_survival, rtol=0, atol=2e-15)
    np.testing.assert_allclose(np.exp(log_density), expected_density, rtol=2e-14)


def test_components_match_independent_base_r_reference() -> None:
    reference_path = Path(__file__).parent / "fixtures" / "log-odds-rate-identities.csv"
    with reference_path.open(newline="", encoding="utf-8") as stream:
        reference = list(csv.DictReader(stream))
    for row in reference:
        log_density, log_survival, cdf = log_odds_rate_components(
            [np.log(float(row["time"]))],
            log_shape=np.log(1.8),
            centered_log_scale=np.log(1.1),
            log_c=np.log(0.65),
        )
        np.testing.assert_allclose(
            log_density[0], float(row["log_density_centered"]), rtol=2e-14, atol=2e-14
        )
        np.testing.assert_allclose(
            log_survival[0], float(row["log_survival"]), rtol=2e-14, atol=2e-14
        )
        np.testing.assert_allclose(cdf[0], float(row["cdf"]), rtol=0, atol=2e-15)


def test_small_c_converges_to_weibull_without_cancellation() -> None:
    relative_log_times = np.array([-2.0, -0.3, 0.4, 2.0])
    log_shape = np.log(1.35)
    centered_log_scale = -0.1
    log_density, log_survival, cdf = log_odds_rate_components(
        relative_log_times, log_shape, centered_log_scale, log_c=-800.0
    )

    shape = np.exp(log_shape)
    log_ratio = relative_log_times - centered_log_scale
    hazard = np.exp(shape * log_ratio)
    np.testing.assert_allclose(log_survival, -hazard, rtol=0, atol=2e-15)
    np.testing.assert_allclose(cdf, -np.expm1(-hazard), rtol=0, atol=2e-15)
    np.testing.assert_allclose(
        log_density,
        log_shape - centered_log_scale + (shape - 1.0) * log_ratio - hazard,
        rtol=0,
        atol=2e-15,
    )


def test_tail_limits_are_finite_or_valid_zero_probability() -> None:
    # Large positive log-product has zero representable survival, while its
    # density and CDF remain well-defined as -inf and one.
    log_density, log_survival, cdf = log_odds_rate_components(
        [1000.0], log_shape=0.0, centered_log_scale=0.0, log_c=-1000.0
    )
    assert np.isneginf(log_density[0])
    assert np.isneginf(log_survival[0])
    assert cdf[0] == 1.0

    # At tiny c and far below scale, CDF remains representable even though
    # exp(log-power) itself underflows.
    _, log_survival, cdf = log_odds_rate_components(
        [-1000.0], log_shape=0.0, centered_log_scale=0.0, log_c=-800.0
    )
    assert log_survival[0] == 0.0
    assert cdf[0] == 0.0


@pytest.mark.parametrize(
    ("times", "log_shape", "centered_log_scale", "log_c"),
    [([], 0.0, 0.0, 0.0), ([1.0, 2.0], np.inf, 0.0, 0.0), ([1.0], 0.0, 0.0, np.nan)],
)
def test_invalid_core_inputs_fail_explicitly(
    times: list[float], log_shape: float, centered_log_scale: float, log_c: float
) -> None:
    with pytest.raises(ValueError):
        log_odds_rate_components(times, log_shape, centered_log_scale, log_c)
