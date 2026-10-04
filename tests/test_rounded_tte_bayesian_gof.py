from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.special import logsumexp, ndtr

from mdanderson_stats.rounded_tte_bayesian_gof import rounded_tte_bayesian_gof


def _quadrature_reference(
    family: str,
    lower: np.ndarray,
    upper: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    order: int = 24,
) -> np.ndarray:
    """Independent prior-coordinate Gauss-Hermite posterior-mean calculation."""
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    chol = np.linalg.cholesky(covariance)
    offsets = np.asarray(list(itertools.product(nodes, repeat=mean.size))) * np.sqrt(2.0)
    standard_weights = np.asarray(
        [np.prod(w) for w in itertools.product(weights, repeat=mean.size)]
    ) / np.pi ** (mean.size / 2)
    states = mean + offsets @ chol.T
    log_likelihood = np.empty(states.shape[0])
    for row, state in enumerate(states):
        if family == "exponential":
            scale = np.exp(state[0])
            interval_cdf = np.exp(-lower / scale) - np.exp(-upper / scale)
        elif family == "lognormal":
            sigma = np.exp(state[1])
            z_lower = np.log(lower) - state[0]
            z_upper = np.log(upper) - state[0]
            interval_cdf = ndtr(z_upper / sigma) - ndtr(z_lower / sigma)
        elif family == "log_odds_rate":
            shape, scale, c = np.exp(state)

            def cdf_at(time: np.ndarray) -> np.ndarray:
                log_product = np.log(c) + shape * (np.log(time) - np.log(scale))
                return -np.expm1(-np.logaddexp(0.0, log_product) / c)

            interval_cdf = cdf_at(upper) - cdf_at(lower)
        else:
            raise AssertionError(family)
        with np.errstate(divide="ignore"):
            log_likelihood[row] = np.log(interval_cdf).sum()
    log_weight = np.log(standard_weights) + log_likelihood
    normalized = np.exp(log_weight - logsumexp(log_weight))
    return normalized @ states


@pytest.mark.parametrize(
    ("family", "mean", "covariance", "bounds"),
    [
        (
            "exponential",
            np.array([0.3]),
            np.array([[0.36]]),
            (np.array([0.1, 0.7, 1.8, 3.2]), np.array([0.6, 1.4, 2.6, 4.3])),
        ),
        (
            "lognormal",
            np.array([0.15, -0.35]),
            np.array([[0.28, 0.06], [0.06, 0.2]]),
            (np.array([0.15, 0.6, 1.2, 2.4]), np.array([0.5, 1.1, 2.0, 3.8])),
        ),
        (
            "log_odds_rate",
            np.log([1.15, 1.7, 0.7]),
            np.array([[0.18, 0.035, -0.01], [0.035, 0.2, 0.02], [-0.01, 0.02, 0.16]]),
            (np.array([0.2, 0.8, 1.6, 2.7]), np.array([0.65, 1.35, 2.35, 4.0])),
        ),
    ],
)
def test_representative_posteriors_match_independent_quadrature(
    family: str, mean: np.ndarray, covariance: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]
) -> None:
    lower, upper = bounds
    fit = rounded_tte_bayesian_gof(
        lower,
        upper,
        family=family,
        prior_mean=mean,
        prior_covariance=covariance,
        draws=1600,
        warmup=500,
        chains=2,
        rng=np.random.default_rng(7214),
        compute_diagnostic=False,
    )
    coarse_reference = _quadrature_reference(family, lower, upper, mean, covariance, order=24)
    reference = _quadrature_reference(family, lower, upper, mean, covariance, order=40)
    centered_coordinate = 1 if family not in ("exponential", "lognormal") else 0
    reference[centered_coordinate] -= fit.time_offset
    coarse_reference[centered_coordinate] -= fit.time_offset
    mcse = fit.parameter_summary.batch_mean_mcse
    quadrature_refinement = np.abs(reference - coarse_reference)
    normalized_error = np.abs(fit.parameter_summary.mean - reference) / np.sqrt(
        mcse**2 + quadrature_refinement**2
    )
    assert np.all(normalized_error < 5)
    assert np.all(np.isfinite(fit.parameter_summary.split_rhat))
    assert fit.parameters.shape == (2, 1600, mean.size)
    assert fit.diagnostic is None


@pytest.mark.parametrize(
    ("family", "mean"),
    [
        ("exponential", [0.0]),
        ("weibull", [np.log(1.2), np.log(1.5)]),
        ("lognormal", [0.1, np.log(0.8)]),
        ("gamma", [np.log(1.4), np.log(1.5)]),
        ("inverse_gamma", [np.log(1.4), np.log(1.5)]),
        ("log_logistic", [np.log(1.4), np.log(1.5)]),
        ("log_odds_rate", [np.log(1.2), np.log(1.5), np.log(0.6)]),
    ],
)
def test_all_family_adapters_return_finite_paired_draws(family: str, mean: list[float]) -> None:
    dimension = len(mean)
    fit = rounded_tte_bayesian_gof(
        [0.4, 0.9, 1.5, 2.4],
        [0.8, 1.3, 2.0, 3.1],
        family=family,
        prior_mean=mean,
        prior_covariance=np.eye(dimension) * 0.2,
        draws=12,
        warmup=8,
        chains=2,
        rng=np.random.default_rng(912),
        compute_diagnostic=False,
    )
    assert fit.parameters.shape == (2, 12, dimension)
    assert np.all(np.isfinite(fit.parameters))
    assert np.all(np.isfinite(fit.log_likelihood))
    assert not fit.interval_lower.flags.writeable


def test_scale_change_preserves_exponential_fit_and_invalid_input_does_not_draw() -> None:
    lower = np.array([0.0, 0.8, 1.7, 3.0])
    upper = np.array([0.75, 1.4, 2.5, 4.5])
    covariance = np.array([[0.4]])
    first_rng = np.random.default_rng(88)
    first = rounded_tte_bayesian_gof(
        lower,
        upper,
        family="exponential",
        prior_mean=[0.2],
        prior_covariance=covariance,
        draws=48,
        warmup=24,
        chains=2,
        rng=first_rng,
    )
    offset = np.log(12.0)
    second = rounded_tte_bayesian_gof(
        lower * 12,
        upper * 12,
        family="exponential",
        prior_mean=[0.2 + offset],
        prior_covariance=covariance,
        draws=48,
        warmup=24,
        chains=2,
        rng=np.random.default_rng(88),
    )
    np.testing.assert_allclose(second.parameters, first.parameters, atol=2e-12, rtol=0)
    np.testing.assert_allclose(second.log_likelihood, first.log_likelihood, atol=2e-12, rtol=0)
    assert first.diagnostic is not None and second.diagnostic is not None

    invalid_rng = np.random.default_rng(37)
    control_rng = np.random.default_rng(37)
    with pytest.raises(ValueError, match="lower < upper"):
        rounded_tte_bayesian_gof(
            [0.5, 1.2],
            [0.5, 1.8],
            family="exponential",
            prior_mean=[0.0],
            prior_covariance=[[1.0]],
            rng=invalid_rng,
        )
    assert invalid_rng.random() == control_rng.random()

    over_budget_rng = np.random.default_rng(45)
    over_budget_control = np.random.default_rng(45)
    too_many = np.ones(7813)
    with pytest.raises(ValueError, match="draw-by-observation"):
        rounded_tte_bayesian_gof(
            np.zeros_like(too_many),
            too_many,
            family="exponential",
            prior_mean=[0.0],
            prior_covariance=[[1.0]],
            draws=8,
            chains=16,
            rng=over_budget_rng,
        )
    assert over_budget_rng.random() == over_budget_control.random()
