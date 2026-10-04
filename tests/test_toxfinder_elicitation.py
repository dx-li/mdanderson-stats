import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammainc, gammaincc, gammaln

from mdanderson_stats.toxfinder_elicitation import (
    elicit_toxfinder_agent_prior,
    elicit_toxfinder_prior,
)


def _integrate_probability(alpha_shape, alpha_scale, beta_shape, beta_scale, ratio, p, upper):
    def integrand(beta):
        if beta <= 0:
            return 0.0
        log_density = (
            (beta_shape - 1) * np.log(beta)
            - beta / beta_scale
            - gammaln(beta_shape)
            - beta_shape * np.log(beta_scale)
        )
        log_argument = np.log(p / (1 - p)) - beta * np.log(ratio) - np.log(alpha_scale)
        if upper and log_argument < -745:
            conditional = 1.0
        elif not upper and log_argument > 709:
            conditional = 1.0
        elif log_argument < -745:
            conditional = 0.0
        else:
            argument = np.exp(min(log_argument, 709))
            conditional = (
                gammaincc(alpha_shape, argument) if upper else gammainc(alpha_shape, argument)
            )
        return np.exp(log_density) * conditional

    return quad(integrand, 0, np.inf, epsabs=1e-10, epsrel=1e-10)


@pytest.mark.parametrize(
    "doses",
    [
        [600, 1200, 1400, 2000],
        [350, 600, 700, 800],
    ],
)
def test_paper_table_one_elicitation(doses):
    fitted = elicit_toxfinder_agent_prior(doses, 0.30)
    assert fitted.alpha_mean == pytest.approx(0.30 / 0.70, abs=1e-15)
    assert fitted.alpha_variance > 0
    assert fitted.beta_mean > 0
    assert fitted.beta_variance > 0
    assert fitted.probability_residuals == pytest.approx([0, 0], abs=2e-5)
    assert fitted.moment_residual == pytest.approx(0, abs=1e-12)
    assert fitted.converged


@pytest.mark.parametrize("doses", [[600, 1200, 1400, 2000], [350, 600, 700, 800]])
def test_probability_constraints_against_independent_adaptive_quadrature(doses):
    fit = elicit_toxfinder_agent_prior(doses, 0.30)
    ratios = np.asarray(doses, dtype=float) / doses[1]
    low, low_error = _integrate_probability(
        fit.alpha_shape, fit.alpha_scale, fit.beta_shape, fit.beta_scale, ratios[0], 0.05, False
    )
    high, high_error = _integrate_probability(
        fit.alpha_shape, fit.alpha_scale, fit.beta_shape, fit.beta_scale, ratios[3], 0.30, True
    )
    assert low == pytest.approx(0.99, abs=2e-6)
    assert high == pytest.approx(0.99, abs=2e-6)
    assert low_error < 1e-8
    assert high_error < 1e-8
    assert np.max(fit.probability_quadrature_errors) < 5e-7


@pytest.mark.parametrize(
    ("doses", "moments", "probabilities"),
    [
        ([600, 1200, 1400, 2000], [0.4286, 0.1054, 7.6494, 5.7145], [0.986614, 0.992176]),
        ([350, 600, 700, 800], [0.4286, 0.0791, 7.8019, 3.9933], [0.977774, 0.978817]),
    ],
)
def test_paper_rounded_moments_do_not_exactly_match_probability_constraints(
    doses, moments, probabilities
):
    alpha_mean, alpha_variance, beta_mean, beta_variance = moments
    alpha_shape = alpha_mean**2 / alpha_variance
    alpha_scale = alpha_variance / alpha_mean
    beta_shape = beta_mean**2 / beta_variance
    beta_scale = beta_variance / beta_mean
    ratios = np.asarray(doses, dtype=float) / doses[1]
    low, _ = _integrate_probability(
        alpha_shape, alpha_scale, beta_shape, beta_scale, ratios[0], 0.05, False
    )
    high, _ = _integrate_probability(
        alpha_shape, alpha_scale, beta_shape, beta_scale, ratios[3], 0.30, True
    )
    assert [low, high] == pytest.approx(probabilities, abs=1e-6)


def test_two_agent_builder_requires_and_uses_interaction_moments():
    result = elicit_toxfinder_prior(
        [600, 1200, 1400, 2000],
        [350, 600, 700, 800],
        0.30,
        interaction_mean=[1, 0.05],
        interaction_variance=[3, 3],
    )
    assert result.prior.mean == pytest.approx(
        [
            result.agent1.alpha_mean,
            result.agent1.beta_mean,
            result.agent2.alpha_mean,
            result.agent2.beta_mean,
            1,
            0.05,
        ]
    )
    assert result.prior.variance[-2:] == pytest.approx([3, 3])
    assert result.prior.shape.shape == (6,)


@pytest.mark.parametrize("doses", [[1, 1, 2, 3], [1, 2, 2, 3], [1, 2, 3, 2], [1, 2, 3]])
def test_invalid_dose_order_or_count_is_rejected(doses):
    with pytest.raises(ValueError):
        elicit_toxfinder_agent_prior(doses, 0.30)


def test_high_toxicity_threshold_must_exceed_target():
    with pytest.raises(ValueError, match="probabilities must satisfy"):
        elicit_toxfinder_agent_prior([1, 2, 3, 4], 0.50, high_toxicity_probability=0.40)
