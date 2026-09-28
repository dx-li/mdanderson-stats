import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.u2oet import U2OETProbabilities
from mdanderson_stats.u2oet_gao import U2OETGAOMarginal, u2oet_gao_probabilities


def _marginal(intercepts: np.ndarray, slopes: np.ndarray, lam: float) -> U2OETGAOMarginal:
    return U2OETGAOMarginal(intercepts, slopes, lam)


def test_gao_lambda_one_is_logistic_of_interaction_sum_and_rho_zero_is_independent() -> None:
    doses1 = np.array([2.0, 5.0])
    doses2 = np.array([10.0, 20.0])
    efficacy = _marginal(np.array([[-1.0, 0.2]]), np.array([[0.1, 0.03]]), 1.0)
    toxicity = _marginal(np.array([[0.5, -1.2]]), np.array([[-0.02, 0.04]]), 1.0)
    kappa = 0.7
    result = u2oet_gao_probabilities(
        doses1, doses2, efficacy=efficacy, toxicity=toxicity, kappa=kappa
    )
    eta1 = efficacy.intercepts[0, 0] + efficacy.slopes[0, 0] * doses1[:, None]
    eta2 = efficacy.intercepts[0, 1] + efficacy.slopes[0, 1] * doses2[None, :]
    total = np.exp(eta1) + np.exp(eta2) + kappa * np.exp(eta1 + eta2)
    continuation = total / (1.0 + total)
    assert_allclose(np.exp(result.log_efficacy[..., 1]), continuation, atol=2e-15, rtol=2e-15)
    assert_allclose(np.exp(result.log_efficacy[..., 0]), 1.0 - continuation, atol=2e-15)
    assert isinstance(result, U2OETProbabilities)
    assert_allclose(
        result.log_joint,
        result.log_efficacy[..., :, None] + result.log_toxicity[..., None, :],
        atol=2e-15,
    )


def test_gaussian_copula_preserves_gao_marginals_at_positive_and_negative_dependence() -> None:
    doses1, doses2 = [1.0, 2.0], [3.0, 6.0]
    efficacy = _marginal(
        np.array([[-1.2, -0.7], [0.5, 0.1]]),
        np.array([[0.12, -0.03], [-0.08, 0.09]]),
        0.8,
    )
    toxicity = _marginal(np.array([[0.3, -1.0]]), np.array([[0.02, 0.06]]), 1.4)
    independent = u2oet_gao_probabilities(
        doses1, doses2, efficacy=efficacy, toxicity=toxicity, kappa=0.4
    )
    for rho in (-0.65, 0.55, -1.0, 1.0):
        result = u2oet_gao_probabilities(
            doses1,
            doses2,
            efficacy=efficacy,
            toxicity=toxicity,
            kappa=0.4,
            association=rho,
        )
        assert_allclose(
            np.exp(result.log_joint).sum(axis=-1), np.exp(result.log_efficacy), atol=2e-11
        )
        assert_allclose(
            np.exp(result.log_joint).sum(axis=-2), np.exp(result.log_toxicity), atol=2e-11
        )
        assert_allclose(np.exp(result.log_joint).sum(axis=(-2, -1)), 1.0, atol=2e-11)
        if abs(rho) < 1:
            assert np.isfinite(result.loglikelihood(np.ones_like(result.log_joint)))
    assert not np.array_equal(independent.log_joint, result.log_joint)


def test_small_lambda_has_complementary_loglog_limit_and_validation_is_explicit() -> None:
    d1, d2 = [1.0, 2.0], [1.0, 2.0]
    marginal = _marginal(np.array([[0.0, 0.0]]), np.array([[0.1, 0.2]]), 1e-8)
    result = u2oet_gao_probabilities(d1, d2, efficacy=marginal, toxicity=marginal, kappa=0.5)
    eta1 = 0.1 * np.asarray(d1)[:, None]
    eta2 = 0.2 * np.asarray(d2)[None, :]
    total = np.exp(eta1) + np.exp(eta2) + 0.5 * np.exp(eta1 + eta2)
    assert_allclose(np.exp(result.log_efficacy[..., 0]), np.exp(-total), atol=2e-8, rtol=2e-8)
    with pytest.raises(ValueError, match="positive"):
        _marginal(np.zeros((1, 2)), np.zeros((1, 2)), 0.0)
    with pytest.raises(ValueError, match="positive scalar"):
        u2oet_gao_probabilities(d1, d2, efficacy=marginal, toxicity=marginal, kappa=0.0)
