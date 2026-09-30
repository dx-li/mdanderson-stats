import math
from decimal import Decimal, localcontext
from typing import cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.u2oet import U2OETProbabilities
from mdanderson_stats.u2oet_gao2010 import (
    U2OETGAO2010Marginal,
    u2oet_gao2010_probabilities,
)


def _marginal(alpha: float, beta1: float, beta2: float, lam: float, gamma: float):
    return U2OETGAO2010Marginal([[alpha, alpha]], [[beta1, beta2]], lam, gamma)


def test_centered_2010_link_and_likelihood_match_direct_equations() -> None:
    d1, d2 = np.array([1.0, 3.0, 8.0]), np.array([2.0, 5.0])
    efficacy = _marginal(-0.2, 0.3, -0.1, 0.8, 0.25)
    toxicity = _marginal(-0.7, -0.12, 0.18, 1.3, -0.05)
    result = u2oet_gao2010_probabilities(
        d1, d2, efficacy=efficacy, toxicity=toxicity, association=0.0
    )

    x1, x2 = d1 - d1.mean(), d2 - d2.mean()
    eta1 = -0.2 + 0.3 * x1[:, None]
    eta2 = -0.2 - 0.1 * x2[None, :]
    bracket = np.exp(eta1) + np.exp(eta2) + 0.25 * np.exp(eta1 + eta2)
    continuation = 1.0 - (1.0 + 0.8 * bracket) ** (-1.0 / 0.8)
    assert_allclose(np.exp(result.log_efficacy[..., 1]), continuation, atol=2e-15, rtol=2e-15)
    assert_allclose(
        result.log_joint,
        result.log_efficacy[..., :, None] + result.log_toxicity[..., None, :],
        atol=2e-15,
    )
    counts = np.ones_like(result.log_joint)
    toxicity_only = np.ones_like(result.log_toxicity)
    expected = float(np.sum(result.log_joint) + np.sum(result.log_toxicity))
    assert math.isclose(
        result.loglikelihood(counts, toxicity_only=toxicity_only), expected, abs_tol=1e-13
    )
    assert isinstance(result, U2OETProbabilities)


def test_negative_interaction_near_domain_boundary_uses_stable_remainder() -> None:
    epsilon = 2.0**-40
    marginal = _marginal(0.0, 0.0, 0.0, 1.0, -2.0 + epsilon)
    result = u2oet_gao2010_probabilities(
        [1.0, 2.0], [1.0, 2.0], efficacy=marginal, toxicity=marginal
    )
    remainder = math.fsum([1.0, 1.0, marginal.gamma])
    expected = remainder / (1.0 + remainder)
    assert expected > 0.0
    assert_allclose(np.exp(result.log_efficacy[..., 1]), expected, rtol=1e-12, atol=0.0)
    assert_allclose(np.exp(result.log_joint).sum(axis=(-2, -1)), 1.0, atol=2e-11)

    invalid = _marginal(0.0, 0.0, 0.0, 1.0, -2.0)
    with pytest.raises(ValueError, match="negative gamma"):
        u2oet_gao2010_probabilities([1.0, 2.0], [1.0, 2.0], efficacy=invalid, toxicity=invalid)

    first, second = 0.1, 0.2
    gamma_boundary = -(math.exp(-first) + math.exp(-second))
    gamma = gamma_boundary + 2.0**-40
    nonzero_eta = U2OETGAO2010Marginal([[first, second]], [[0.0, 0.0]], 1.0, gamma)
    near = u2oet_gao2010_probabilities(
        [0.0, 1.0], [0.0, 1.0], efficacy=nonzero_eta, toxicity=nonzero_eta
    )
    with localcontext() as context:
        context.prec = 80
        e1, e2 = Decimal.from_float(first), Decimal.from_float(second)
        bracket = e1.exp() + e2.exp() + Decimal.from_float(gamma) * (e1 + e2).exp()
        expected_nonzero = float(bracket / (1 + bracket))
    assert_allclose(np.exp(near.log_efficacy[..., 1]), expected_nonzero, rtol=1e-12, atol=0.0)


def test_centering_is_translation_and_dose_unit_invariant() -> None:
    d1, d2 = np.array([1.0, 2.5, 7.0]), np.array([3.0, 9.0])
    original_e = _marginal(0.3, 0.2, -0.1, 0.7, 0.15)
    original_t = _marginal(-0.4, -0.1, 0.08, 1.2, 0.0)
    original = u2oet_gao2010_probabilities(
        d1, d2, efficacy=original_e, toxicity=original_t, association=-0.4
    )
    translated = u2oet_gao2010_probabilities(
        d1 + 100.0,
        d2 + 20.0,
        efficacy=original_e,
        toxicity=original_t,
        association=-0.4,
    )
    scaled = u2oet_gao2010_probabilities(
        d1 * 100.0,
        d2 * 100.0,
        efficacy=_marginal(0.3, 0.002, -0.001, 0.7, 0.15),
        toxicity=_marginal(-0.4, -0.001, 0.0008, 1.2, 0.0),
        association=-0.4,
    )
    assert_allclose(translated.log_joint, original.log_joint, atol=2e-14, rtol=2e-14)
    assert_allclose(scaled.log_joint, original.log_joint, atol=2e-14, rtol=2e-14)


def test_centering_large_positive_doses_does_not_overflow_the_grid_mean() -> None:
    d1 = np.array([1.0, 1e308, 1.5e308])
    d2 = np.array([2.0, 1e308])
    efficacy = _marginal(0.0, 1e-308, -1e-308, 1.0, 0.1)
    result = u2oet_gao2010_probabilities(
        d1, d2, efficacy=efficacy, toxicity=efficacy, association=0.0
    )
    assert np.all(np.isfinite(result.log_joint))
    assert_allclose(np.exp(result.log_joint).sum(axis=(-2, -1)), 1.0, atol=2e-11)


def test_negative_gamma_cancellation_preserves_small_positive_term_at_large_eta() -> None:
    marginal = U2OETGAO2010Marginal([[1000.0, 0.0]], [[0.0, 0.0]], 1.0, -1.0)
    result = u2oet_gao2010_probabilities(
        [0.0, 1.0], [0.0, 1.0], efficacy=marginal, toxicity=marginal
    )
    # exp(1000) + 1 - exp(1000) == 1, despite the small positive term being
    # far below 80 decimal digits relative to either large term.
    assert_allclose(np.exp(result.log_efficacy[..., 1]), 0.5, atol=2e-14, rtol=2e-14)

    invalid = U2OETGAO2010Marginal([[1e7, 1e7]], [[0.0, 0.0]], 1.0, -1.0)
    with pytest.raises(ValueError, match="negative gamma"):
        u2oet_gao2010_probabilities([0.0, 1.0], [0.0, 1.0], efficacy=invalid, toxicity=invalid)


def test_zero_dose_grid_is_accepted_and_matches_its_translated_grid() -> None:
    marginal = _marginal(-0.3, 0.4, -0.2, 0.9, 0.1)
    zero = u2oet_gao2010_probabilities(
        [0.0, 1.0, 2.0], [0.0, 2.0], efficacy=marginal, toxicity=marginal
    )
    translated = u2oet_gao2010_probabilities(
        [10.0, 11.0, 12.0], [20.0, 22.0], efficacy=marginal, toxicity=marginal
    )
    assert_allclose(zero.log_joint, translated.log_joint, atol=2e-14, rtol=2e-14)


def test_threshold_shared_gamma_and_input_domains_are_checked() -> None:
    multi = U2OETGAO2010Marginal([[0.0, 0.0], [-0.3, 0.1]], [[0.4, -0.2], [0.1, 0.3]], 0.9, -0.2)
    result = u2oet_gao2010_probabilities(
        [1.0, 2.0], [1.0, 4.0], efficacy=multi, toxicity=multi, association=0.55
    )
    assert result.log_joint.shape == (2, 2, 3, 3)
    assert_allclose(np.exp(result.log_joint).sum(axis=(-2, -1)), 1.0, atol=2e-11)
    with pytest.raises(ValueError, match="same threshold-by-agent shape"):
        U2OETGAO2010Marginal([[0.0, 0.0]], [[0.1, 0.2], [0.2, 0.3]], 1.0, 0.0)
    with pytest.raises(ValueError, match="scalar"):
        U2OETGAO2010Marginal([[0.0, 0.0]], [[0.1, 0.2]], 1.0, cast(float, [0.0, 0.1]))
