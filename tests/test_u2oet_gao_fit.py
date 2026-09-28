from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from mdanderson_stats.u2oet_gao import U2OETGAOMarginal, u2oet_gao_probabilities
from mdanderson_stats.u2oet_gao_fit import (
    fit_u2oet_gao,
    u2oet_gao_parameter_names,
)


def _fixed_coordinates() -> np.ndarray:
    # Each threshold has two agent-specific intercepts and slopes, followed
    # by the endpoint log-lambda; shared log-kappa and Fisher-z finish.
    return np.array([-1.0, -0.2, 0.4, 0.7, 0.0, -0.2, 0.3, 0.2, 0.3, 0.0, np.log(1.2), 0.0])


def test_fixed_gao_prior_retains_exact_probabilities_and_chain_axes() -> None:
    coordinates = _fixed_coordinates()
    names = u2oet_gao_parameter_names(2, 2)
    assert names == (
        "efficacy.intercept.1.agent1",
        "efficacy.intercept.1.agent2",
        "efficacy.slope.1.agent1",
        "efficacy.slope.1.agent2",
        "efficacy.log_lambda",
        "toxicity.intercept.1.agent1",
        "toxicity.intercept.1.agent2",
        "toxicity.slope.1.agent1",
        "toxicity.slope.1.agent2",
        "toxicity.log_lambda",
        "shared.log_kappa",
        "association.fisher_z",
    )
    d1, d2 = np.array([1.0, 2.0]), np.array([1.0, 2.0])
    counts = np.zeros((2, 2, 2, 2))
    expected = u2oet_gao_probabilities(
        d1,
        d2,
        efficacy=U2OETGAOMarginal(coordinates[:2][None, :], coordinates[2:4][None, :], 1.0),
        toxicity=U2OETGAOMarginal(coordinates[5:7][None, :], coordinates[7:9][None, :], 1.0),
        kappa=1.2,
        association=0.0,
    ).joint
    fit = fit_u2oet_gao(
        d1,
        d2,
        counts,
        prior_mean=coordinates,
        prior_sd=np.zeros(12),
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(9),
    )
    assert fit.parameters.shape == (2, 8, 12)
    assert fit.joint.shape == (2, 8, 2, 2, 2, 2)
    np.testing.assert_array_equal(fit.parameters, np.broadcast_to(coordinates, (2, 8, 12)))
    np.testing.assert_allclose(fit.joint, np.broadcast_to(expected, fit.joint.shape), rtol=1e-13)
    assert fit.likelihood_evaluations == 2
    assert not fit.parameters.flags.writeable


def test_one_free_coordinate_matches_independent_normal_quadrature() -> None:
    d1 = d2 = np.array([1.0, 2.0])
    counts = np.zeros((2, 2, 2, 2))
    counts[0, 0, 1, 0] = 3
    counts[0, 0, 0, 0] = 1
    toxicity_only = np.zeros((2, 2, 2))
    toxicity_only[1, 1, 1] = 2
    center = _fixed_coordinates()
    sd = np.zeros(center.size)
    sd[0] = 0.7

    def probability_and_weight(z: float) -> tuple[float, float]:
        point = center.copy()
        point[0] += sd[0] * z
        result = u2oet_gao_probabilities(
            d1,
            d2,
            efficacy=U2OETGAOMarginal(point[:2][None, :], point[2:4][None, :], 1.0),
            toxicity=U2OETGAOMarginal(point[5:7][None, :], point[7:9][None, :], 1.0),
            kappa=1.2,
            association=0.0,
        )
        loglik = float(np.sum(counts[0, 0] * result.log_joint[0, 0]))
        loglik += float(np.sum(toxicity_only[1, 1] * result.log_toxicity[1, 1]))
        return float(result.joint[0, 0, 1, 0]), loglik

    def weighted(z: float, moment: bool) -> float:
        probability, loglik = probability_and_weight(z)
        return (probability if moment else 1.0) * np.exp(loglik - 0.5 * z * z)

    normalizer = quad(lambda z: weighted(z, False), -10, 10, epsabs=1e-11)[0]
    reference = quad(lambda z: weighted(z, True), -10, 10, epsabs=1e-11)[0] / normalizer
    fit = fit_u2oet_gao(
        d1,
        d2,
        counts,
        prior_mean=center,
        prior_sd=sd,
        toxicity_only=toxicity_only,
        draws=1200,
        warmup=300,
        chains=2,
        rng=np.random.default_rng(771),
    )
    estimate = float(fit.joint[:, :, 0, 0, 1, 0].mean())
    assert estimate == pytest.approx(reference, abs=0.035)
    assert np.isfinite(fit.parameter_summary.split_rhat[0])


def test_work_preflight_does_not_consume_rng_and_fixed_start_must_match() -> None:
    coordinates = _fixed_coordinates()
    counts = np.zeros((2, 2, 2, 2))
    rng1 = np.random.default_rng(15)
    rng2 = np.random.default_rng(15)
    with pytest.raises(ValueError, match="minimum GAO likelihood work"):
        fit_u2oet_gao(
            [1, 2],
            [1, 2],
            counts,
            prior_mean=coordinates,
            prior_sd=np.zeros(12),
            draws=8,
            warmup=0,
            chains=2,
            rng=rng1,
            max_work=1,
        )
    assert rng1.random() == rng2.random()

    start = coordinates.copy()
    start[0] += 0.1
    with pytest.raises(ValueError, match="zero prior SD"):
        fit_u2oet_gao(
            [1, 2],
            [1, 2],
            counts,
            prior_mean=coordinates,
            prior_sd=np.zeros(12),
            initial=start,
            draws=8,
            warmup=0,
            chains=2,
            rng=np.random.default_rng(2),
        )


def test_fisher_z_saturation_is_rejected_instead_of_using_singular_limit() -> None:
    coordinates = _fixed_coordinates()
    coordinates[-1] = 40.0
    with pytest.raises(ArithmeticError, match="interior correlation range"):
        fit_u2oet_gao(
            [1, 2],
            [1, 2],
            np.zeros((2, 2, 2, 2)),
            prior_mean=coordinates,
            prior_sd=np.zeros(coordinates.size),
            draws=8,
            warmup=0,
            chains=2,
            rng=np.random.default_rng(11),
        )
