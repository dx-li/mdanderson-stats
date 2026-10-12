"""Native observation contracts and independent partial-data posterior checks."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit, log_expit
from scipy.stats import norm

from mdanderson_stats import fit_uaroet

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/uaroet-desktop-likelihood.json").read_text()
)


@pytest.mark.parametrize("case", REFERENCE["cases"])
def test_original_likelihood_observation_contract(case):
    value = 0.0
    used = 0
    for dose, efficacy, toxicity, entry, eff_delay, tox_delay in case["patients"]:
        if case["clock"] < entry + tox_delay or toxicity == -1:
            continue
        efficacy_known = efficacy >= 0 and case["clock"] >= entry + eff_delay
        if not efficacy_known and not case["partial"]:
            continue
        p = np.asarray(case["joint"][dose])
        value += np.log(p[efficacy, toxicity] if efficacy_known else p[:, toxicity].sum())
        used += 1
    assert case["result"]["log_likelihood"] == pytest.approx(value, abs=2e-14)
    assert case["result"]["likelihood"] == pytest.approx(np.exp(value), abs=2e-14)
    assert case["result"]["used"] == used


def test_partial_only_posterior_against_independent_scalar_integration():
    mean, sd = -0.5, 0.9

    def density(x):
        return norm.pdf(x, loc=mean, scale=sd) * np.exp(6 * log_expit(x) + 2 * log_expit(-x))

    normalizer = quad(density, -12, 12, epsabs=1e-12)[0]
    expected = quad(lambda x: expit(x) * density(x), -12, 12, epsabs=1e-12)[0] / normalizer
    fit = fit_uaroet(
        np.zeros((1, 2, 2)),
        toxicity_only_counts=[[2, 6]],
        prior_mean=[-0.2, mean],
        prior_sd=[0.8, sd],
        association=0.0,
        draws=1500,
        warmup=500,
        chains=4,
        rng=np.random.default_rng(8301),
    )
    assert not fit.direct_prior
    risk = fit.joint.sum(axis=3)[..., 0, 1]
    assert abs(risk.mean() - expected) < 0.015
    assert abs(fit.parameters[..., 0].mean() + 0.2) < 0.06
    expected_log = 2 * np.log1p(-risk) + 6 * np.log(risk)
    np.testing.assert_allclose(fit.log_likelihood, expected_log, atol=2e-12)


def test_complete_and_partial_counts_combine_without_double_counting():
    counts = np.array([[[2, 1], [3, 4]]])
    fit = fit_uaroet(
        counts,
        toxicity_only_counts=[[1, 2]],
        prior_mean=[0, 0],
        prior_sd=[1, 1],
        association=0.0,
        draws=16,
        warmup=8,
        chains=2,
        rng=np.random.default_rng(12),
    )
    joint = fit.joint
    expected = (np.log(joint) * counts).sum(axis=(-3, -2, -1)) + (
        np.log(joint.sum(axis=3)) * [1, 2]
    ).sum(axis=(-2, -1))
    np.testing.assert_allclose(fit.log_likelihood, expected, atol=2e-12)


@pytest.mark.parametrize(
    "partial", ([[1]], [[-0.1, 0]], [[1.5, 0]], [[True, False]], [[np.inf, 0]], [[10001, 0]])
)
def test_partial_validation_precedes_rng_use(partial):
    rng = np.random.default_rng(30)
    before = rng.bit_generator.state
    with pytest.raises(ValueError):
        fit_uaroet(
            np.zeros((1, 2, 2)),
            toxicity_only_counts=partial,
            prior_mean=[0, 0],
            prior_sd=[1, 1],
            rng=rng,
        )
    assert rng.bit_generator.state == before
