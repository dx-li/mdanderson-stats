"""Prior information checks independent of native RNG and posterior MCMC."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm

from mdanderson_stats.uaroet_desktop_inference import uaroet_marginal_ess

ESS = json.loads((Path(__file__).parent / "fixtures/uaroet-desktop-ess.json").read_text())


@pytest.mark.parametrize("case", ESS["cases"])
def test_native_ordinal_beta_moment_matching(case):
    actual = uaroet_marginal_ess(case["efficacy"], case["toxicity"])
    for endpoint in ("efficacy", "toxicity"):
        np.testing.assert_allclose(
            getattr(actual, endpoint), case["result"][endpoint], rtol=3e-14, atol=3e-13
        )
    np.testing.assert_allclose(
        [actual.efficacy_mean, actual.toxicity_mean, actual.mean, actual.minimum, actual.maximum],
        case["result"]["summary"],
        rtol=3e-14,
        atol=3e-13,
    )


def test_scalar_logistic_normal_information_against_independent_quadrature():
    # Baseline-only marginal isolates the independent normal-prior transformation.
    rng = np.random.default_rng(391)
    values = expit(rng.normal(-0.3, 0.9, 100_000))
    draws = np.stack((1 - values, values), axis=-1)[:, None, :]
    actual = uaroet_marginal_ess(draws, draws)
    mean = quad(lambda z: expit(z) * norm.pdf(z, loc=-0.3, scale=0.9), -12, 12, epsabs=1e-12)[0]
    second = quad(
        lambda z: expit(z) ** 2 * norm.pdf(z, loc=-0.3, scale=0.9), -12, 12, epsabs=1e-12
    )[0]
    expected = mean * (1 - mean) / (second - mean**2) - 1
    assert actual.mean == pytest.approx(expected, rel=0.012)


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((8, 2, 2)),
        np.ones((8, 2, 2)) * 0.5,
        np.ones((8, 2, 2)),
        np.zeros((1, 2, 2)),
        np.zeros((8, 2, 5)),
        np.array([[[np.nan, 1]]] * 8),
    ],
)
def test_unidentified_or_invalid_beta_moments_rejected(value):
    with pytest.raises(ValueError):
        uaroet_marginal_ess(value, value)


def test_negative_empirical_ess_is_reported_not_clipped():
    values = np.array([[[0, 1]], [[1, 0]]])
    actual = uaroet_marginal_ess(values, values)
    assert actual.mean == -0.5


def test_prior_sampling_ignores_observations_and_matches_logistic_normal_information():
    from dataclasses import replace

    from mdanderson_stats.uaroet_desktop_inference import uaroet_desktop_prior_ess
    from mdanderson_stats.uaroet_desktop_study import read_uaroet_desktop_study

    study = read_uaroet_desktop_study(
        (Path(__file__).parent / "fixtures/uaroet-python-study.json").read_text()
    )
    a = uaroet_desktop_prior_ess(study, draws=100_000, rng=np.random.default_rng(1002))
    # Independent quadrature of the two baseline logistic-normal marginals.
    for endpoint, mu, sd in [("efficacy", 0, 0.7), ("toxicity", -1, 0.7)]:
        mean = quad(lambda z: expit(z) * norm.pdf(z, mu, sd), -12, 12, epsabs=1e-12)[0]
        second = quad(lambda z: expit(z) ** 2 * norm.pdf(z, mu, sd), -12, 12, epsabs=1e-12)[0]
        reference = mean * (1 - mean) / (second - mean**2) - 1
        assert getattr(a, endpoint)[0, 1] == pytest.approx(reference, rel=0.02)
    b = uaroet_desktop_prior_ess(study, draws=64, rng=np.random.default_rng(8))
    c = uaroet_desktop_prior_ess(
        replace(study, patients=()), draws=64, rng=np.random.default_rng(8)
    )
    np.testing.assert_array_equal(b.efficacy, c.efficacy)
    np.testing.assert_array_equal(b.toxicity, c.toxicity)
