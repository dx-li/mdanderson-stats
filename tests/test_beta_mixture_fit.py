import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import digamma

from mdanderson_stats.beta_mixture import BetaMixture
from mdanderson_stats.beta_mixture_fit import (
    BetaMixtureFitError,
    beta_mixture_start,
    fit_beta_mixture_em,
)

REFERENCE = json.loads((Path(__file__).parent / "fixtures/beta_mixture.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["starts"])
def test_original_stbeta_initialization(case):
    previous = BetaMixture(**case["previous"])
    if case["reference"]["status"]:
        with pytest.raises(BetaMixtureFitError):
            beta_mixture_start(case["pvalues"], previous)
        return
    result = beta_mixture_start(case["pvalues"], previous)
    reference = case["reference"]["model"]
    assert result.null_weight == pytest.approx(reference["null_weight"], rel=2e-13)
    for key in ("weights", "a", "b"):
        np.testing.assert_allclose(getattr(result, key), reference[key], rtol=2e-13)


@pytest.mark.parametrize("case", REFERENCE["em_fits"])
def test_original_embeta_final_parameters_and_likelihood(case):
    initial = BetaMixture(**case["initial"])
    result = fit_beta_mixture_em(
        case["pvalues"],
        initial,
        tolerance=case["tolerance"],
        legacy_endpoints=case.get("legacy_endpoints", False),
    )
    reference = case["reference"]
    assert reference["status"] == 0
    assert result.model.null_weight == pytest.approx(reference["model"]["null_weight"], rel=2e-8)
    for key in ("weights", "a", "b"):
        np.testing.assert_allclose(getattr(result.model, key), reference["model"][key], rtol=2e-8)
    assert result.log_likelihood == pytest.approx(reference["log_likelihood"], abs=5e-8)
    assert result.cramer_von_mises == pytest.approx(reference["cramer_von_mises"], rel=2e-8)
    assert result.iterations == len(result.log_likelihood_history) - 1
    assert np.min(np.diff(result.log_likelihood_history)) > -5e-8


def test_single_beta_mle_satisfies_score_equations_and_improves_likelihood():
    x = np.random.default_rng(232).beta(2, 5, size=500)
    initial = BetaMixture(0, [1], [1], [1])
    fit = fit_beta_mixture_em(x, initial)
    a, b = fit.model.a[0], fit.model.b[0]
    np.testing.assert_allclose(
        [digamma(a) - digamma(a + b), digamma(b) - digamma(a + b)],
        [np.log(x).mean(), np.log1p(-x).mean()],
        atol=1e-12,
    )
    assert fit.log_likelihood >= BetaMixture(0, [1], [2], [5]).log_likelihood(x)
    assert fit.model.null_weight == 0


def test_uniform_fit_and_default_initialization():
    fit = fit_beta_mixture_em([0, 0.5, 1], BetaMixture(1))
    assert fit.iterations == 0 and fit.log_likelihood == 0
    values = REFERENCE["em_fits"][0]["pvalues"]
    default = fit_beta_mixture_em(values)
    explicit = fit_beta_mixture_em(values, beta_mixture_start(values))
    assert default.log_likelihood == explicit.log_likelihood


def test_start_uses_sample_variance_and_strict_cdf_cutoff():
    x = [0.01, 0.02, 0.03, 0.05, 0.9]
    result = beta_mixture_start(x)
    mean, variance = 0.02, 0.0001
    concentration = mean * (1 - mean) / variance - 1
    assert result.weights[0] == 0.6
    assert result.a[0] == pytest.approx(mean * concentration)
    assert result.b[0] == pytest.approx((1 - mean) * concentration)
    with pytest.raises(BetaMixtureFitError):
        beta_mixture_start([0.01] * 5)


def test_failed_convergence_and_empty_components_are_explicit():
    values = REFERENCE["em_fits"][0]["pvalues"]
    with pytest.raises(BetaMixtureFitError, match="without convergence"):
        fit_beta_mixture_em(values, max_iterations=1)
    with pytest.raises(BetaMixtureFitError, match="effective weight"):
        fit_beta_mixture_em(values, BetaMixture(1, [0], [2], [3]))
    with pytest.raises(BetaMixtureFitError, match="n>3"):
        fit_beta_mixture_em([0.1, 0.2, 0.3, 0.4], BetaMixture(0.5, [0.5], [1], [2]))
    with pytest.raises(ValueError, match="interior"):
        fit_beta_mixture_em([0, 0.1, 0.3, 0.5, 1], BetaMixture(0.5, [0.5], [1], [2]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tolerance": 0},
        {"tolerance": 1},
        {"tolerance": np.nan},
        {"max_iterations": 0},
        {"max_iterations": True},
        {"max_iterations": 1.5},
    ],
)
def test_invalid_fit_controls(kwargs):
    with pytest.raises(ValueError):
        fit_beta_mixture_em([0.1, 0.2, 0.3, 0.4, 0.5], BetaMixture(0.5, [0.5], [1], [2]), **kwargs)
