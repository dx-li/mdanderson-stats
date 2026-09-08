import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import BetaMixture, BetaMixtureFitError, fit_beta_mixture_ml
from mdanderson_stats.beta_mixture_ml import _coordinates, _model, _objective

REFERENCE = json.loads((Path(__file__).parent / "fixtures/beta_mixture.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["ml_fits"])
def test_direct_fit_against_original_likelihood(case):
    x = case["pvalues"]
    fit = fit_beta_mixture_ml(
        x, BetaMixture(**case["initial"]), legacy_endpoints=case["legacy_endpoints"]
    )
    reference = case["reference"]
    assert reference["status"] in (3, 4, 5, 6)
    # Different solvers may choose different local optima. These starts reach
    # likelihoods at least as high as the archived optimizer's results.
    assert fit.log_likelihood >= reference["log_likelihood"] - 1e-7
    if len(fit.model.weights) == 1:
        assert fit.log_likelihood == pytest.approx(reference["log_likelihood"], abs=4e-5)
    else:
        em = REFERENCE["em_fits"][1]["reference"]
        assert fit.log_likelihood == pytest.approx(em["log_likelihood"], abs=1e-5)
    assert fit.iterations == len(fit.log_likelihood_history) - 1
    assert np.min(np.diff(fit.log_likelihood_history)) > -1e-7
    assert fit.model.null_weight + fit.model.weights.sum() == pytest.approx(1)


@pytest.mark.parametrize("weights", [[0.2, 0.3], [0, 0.3], [0.7, 0.3], [1, 0]])
def test_feasible_boundaries_and_analytic_scores(weights):
    model = BetaMixture(1 - sum(weights), weights, [0.7, 2], [4, 3])
    theta = _coordinates(model)
    restored = _model(theta)
    np.testing.assert_allclose(restored.weights, weights, atol=1e-15)
    assert restored.null_weight == pytest.approx(model.null_weight)
    x = np.linspace(0.1, 0.9, 31)
    lx, ly = np.log(x), np.log1p(-x)
    value, gradient = _objective(theta, lx, ly)
    assert value == pytest.approx(-model.log_likelihood(x) / len(x))
    for j in range(len(theta)):
        lo, hi = theta.copy(), theta.copy()
        step = 1e-6
        lo[j] -= step
        hi[j] += step
        if j < len(weights):
            lo[j] = max(0, lo[j])
            hi[j] = min(1, hi[j])
        numerical = (_objective(hi, lx, ly)[0] - _objective(lo, lx, ly)[0]) / (hi[j] - lo[j])
        assert gradient[j] == pytest.approx(numerical, rel=2e-4, abs=2e-5)


def test_uniform_model_and_failure_controls():
    fit = fit_beta_mixture_ml([0, 0.5, 1], BetaMixture(1))
    assert fit.iterations == 0 and fit.log_likelihood == 0
    case = REFERENCE["ml_fits"][0]
    for limits in ({"max_iterations": 1}, {"max_evaluations": 1}):
        with pytest.raises(BetaMixtureFitError, match="optimizer failed"):
            fit_beta_mixture_ml(case["pvalues"], BetaMixture(**case["initial"]), **limits)
    with pytest.raises(ValueError, match="shapes"):
        fit_beta_mixture_ml(case["pvalues"], BetaMixture(0.5, [0.5], [1e-11], [2]))
    with pytest.raises(ValueError, match="interior"):
        fit_beta_mixture_ml([0, 0.1, 0.3, 0.5, 1], BetaMixture(0.5, [0.5], [1], [2]))
    with pytest.raises(BetaMixtureFitError, match="n>3"):
        fit_beta_mixture_ml([0.1, 0.2, 0.3], BetaMixture(0.5, [0.5], [1], [2]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tolerance": 0},
        {"gradient_tolerance": np.nan},
        {"max_iterations": True},
        {"max_evaluations": 0},
        {"max_evaluations": 1.5},
    ],
)
def test_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        fit_beta_mixture_ml(np.linspace(0.1, 0.9, 20), BetaMixture(1), **kwargs)
