import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import BetaMixture, beta_mixture_testing

REFERENCE = json.loads((Path(__file__).parent / "fixtures/beta_testing.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["cases"])
def test_original_desktop_decisions(case):
    result = beta_mixture_testing(
        case["pvalues"],
        BetaMixture(**case["model"]),
        alpha=case["alpha"],
        sequence=case["sequence"],
        legacy_endpoints=True,
    )
    np.testing.assert_allclose(result.scores, case["scores"], rtol=2e-12, atol=1e-300)
    np.testing.assert_array_equal(result.reject, case["reject"])


def test_desktop_score_is_not_null_posterior():
    model = BetaMixture(0.2, [0.8], [2], [2])
    result = beta_mixture_testing([0.5], model, alpha=0.2)
    assert result.scores[0] == pytest.approx(1 / 1.4)
    assert model.null_posterior([0.5])[0] == pytest.approx(0.2 / 1.4)
    assert not result.reject[0]


def test_rank_and_entered_order_have_different_prefixes():
    model = BetaMixture(0, [1], [2], [2])
    values = [0.5, 0.01, 0.99]
    ranked = beta_mixture_testing(values, model, alpha=0.7)
    entered = beta_mixture_testing(values, model, alpha=0.7, sequence="input")
    np.testing.assert_array_equal(ranked.scores, [1, 1, 1])
    np.testing.assert_allclose(entered.scores, [2 / 3, 1, 1])
    np.testing.assert_array_equal(entered.reject, [True, False, False])
    np.testing.assert_array_equal(ranked.order, [1, 0, 2])


def test_polynomial_density_and_inclusive_threshold():
    result = beta_mixture_testing([0, 0.25, 0.5, 1], BetaMixture(0, [1], [1], [2]), alpha=0.5)
    np.testing.assert_allclose(result.scores, [0.5, 2 / 3, 1, 1])
    np.testing.assert_array_equal(result.reject, [True, False, False, False])


def test_batched_families_preserve_inputs_and_do_not_share_prefixes():
    x = np.array([[0.5, 0.01, 0.99], [0.01, 0.5, 0.99]])
    original = x.copy()
    model = BetaMixture(0, [1], [2], [2])
    result = beta_mixture_testing(x, model, sequence="input", alpha=0.7)
    np.testing.assert_allclose(result.scores, [[2 / 3, 1, 1], [1, 1, 1]])
    np.testing.assert_array_equal(x, original)
    np.testing.assert_allclose(result.log_density, model.logpdf(x))


def test_singular_and_zero_endpoint_limits():
    model = BetaMixture(0, [1], [0.5], [2])
    result = beta_mixture_testing([0, 1], model, alpha=0)
    np.testing.assert_array_equal(result.scores, [0, 1])
    np.testing.assert_array_equal(result.reject, [True, False])
    legacy = beta_mixture_testing([0], model, alpha=0, legacy_endpoints=True)
    assert legacy.scores[0] > 0 and not legacy.reject[0]
    uniform = beta_mixture_testing([0, 1], BetaMixture(1), alpha=1)
    assert np.all(uniform.reject)


@pytest.mark.parametrize("values", [[], [np.nan], [-0.1], [1.1], 0.5])
def test_invalid_pvalues(values):
    with pytest.raises(ValueError):
        beta_mixture_testing(values, BetaMixture(1))


@pytest.mark.parametrize("kwargs", [{"alpha": np.inf}, {"alpha": -1}, {"sequence": "sorted"}])
def test_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        beta_mixture_testing([0.5], BetaMixture(1), **kwargs)
