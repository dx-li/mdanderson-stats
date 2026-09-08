import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    BetaMixture,
    BetaMixtureFitError,
    beta_mixture_bootstrap,
    fit_beta_mixture_k,
    select_beta_mixture,
)
from mdanderson_stats.beta_mixture_bootstrap import _sample
from mdanderson_stats.beta_mixture_selection import _relative_likelihood_change

REFERENCE = json.loads((Path(__file__).parent / "fixtures/beta_selection.json").read_text())


@pytest.mark.parametrize("criterion, selected, count", [("data", 1, 2), ("lglk", 2, 3)])
@pytest.mark.parametrize("workflow", ["s", "desktop"])
def test_em_selection_matches_native_sequential_fits(criterion, selected, count, workflow):
    result = select_beta_mixture(
        REFERENCE["published"], criterion=criterion, algorithm="em", workflow=workflow
    )
    assert result.status == "criterion_met"
    assert result.workflow == workflow
    assert len(result.fit.model.weights) == selected
    assert len(result.candidates) == count + 1
    for candidate, reference in zip(result.candidates[1:], REFERENCE["em_chain"], strict=False):
        assert candidate.log_likelihood == pytest.approx(reference["log_likelihood"], abs=2e-7)
        assert candidate.cramer_von_mises == pytest.approx(reference["cramer_von_mises"], rel=1e-6)
        np.testing.assert_allclose(
            candidate.model.weights, reference["model"]["weights"], rtol=1e-6
        )


def test_manual_component_fits_match_desktop_em():
    previous = fit_beta_mixture_k(REFERENCE["published"], 0, algorithm="em")
    assert previous.model.null_weight == 1 and previous.iterations == 0
    for k, reference in enumerate(REFERENCE["desktop_em_fits"], 1):
        fit = fit_beta_mixture_k(REFERENCE["published"], k, previous, algorithm="em")
        assert fit.log_likelihood == pytest.approx(reference["log_likelihood"], abs=2e-7)
        np.testing.assert_allclose(fit.model.weights, reference["model"]["weights"], rtol=1e-6)
        previous = fit


def test_desktop_likelihood_denominator_floor():
    floor = 100 * np.finfo(float).tiny
    assert _relative_likelihood_change(0, 0, "desktop") == 0
    assert _relative_likelihood_change(0, floor, "desktop") == 1
    assert np.isinf(_relative_likelihood_change(0, floor, "s"))
    assert _relative_likelihood_change(-1, 0, "desktop") == pytest.approx(1 / floor)
    assert _relative_likelihood_change(-1, 0, "s") == -1
    assert _relative_likelihood_change(100, 104, "desktop") == pytest.approx(0.04)


@pytest.mark.parametrize("k", [-1, 11, True, 1.5])
def test_invalid_manual_component_count(k):
    with pytest.raises(ValueError, match="integer"):
        fit_beta_mixture_k(REFERENCE["published"], k)


def test_manual_fit_requires_the_correct_preceding_model():
    x = REFERENCE["published"]
    uniform = fit_beta_mixture_k(x, 0)
    with pytest.raises(ValueError, match="required"):
        fit_beta_mixture_k(x, 2)
    with pytest.raises(ValueError, match="previous"):
        fit_beta_mixture_k(x, 2, uniform)
    with pytest.raises(ValueError, match="previous"):
        fit_beta_mixture_k(x, 0, uniform)
    with pytest.raises(BetaMixtureFitError, match="n>3"):
        fit_beta_mixture_k([0.01, 0.02, 0.03], 1)
    with pytest.raises(ValueError, match="workflow"):
        select_beta_mixture(x, workflow="unknown")


@pytest.mark.parametrize("algorithm", ["em", "ml"])
def test_selection_limit_and_failed_start_are_distinguished(algorithm):
    limited = select_beta_mixture(REFERENCE["published"], algorithm=algorithm, max_components=1)
    assert limited.status == "component_limit" and len(limited.fit.model.weights) == 1
    failed = select_beta_mixture(np.linspace(0.1, 0.9, 40), algorithm=algorithm)
    assert failed.status == "fit_failed"
    assert "k=1" in failed.message
    assert failed.fit.model.null_weight == 1
    assert len(failed.candidates) == 1


def test_bootstrap_refits_agree_with_native_em():
    result = beta_mixture_bootstrap(
        np.linspace(0.01, 0.9, 60),
        BetaMixture(**REFERENCE["bootstrap_model"]),
        algorithm="em",
        replicates=4,
        rng=REFERENCE["bootstrap_seed"],
    )
    expected = [case["reference"]["cramer_von_mises"] for case in REFERENCE["bootstrap_samples"]]
    np.testing.assert_allclose(result.simulated_statistics, expected, rtol=2e-7)
    assert result.pvalue == 0 and result.attempts == 4 and not result.failures
    assert not result.simulated_statistics.flags.writeable


def test_uniform_bootstrap_exact_statistic_strict_tail_and_reproducibility():
    seed, n, m = 41, 20, 30
    draws = np.random.default_rng(seed).random((m, n))
    # Observed data equals one simulated replicate: equality must not count.
    x = draws[0]
    ranks = (np.arange(n) + 0.5) / n
    expected = np.mean((np.sort(draws, axis=1) - ranks) ** 2, axis=1) + 1 / (12 * n**2)
    result = beta_mixture_bootstrap(x, BetaMixture(1), replicates=m, rng=seed)
    np.testing.assert_allclose(result.simulated_statistics, expected, atol=1e-17)
    assert result.pvalue == np.count_nonzero(expected > expected[0]) / m
    assert result.attempts == m
    again = beta_mixture_bootstrap(x, BetaMixture(1), replicates=m, rng=seed)
    np.testing.assert_array_equal(result.simulated_statistics, again.simulated_statistics)


def test_pcvm_selects_current_uniform_model():
    result = select_beta_mixture((np.arange(50) + 0.5) / 50, criterion="pcvm", rng=2, replicates=20)
    assert result.status == "criterion_met" and result.fit.model.null_weight == 1
    assert len(result.candidates) == 1 and result.bootstrap_checks[0].pvalue == 1


@pytest.mark.parametrize("algorithm", ["ml", "em"])
def test_pcvm_rejects_uniform_then_refits_mixture(algorithm):
    result = select_beta_mixture(
        REFERENCE["published"],
        criterion="pcvm",
        algorithm=algorithm,
        replicates=10,
        rng=82,
        max_components=2,
    )
    assert result.status == "criterion_met"
    assert len(result.fit.model.weights) == 1
    assert result.bootstrap_checks[0].pvalue == 0
    assert result.bootstrap_checks[1].pvalue > 0.05


def test_generated_mixture_has_specified_cdf():
    model = BetaMixture(0.4, [0.2, 0.4], [0.5, 3], [5, 2])
    samples = _sample(model, 50000, np.random.default_rng(8))
    points = np.array([0.05, 0.2, 0.5, 0.8])
    np.testing.assert_allclose(
        (samples[:, None] <= points).mean(axis=0), model.cdf(points), atol=0.008
    )


def test_refit_failure_exhausts_bounded_attempts():
    with pytest.raises(BetaMixtureFitError, match="0/2 valid fits in 3 attempts"):
        beta_mixture_bootstrap(
            np.linspace(0.01, 0.9, 60),
            BetaMixture(**REFERENCE["bootstrap_model"]),
            algorithm="em",
            replicates=2,
            max_attempts=3,
            max_iterations=1,
            rng=56,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"criterion": "aic"},
        {"threshold": 0},
        {"threshold": np.nan},
        {"max_components": 11},
        {"max_components": True},
        {"replicates": 0},
        {"algorithm": "dgay"},
        {"tolerance": 1},
        {"max_iterations": 0},
    ],
)
def test_invalid_selection_controls(kwargs):
    with pytest.raises(ValueError):
        select_beta_mixture(np.linspace(0.1, 0.9, 40), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"replicates": True},
        {"replicates": 0},
        {"max_attempts": 1},
        {"max_attempts": 1.5},
        {"tolerance": np.inf},
        {"algorithm": "invalid"},
    ],
)
def test_invalid_bootstrap_controls(kwargs):
    with pytest.raises(ValueError):
        beta_mixture_bootstrap(np.linspace(0.1, 0.9, 40), BetaMixture(1), **kwargs)
