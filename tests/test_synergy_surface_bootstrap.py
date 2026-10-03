import copy
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from mdanderson_stats.synergy_surface import fit_synergy_surface
from mdanderson_stats.synergy_surface_bootstrap import bootstrap_synergy_surface


def _raw_example() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = [(a, b) for a in (0.0, 1.0, 2.0) for b in (0.0, 1.0, 2.0)]
    pairs.append((1.0, 1.0))
    dose1 = np.array([a for a, _ in pairs])
    dose2 = np.array([b for _, b in pairs])
    response = 2.0 + 0.5 * dose1 - 0.25 * dose2 + 0.2 * dose1 * dose2
    response[-1] += 0.07
    return dose1, dose2, response


def _log_example() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = (
        [(dose, 0.0) for dose in (1.0, 2.0, 4.0)]
        + [(0.0, dose) for dose in (1.0, 2.0, 4.0)]
        + [(a, b) for a in (1.0, 2.0) for b in (1.0, 2.0)]
        + [(0.0, 0.0)]
    )
    beta0, beta1, alpha0, alpha1 = 0.3, 1.2, -0.4, 0.9
    gamma1 = (alpha0 - beta0) / beta1
    gamma2 = alpha1 / beta1 - 1.0
    response = []
    for dose1, dose2 in pairs:
        if dose1 > 0.0 and dose2 > 0.0:
            root = brentq(
                lambda u: u - gamma1 - gamma2 * np.log(dose1 * np.exp(-u) + dose2),
                -10.0,
                10.0,
            )
            value = beta0 + beta1 * np.log(dose1 + np.exp(root) * dose2)
            value += 0.05 * dose1 * dose2
        elif dose1 > 0.0:
            value = beta0 + beta1 * np.log(dose1)
        elif dose2 > 0.0:
            value = alpha0 + alpha1 * np.log(dose2)
        else:
            value = 7.0
        response.append(value)
    return (
        np.array([a for a, _ in pairs]),
        np.array([b for _, b in pairs]),
        np.array(response),
    )


def test_multiplier_tape_replays_documented_resample_and_reports_sample_sd() -> None:
    dose1, dose2, response = _raw_example()
    sqrt5 = np.sqrt(5.0)
    low, high = (1.0 - sqrt5) / 2.0, (1.0 + sqrt5) / 2.0
    tape = np.array([[low, high] * 5, [high, low] * 5])
    lam = 0.2

    result = bootstrap_synergy_surface(
        dose1,
        dose2,
        response,
        replicates=2,
        smoothing_parameter=lam,
        multiplier_tape=tape,
    )
    assert result.departure_draws is not None
    lower = fit_synergy_surface(dose1, dose2, response, smoothing_parameter=lam / 2.0)
    upper = fit_synergy_surface(dose1, dose2, response, smoothing_parameter=2.0 * lam)
    residual = response - lower.fitted_baseline - lower.fitted_surface
    center = upper.fitted_baseline + upper.fitted_surface
    expected = []
    for weights in tape:
        synthetic = center + residual * weights
        expected.append(fit_synergy_surface(dose1, dose2, synthetic).fitted_surface)
    expected_draws = np.stack(expected)

    np.testing.assert_allclose(result.departure_draws, expected_draws, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result.original_departure, result.fit.fitted_surface)
    np.testing.assert_allclose(result.departure_mean, np.mean(expected_draws, axis=0))
    np.testing.assert_allclose(
        result.departure_standard_deviation, np.std(expected_draws, axis=0, ddof=1)
    )
    assert result.multiplier_tape_used
    assert result.replicate_smoothing_parameters.shape == (2,)
    assert result.replicate_optimizer_evaluations.shape == (2,)
    assert result.replicate_smoothing_at_boundary.shape == (2,)


def test_fixed_tape_matches_independent_r_bootstrap_reference() -> None:
    fixture_dir = Path(__file__).parent / "fixtures"
    anchors = np.genfromtxt(
        fixture_dir / "synergy-bootstrap-anchors.csv", delimiter=",", names=True
    )
    reml = np.genfromtxt(fixture_dir / "synergy-bootstrap-reml.csv", delimiter=",", names=True)
    reference_draws = np.genfromtxt(
        fixture_dir / "synergy-bootstrap-draws.csv", delimiter=",", names=True
    )
    reference_summary = np.genfromtxt(
        fixture_dir / "synergy-bootstrap-summary.csv", delimiter=",", names=True
    )
    dose1 = anchors["dose1"]
    dose2 = anchors["dose2"]
    response = anchors["response"]
    replicate_ids = np.unique(reference_draws["replicate"])
    tape = np.stack(
        [
            reference_draws[reference_draws["replicate"] == replicate]["multiplier"]
            for replicate in replicate_ids
        ]
    )

    result = bootstrap_synergy_surface(
        dose1,
        dose2,
        response,
        replicates=len(replicate_ids),
        multiplier_tape=tape,
    )
    assert result.departure_draws is not None
    lower_residual = response - result.lower_smoothing_fit.fitted_baseline
    lower_residual -= result.lower_smoothing_fit.fitted_surface
    upper_center = (
        result.upper_smoothing_fit.fitted_baseline + result.upper_smoothing_fit.fitted_surface
    )
    python_synthetic = upper_center + lower_residual * tape
    reference_synthetic = np.stack(
        [
            reference_draws[reference_draws["replicate"] == replicate]["synthetic_response"]
            for replicate in replicate_ids
        ]
    )
    np.testing.assert_allclose(python_synthetic, reference_synthetic, atol=2e-8, rtol=0.0)
    np.testing.assert_allclose(result.fit.smoothing_parameter, reml["original_lambda"], rtol=3e-7)
    np.testing.assert_allclose(result.original_departure, anchors["original_departure"], atol=3e-9)
    np.testing.assert_allclose(
        result.lower_smoothing_fit.fitted_surface, anchors["lower_departure"], atol=3e-9
    )
    np.testing.assert_allclose(
        result.upper_smoothing_fit.fitted_surface, anchors["upper_departure"], atol=3e-9
    )
    reference_matrix = np.stack(
        [
            reference_draws[reference_draws["replicate"] == replicate]["departure_draw"]
            for replicate in replicate_ids
        ]
    )
    np.testing.assert_allclose(result.departure_draws, reference_matrix, rtol=5e-7, atol=3e-9)
    np.testing.assert_allclose(
        result.departure_mean, reference_summary["departure_mean"], atol=3e-9
    )
    np.testing.assert_allclose(
        result.departure_standard_deviation,
        reference_summary["departure_sd_ddof1"],
        rtol=1e-7,
        atol=3e-9,
    )
    np.testing.assert_allclose(
        result.replicate_smoothing_parameters,
        [
            reference_draws[reference_draws["replicate"] == replicate]["refit_lambda"][0]
            for replicate in replicate_ids
        ],
        rtol=5e-7,
    )


def test_bootstrap_can_summarize_without_retaining_draw_matrix() -> None:
    dose1, dose2, response = _raw_example()
    result = bootstrap_synergy_surface(
        dose1,
        dose2,
        response,
        replicates=2,
        smoothing_parameter=0.2,
        multiplier_tape=np.full((2, response.size), (1.0 - np.sqrt(5.0)) / 2.0),
        store_draws=False,
    )
    assert result.departure_draws is None
    assert np.all(np.isfinite(result.departure_mean))
    assert np.all(np.isfinite(result.departure_standard_deviation))


def test_log_baseline_rejects_both_zero_before_consuming_randomness() -> None:
    dose1, dose2, response = _log_example()
    generator = np.random.default_rng(934)
    state_before = copy.deepcopy(generator.bit_generator.state)
    with pytest.raises(ValueError, match="both-zero"):
        bootstrap_synergy_surface(
            dose1,
            dose2,
            response,
            baseline="log",
            replicates=2,
            rng=generator,
        )
    assert generator.bit_generator.state == state_before


def test_log_baseline_refits_with_positive_marginal_slopes() -> None:
    dose1, dose2, response = _log_example()
    keep = (dose1 != 0.0) | (dose2 != 0.0)
    dose1, dose2, response = dose1[keep], dose2[keep], response[keep]
    sqrt5 = np.sqrt(5.0)
    tape = np.resize(
        np.array([(1.0 - sqrt5) / 2.0, (1.0 + sqrt5) / 2.0]),
        (2, response.size),
    )
    result = bootstrap_synergy_surface(
        dose1,
        dose2,
        response,
        baseline="log",
        replicates=2,
        smoothing_parameter=0.2,
        multiplier_tape=tape,
    )
    assert result.fit.baseline == "log"
    assert result.departure_draws is not None
    assert np.all(np.isfinite(result.departure_draws))
    assert np.all(result.replicate_smoothing_parameters > 0.0)


def test_invalid_multiplier_tape_is_rejected() -> None:
    dose1, dose2, response = _raw_example()
    with pytest.raises(ValueError, match="Mammen"):
        bootstrap_synergy_surface(
            dose1,
            dose2,
            response,
            replicates=2,
            smoothing_parameter=0.2,
            multiplier_tape=np.zeros((2, response.size)),
        )


def test_oversized_fit_work_is_rejected_before_randomness() -> None:
    dose1 = np.arange(500, dtype=float)
    dose2 = np.arange(500, dtype=float)
    response = 1.0 + dose1 * 0.001
    generator = np.random.default_rng(2901)
    state_before = copy.deepcopy(generator.bit_generator.state)
    with pytest.raises(ValueError, match="work budget"):
        bootstrap_synergy_surface(dose1, dose2, response, replicates=2, rng=generator)
    assert generator.bit_generator.state == state_before
