import numpy as np
import pytest
from scipy.optimize import brentq

from mdanderson_stats.synergy_surface import (
    fit_synergy_surface,
    predict_synergy_surface,
)


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


def test_raw_marginal_baseline_and_thin_plate_penalized_system() -> None:
    dose1, dose2, response = _raw_example()
    fit = fit_synergy_surface(dose1, dose2, response, smoothing_parameter=0.2)

    raw_coefficients = fit.baseline_coefficients * fit.response_scale
    raw_coefficients[1:] /= fit.dose_scale
    np.testing.assert_allclose(raw_coefficients, [2.0, 0.5, -0.25], atol=2e-14)
    assert (
        np.max(abs(fit.fitted_surface - predict_synergy_surface(fit, dose1, dose2).surface)) < 1e-12
    )

    knot_affine = np.column_stack((np.ones(fit.knots.shape[0]), fit.knots))
    np.testing.assert_allclose(knot_affine.T @ fit.radial_weights, 0.0, atol=1e-12)
    normalized_affine = np.column_stack(
        (np.ones(dose1.size), dose1 / fit.dose_scale[0], dose2 / fit.dose_scale[1])
    )
    combination = (dose1 > 0.0) & (dose2 > 0.0)
    residual = np.zeros(response.size)
    residual[combination] = (
        response[combination] - fit.fitted_baseline[combination]
    ) / fit.response_scale
    kernel_distance2 = (dose1[:, None] - fit.knots[None, :, 0]) ** 2 + (
        dose2[:, None] - fit.knots[None, :, 1]
    ) ** 2
    kernel = np.zeros_like(kernel_distance2)
    positive = kernel_distance2 > 0.0
    kernel[positive] = (
        kernel_distance2[positive] * np.log(kernel_distance2[positive]) / (16.0 * np.pi)
    )
    random_basis = kernel @ fit.constraint_nullspace
    penalty_coefficients = fit.constraint_nullspace.T @ fit.radial_weights
    lhs = np.block(
        [
            [normalized_affine.T @ normalized_affine, normalized_affine.T @ random_basis],
            [
                random_basis.T @ normalized_affine,
                random_basis.T @ random_basis + fit.smoothing_parameter * fit.penalty_matrix,
            ],
        ]
    )
    rhs = np.concatenate((normalized_affine.T @ residual, random_basis.T @ residual))
    direct = np.linalg.solve(lhs, rhs)
    np.testing.assert_allclose(direct[:3], fit.affine_coefficients, atol=2e-10)
    np.testing.assert_allclose(direct[3:], penalty_coefficients, atol=2e-10)


def test_log_baseline_recovers_marginals_and_varying_potency_root() -> None:
    dose1, dose2, response = _log_example()
    fit = fit_synergy_surface(dose1, dose2, response, baseline="log", smoothing_parameter=0.2)

    np.testing.assert_allclose(
        fit.baseline_coefficients * fit.response_scale,
        [0.3, 1.2, -0.4, 0.9],
        atol=2e-13,
    )
    prediction = predict_synergy_surface(fit, np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    dose = np.array([1.0, 2.0])
    additive_dose_fraction = dose / np.exp((prediction.baseline - 0.3) / 1.2) + dose / np.exp(
        (prediction.baseline + 0.4) / 0.9
    )
    np.testing.assert_allclose(additive_dose_fraction, 1.0, atol=2e-13)
    assert np.isnan(fit.fitted_baseline[-1])
    with pytest.raises(ValueError, match="both-zero"):
        predict_synergy_surface(fit, [0.0], [0.0])


def test_reml_fit_and_exact_affine_residual_failure() -> None:
    dose1, dose2, response = _raw_example()
    fit = fit_synergy_surface(dose1, dose2, response)
    assert fit.optimizer_success
    assert not fit.smoothing_at_boundary
    assert 0.0 < fit.smoothing_parameter < np.inf
    assert fit.residual_variance_scaled > 0.0
    assert fit.optimizer_evaluations > 0

    additive = 2.0 + 0.5 * dose1 - 0.25 * dose2
    with pytest.raises(ValueError, match="affine to numerical precision"):
        fit_synergy_surface(dose1, dose2, additive, smoothing_parameter=0.2)
