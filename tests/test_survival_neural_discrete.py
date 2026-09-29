import numpy as np
import pytest
from scipy.special import expit

from mdanderson_stats.survival_neural_discrete import _loss_gradient


def _finite_difference(function, value, step=1e-6):
    gradient = np.empty_like(value)
    for index in np.ndindex(value.shape):
        plus = value.copy()
        minus = value.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (function(plus) - function(minus)) / (2.0 * step)
    return gradient


def test_logistic_hazard_closed_form_event_and_exact_cut_censor() -> None:
    cuts = np.array([0.0, 1.0, 2.0])
    logits = np.array([[0.2, -0.4, 0.7]])
    event_loss, event_gradient, _, _ = _loss_gradient("loghaz", [1.5], [1], logits, cuts)
    expected_event_loss = (
        np.logaddexp(0.0, logits[0, 0])
        + np.logaddexp(0.0, logits[0, 1])
        + np.logaddexp(0.0, -logits[0, 2])
    )
    expected_event_gradient = np.array(
        [[expit(logits[0, 0]), expit(logits[0, 1]), expit(logits[0, 2]) - 1.0]]
    )
    assert event_loss == pytest.approx(expected_event_loss)
    np.testing.assert_allclose(event_gradient, expected_event_gradient, rtol=0.0, atol=1e-14)

    censor_loss, censor_gradient, _, _ = _loss_gradient("loghaz", [1.0], [0], logits, cuts)
    assert censor_loss == pytest.approx(
        np.logaddexp(0.0, logits[0, 0]) + np.logaddexp(0.0, logits[0, 1])
    )
    np.testing.assert_allclose(censor_gradient, [[expit(logits[0, 0]), expit(logits[0, 1]), 0.0]])


def test_logistic_hazard_admin_censoring_never_turns_late_event_into_event() -> None:
    cuts = np.array([0.0, 1.0, 2.0])
    logits = np.array([[0.1, -0.2, 0.3]])
    late, _, _, _ = _loss_gradient("loghaz", [8.0], [1], logits, cuts)
    at_max_censored, _, _, _ = _loss_gradient("loghaz", [2.0], [0], logits, cuts)
    assert late == pytest.approx(at_max_censored)


def test_deephit_joint_nll_and_ranking_gradient_matches_finite_difference() -> None:
    cuts = np.array([0.0, 1.0, 2.0])
    time = np.array([0.5, 1.0, 1.5])
    event = np.array([1.0, 1.0, 0.0])
    logits = np.array([[0.2, -0.1, 0.4], [-0.3, 0.5, 0.1], [0.1, -0.4, 0.3]])
    options = {"alpha": 0.35, "rank_sigma": 0.6}
    loss, gradient, _, _ = _loss_gradient("deephit", time, event, logits, cuts, **options)
    numerical = _finite_difference(
        lambda candidate: _loss_gradient("deephit", time, event, candidate, cuts, **options)[0],
        logits,
    )
    assert np.isfinite(loss)
    np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-8)


def test_pchazard_fractional_exposure_gradient_and_initial_censor_label() -> None:
    cuts = np.array([0.0, 1.0, 3.0])
    time = np.array([0.0, 0.5, 2.0, 3.0, 4.0])
    event = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    logits = np.array([[0.0, 0.0], [0.2, -0.4], [-0.1, 0.6], [0.5, -0.2], [0.3, 0.1]])
    loss, gradient, _, _ = _loss_gradient("pchazard", time, event, logits, cuts)
    numerical = _finite_difference(
        lambda candidate: _loss_gradient("pchazard", time, event, candidate, cuts)[0],
        logits,
    )
    assert gradient[0].tolist() == [0.0, 0.0]
    assert np.isfinite(loss)
    np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-8)
    with pytest.raises(ValueError, match="initial cut"):
        _loss_gradient("pchazard", [0.0], [1.0], [[0.1, 0.2]], cuts)


def test_discrete_family_rejects_out_of_domain_start_and_bad_shapes() -> None:
    with pytest.raises(ValueError, match="start at or before"):
        _loss_gradient("loghaz", [0.0], [1], [[0.0, 0.0]], [0.1, 1.0])
    with pytest.raises(ValueError, match="one output per interval"):
        _loss_gradient("pchazard", [0.5], [1], [[0.0, 0.1, 0.2]], [0.0, 1.0])
