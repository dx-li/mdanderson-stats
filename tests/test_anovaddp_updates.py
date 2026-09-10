import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import anovaddp_curve, anovaddp_subject_update, anovaddp_variance_posterior


def test_subject_transition_against_independent_r():
    reference = json.loads((Path(__file__).parent / "fixtures/anovaddp-subject.json").read_text())
    c = reference["conditional"]
    for case in reference["cases"]:
        fit = anovaddp_subject_update(
            c["parameters"],
            c["time"],
            c["observations"],
            prior_mean=c["prior_mean"],
            prior_covariance=c["prior_covariance"],
            variance=c["variance"],
            seed=case["seed"],
        )
        assert_allclose(fit.parameters, case["parameters"], atol=2e-13, rtol=0)
        assert_allclose(fit.log_acceptance, case["log_acceptance"], atol=2e-13, rtol=0)
        assert np.array_equal(fit.accepted, case["accepted"])
        assert_allclose(fit.normal_draws, case["normal_draws"], atol=0, rtol=0)
        assert_allclose(fit.uniform_draws, case["uniform_draws"], atol=0, rtol=0)
        with pytest.raises(ValueError):
            fit.parameters[0] = 0


def test_variance_prior_and_interleaved_subjects():
    theta = np.array([[2, -1, 4, 1, 3, 0.7], [1, 2, 3, 2, 4, 0.5]])
    t = np.array([0, 2, 5, 7])
    subject = np.array([1, 0, 1, 0])
    fitted = np.array([anovaddp_curve(theta[i], [v])[0] for i, v in zip(subject, t)])
    y = fitted + np.array([1, -2, 0.5, 3])
    documented = anovaddp_variance_posterior(theta, t, y, subject, alpha0=6, beta0=4)
    native = anovaddp_variance_posterior(theta, t, y, subject, alpha0=6, beta0=4, mode="source")
    assert documented.shape == native.shape == 5
    assert documented.residual_sum_squares == pytest.approx(14.25)
    assert documented.scale == pytest.approx(9.125)
    assert native.scale == pytest.approx(7.125)
    exact = anovaddp_variance_posterior(theta, t, fitted, subject, alpha0=6, beta0=4)
    assert exact.scale == 2
    with pytest.raises(ArithmeticError, match="zero"):
        anovaddp_variance_posterior(theta, t, fitted, subject, alpha0=6, beta0=4, mode="source")
    scaled_theta = theta.copy()
    scaled_theta[:, :3] *= 1e100
    scaled = anovaddp_variance_posterior(scaled_theta, t, y * 1e100, subject, alpha0=6, beta0=4e200)
    assert scaled.scale / 1e200 == pytest.approx(documented.scale, rel=1e-14)
