from dataclasses import astuple
from functools import partial
from math import asin, pi

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import beta, binom, norm

from mdanderson_stats.success_calibration import (
    binary_success_oc,
    binary_two_arm_success_oc,
    calibrate_success_cutoff,
    normal_success_oc,
)


def test_exact_binary_and_calibration():
    result = binary_success_oc(1, 0.5, margin=0.5)
    np.testing.assert_allclose(astuple(result)[:4], [3 / 8, 1 / 8, 3 / 8, 1 / 8], atol=1e-15)
    assert result.incorrect_decision_probability == pytest.approx(1 / 4)
    # Integrate the frequentist count rejection probability over the design prior.
    n, cutoff, margin = 19, 0.8, 0.35
    design, analysis = (2.4, 3.1), (0.7, 1.3)
    counts = np.arange(n + 1)
    success = beta.sf(margin, analysis[0] + counts, analysis[1] + n - counts) > cutoff

    def integrand(p):
        return binom.pmf(counts[success], n, p).sum() * beta.pdf(p, *design)

    fp = quad(integrand, 0, margin, epsabs=1e-12)[0]
    tp = quad(integrand, margin, 1, epsabs=1e-12)[0]
    result = binary_success_oc(
        n, cutoff, margin=margin, design_prior=design, analysis_prior=analysis
    )
    np.testing.assert_allclose([result.true_positive, result.false_positive], [tp, fp], atol=1e-11)
    reflected = binary_success_oc(
        n,
        cutoff,
        margin=1 - margin,
        design_prior=design[::-1],
        analysis_prior=analysis[::-1],
        direction="less",
    )
    np.testing.assert_allclose(astuple(result), astuple(reflected), atol=1e-13)
    evaluate = partial(binary_success_oc, 10, margin=0.5)
    chosen = calibrate_success_cutoff(evaluate, 0.05, np.linspace(0.5, 0.99, 50))
    assert chosen.operating_characteristics.incorrect_decision_probability <= 0.05
    assert evaluate(chosen.cutoff - 0.01).incorrect_decision_probability > 0.05
    assert evaluate(1).incorrect_decision_probability is None
    with pytest.raises(ValueError, match="no candidate"):
        calibrate_success_cutoff(evaluate, 0.01, [1.0])


def test_normal_quadrants_and_unequal_arm_priors():
    result = normal_success_oc(0.5, standard_error=1.0)
    q = asin(1 / np.sqrt(2)) / (2 * pi)
    np.testing.assert_allclose(
        astuple(result)[:4], [0.25 + q, 0.25 - q, 0.25 + q, 0.25 - q], atol=1e-12
    )
    # Independent simulation draws the true arm means, sample means, and conjugate
    # posterior via precision arithmetic. Unequal priors make pooling invalid.
    rng = np.random.default_rng(173)
    dm, ds = np.array([0.4, -0.1]), np.array([0.7, 0.4])
    am, ass, se = np.array([0.1, 0.2]), np.array([0.3, 1.2]), np.array([0.4, 0.6])
    truth = rng.normal(dm, ds, (300000, 2))
    data = rng.normal(truth, se)
    variance = 1 / (1 / ass**2 + 1 / se**2)
    means = variance * (am / ass**2 + data / se**2)
    effective = truth[:, 0] - truth[:, 1] > 0.2
    success = norm.sf((0.2 - (means[:, 0] - means[:, 1])) / np.sqrt(variance.sum())) > 0.85
    observed = np.array(
        [
            np.mean(effective & success),
            np.mean(~effective & success),
            np.mean(~effective & ~success),
            np.mean(effective & ~success),
        ]
    )
    result = normal_success_oc(
        0.85,
        standard_error=se,
        design_mean=dm,
        design_sd=ds,
        analysis_mean=am,
        analysis_sd=ass,
        margin=0.2,
        null_mean=[0.2, 0.0],
    )
    exact = np.array(astuple(result)[:4])
    assert np.all(abs(observed - exact) < 6 * np.sqrt(exact * (1 - exact) / len(truth)))
    reflected = normal_success_oc(
        0.85,
        standard_error=se,
        design_mean=-dm,
        design_sd=ds,
        analysis_mean=-am,
        analysis_sd=ass,
        margin=-0.2,
        direction="less",
        null_mean=[-0.2, 0.0],
    )
    np.testing.assert_allclose(astuple(result), astuple(reflected), atol=1e-12)
    # A very rare success tail is evaluated directly, not 1 minus a rounded CDF.
    rare = normal_success_oc(0.999999, standard_error=1.0, analysis_mean=-10.0)
    assert 0 < rare.bayesian_power < 1e-20
    assert normal_success_oc(1.0, standard_error=1.0).incorrect_decision_probability is None


def test_two_arm_binary_exact_identity():
    result = binary_two_arm_success_oc(1, 1, 0.5)
    # Only (treatment success, control failure) declares superiority. Its design
    # posterior ordering probability is 5/6, with prior predictive mass 1/4.
    np.testing.assert_allclose(astuple(result)[:4], [5 / 24, 1 / 24, 11 / 24, 7 / 24], atol=1e-10)
    assert result.frequentist_type1_error == pytest.approx(0.25)
    assert result.incorrect_decision_probability == pytest.approx(1 / 6)
    reflected = binary_two_arm_success_oc(1, 1, 0.5, direction="less")
    np.testing.assert_allclose(astuple(result), astuple(reflected), atol=1e-10)


def test_normal_weak_truth_score_correlation():
    result = normal_success_oc(0.7, standard_error=1e6, analysis_sd=1e6)
    assert result.incorrect_decision_probability == pytest.approx(0.5, abs=1e-6)
    assert result.bayesian_power == pytest.approx(norm.sf(np.sqrt(2) * norm.ppf(0.7)), abs=1e-12)
