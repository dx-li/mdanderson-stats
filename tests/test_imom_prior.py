"""Variable-shape iMOM identities and the live BFMonitor default boundaries."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.special import expit, logit

from mdanderson_stats import IMOMBinaryPrior, bayes_factor_binary_design


@pytest.mark.parametrize("shape", [0.5, 1.25, 2.04, 5])
def test_prior_and_bayes_factor_against_direct_integrals(shape):
    prior = IMOMBinaryPrior(0.2, 0.4, shape)
    density_mass, error = quad(
        lambda p: float(prior.pdf(p)), 0.2, 1, points=[0.4], epsabs=1e-12, epsrel=1e-12
    )
    assert error < 1e-8
    assert_allclose(density_mass, 1, atol=1e-11)
    assert prior.pdf(0.4) > max(prior.pdf(0.399), prior.pdf(0.401))
    p = np.array([1e-8, 0.025, 0.5, 0.975, 1 - 1e-8])
    assert_allclose(prior.cdf(prior.quantile(p)), p, rtol=1e-10, atol=1e-13)
    assert_array_equal(prior.quantile([0, 1]), [0.2, 1])
    trial = bayes_factor_binary_design(10, alternative_mode=0.4, imom_shape=shape, min_subjects=5)
    for m in [0, 4, 10]:
        bf, error = quad(
            lambda p: float(prior.pdf(p)) * (p / 0.2) ** m * ((1 - p) / 0.8) ** (10 - m),
            0.2,
            1,
            points=[0.4],
            epsabs=1e-11,
            epsrel=1e-11,
        )
        assert error < max(1e-10, bf * 1e-8)
        assert_allclose(trial.log_bayes_factor[10, m], np.log(bf), atol=2e-9, rtol=0)


def test_live_app_default_reference_and_inclusive_cutoffs():
    trial = bayes_factor_binary_design(
        50, alternative_mode=0.4, imom_shape=1.25, inferiority_cutoff=0, strict_thresholds=False
    )
    assert_array_equal(trial.superiority_min, [6, 7, 9, 11, 12, 14, 15, 17, 18])
    assert_array_equal(trial.inferiority_max, -1)
    assert_allclose(trial.prior.tau, 0.0524, atol=0.00005, rtol=0)
    # Construct an exactly representable equality to distinguish the two policies.
    base = bayes_factor_binary_design(10, min_subjects=1, cohort_size=1)
    n, m, cutoff = next(
        (n, m, float(expit(base.log_bayes_factor[n, m])))
        for n in range(1, 11)
        for m in range(n + 1)
        if base.log_bayes_factor[n, m] == logit(expit(base.log_bayes_factor[n, m]))
        and 0.1 < expit(base.log_bayes_factor[n, m]) < 1
    )
    strict = bayes_factor_binary_design(
        10, min_subjects=1, cohort_size=1, superiority_cutoff=cutoff
    )
    inclusive = bayes_factor_binary_design(
        10, min_subjects=1, cohort_size=1, superiority_cutoff=cutoff, strict_thresholds=False
    )
    assert strict.monitor(m, n).decision in ("continue", "inconclusive")
    assert inclusive.monitor(m, n).decision == "superiority"
