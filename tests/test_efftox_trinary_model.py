import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import truncnorm

from mdanderson_stats.efftox_model import efftox_standardize
from mdanderson_stats.efftox_trinary_model import (
    EffToxTrinaryPrior,
    efftox_trinary_log_likelihood,
    efftox_trinary_log_probabilities,
    efftox_trinary_predict,
    fit_efftox_trinary,
)


def test_continuation_ratio_probabilities_and_likelihood_match_independent_values():
    doses = np.asarray([1.0, 2.0, 4.0])
    codes = efftox_standardize(doses)
    parameters = np.asarray([-1.0, 0.6, 0.4, 0.9])
    counts = np.asarray([[2, 2, 1], [1, 3, 1], [1, 2, 2]])
    expected = np.asarray(
        [
            [0.447188554002106, 0.357504469517202, 0.195306976480692],
            [0.293382828784872, 0.437675749845133, 0.268941421369995],
            [0.169672183094312, 0.472340773662764, 0.357987043242924],
        ]
    )

    probabilities = efftox_trinary_predict(codes, parameters)
    np.testing.assert_allclose(probabilities, expected, rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=2e-15)
    assert float(efftox_trinary_log_likelihood(codes, counts, parameters)) == pytest.approx(
        -15.6468300616634, abs=5e-14
    )
    logs = efftox_trinary_log_probabilities(codes, parameters)
    np.testing.assert_allclose(np.exp(logs), probabilities, rtol=0, atol=0)


def test_posterior_retains_marginal_and_conditional_efficacy_with_reference_check():
    doses = np.asarray([1.0, 2.0, 4.0])
    counts = np.asarray([[2, 2, 1], [1, 3, 1], [1, 2, 2]])
    prior = EffToxTrinaryPrior(mean=[-1.0, 0.6, 0.4, 0.9], sd=[0.7, 0.0, 0.8, 0.0])
    fit = fit_efftox_trinary(
        doses,
        counts,
        prior=prior,
        draws=4000,
        warmup=500,
        chains=4,
        rng=np.random.default_rng(17),
    )

    reference_efficacy = np.asarray([0.390690854235906, 0.461402779806015, 0.48837341746276])
    reference_toxicity = np.asarray([0.197303666071384, 0.268615800809073, 0.353737796322606])
    np.testing.assert_allclose(
        fit.efficacy_probabilities.mean(axis=(0, 1)), reference_efficacy, atol=0.004
    )
    np.testing.assert_allclose(
        fit.toxicity_probabilities.mean(axis=(0, 1)), reference_toxicity, atol=0.004
    )
    marginal_from_cells = fit.joint_probabilities[..., 1]
    conditional = fit.conditional_efficacy_probabilities
    np.testing.assert_allclose(fit.efficacy_probabilities, marginal_from_cells, atol=0, rtol=0)
    np.testing.assert_allclose(
        conditional,
        expit(fit.conditional_efficacy_logits),
        atol=2e-15,
    )
    np.testing.assert_allclose(
        fit.efficacy_probabilities,
        (1.0 - fit.toxicity_probabilities) * conditional,
        atol=2e-15,
    )
    slopes = fit.parameters[..., (1, 3)]
    np.testing.assert_array_equal(slopes, np.broadcast_to([0.6, 0.9], slopes.shape))
    assert fit.likelihood_evaluations > 0
    assert fit.summary.mean.shape == (4,)


def test_fixed_positive_slopes_and_prior_only_sampling_are_supported():
    prior = EffToxTrinaryPrior(mean=[-0.5, 0.7, 0.25, 0.8], sd=[0.0, 0.0, 0.0, 0.0])
    fit = fit_efftox_trinary(
        [0.0, 1.0],
        np.zeros((2, 3)),
        prior=prior,
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(5),
    )
    np.testing.assert_array_equal(fit.parameters, np.broadcast_to(prior.mean, (2, 8, 4)))
    np.testing.assert_allclose(fit.joint_probabilities.sum(axis=-1), 1.0, atol=2e-15)
    assert fit.likelihood_evaluations == 0
    with pytest.raises(ValueError, match="fixed monotone"):
        EffToxTrinaryPrior(mean=[0.0, 0.0, 0.0, 1.0], sd=[1.0, 0.0, 1.0, 1.0])


def test_nonempty_data_preserves_positive_truncated_slope_priors():
    prior = EffToxTrinaryPrior(mean=[-0.7, 0.4, 0.2, -0.2], sd=[0.5, 0.8, 0.6, 0.6])
    fit = fit_efftox_trinary(
        [1.0, 2.0, 4.0],
        [[0, 0, 0], [4, 4, 4], [0, 0, 0]],
        prior=prior,
        draws=1200,
        warmup=400,
        chains=2,
        rng=np.random.default_rng(23),
    )
    for index in (1, 3):
        standardized_lower = -prior.mean[index] / prior.sd[index]
        expected = truncnorm.mean(
            standardized_lower,
            np.inf,
            loc=prior.mean[index],
            scale=prior.sd[index],
        )
        assert np.all(fit.parameters[..., index] > 0)
        assert fit.parameters[..., index].mean() == pytest.approx(expected, abs=0.09)


def test_extreme_logits_zero_cells_and_sampling_preflight_are_stable():
    dose_codes = np.asarray([-0.5, 0.5])
    extreme = np.asarray([100.0, 1.0, -100.0, 1.0])
    logs = efftox_trinary_log_probabilities(dose_codes, extreme)
    assert np.all(np.isfinite(logs))
    counts = np.asarray([[0, 0, 1], [0, 0, 1]])
    assert np.isfinite(efftox_trinary_log_likelihood(dose_codes, counts, extreme))
    assert np.all(np.isfinite(efftox_trinary_predict(dose_codes, extreme)))

    prior = EffToxTrinaryPrior(mean=[-1.0, 0.5, 0.5, 0.5], sd=[1.0, 0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="retained trinary probability draws"):
        fit_efftox_trinary(
            np.arange(1.0, 21.0),
            np.zeros((20, 3)),
            prior=prior,
            draws=10_000,
            warmup=0,
            chains=4,
            rng=np.random.default_rng(1),
        )
