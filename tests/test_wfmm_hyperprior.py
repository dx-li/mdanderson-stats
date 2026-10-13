"""Independent conjugacy, marginalization, replay and saved-study checks."""

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import betaln
from scipy.stats import beta as beta_distribution
from scipy.stats import invgamma, norm

from mdanderson_stats import (
    WFMMPrior,
    WFMMShrinkageHyperprior,
    WFMMStudy,
    fit_wfmm_coefficients,
    read_wfmm_study,
    run_wfmm_study,
    wfmm_mode_shrinkage_hyperprior,
    wfmm_shrinkage_conditional,
)


def fit(hyper=None, seed=21, **kwargs):
    options = dict(
        coefficients=np.tile([1.5, -0.2], (4, 1)),
        fixed_design=np.ones((4, 1)),
        prior=WFMMPrior(0.4, 1),
        residual_variance=1,
        estimate_variances=False,
        draws=16,
        warmup=5,
    )
    options.update(kwargs)
    return fit_wfmm_coefficients(
        **options, shrinkage_hyperprior=hyper, rng=np.random.default_rng(seed)
    )


def test_exact_partition_conditional_and_included_zero():
    prior = WFMMShrinkageHyperprior([2, 3], [5, 7], 4, 6)
    out = wfmm_shrinkage_conditional(
        [[2, 0, 0, -3]],
        [[True, True, False, True]],
        hyperprior=prior,
        coefficient_partition=[9, 2, 9, 2],
    )
    np.testing.assert_array_equal(out.partition_labels, [2, 9])
    np.testing.assert_array_equal(out.probability_shape_a, [[4, 4]])
    np.testing.assert_array_equal(out.probability_shape_b, [[5, 8]])
    np.testing.assert_array_equal(out.slab_shape, [[5, 4.5]])
    np.testing.assert_array_equal(out.slab_scale, [[10.5, 8]])
    assert not out.slab_scale.flags.writeable
    # Do not lose a small positive prior b through subtractive cancellation.
    tiny = wfmm_shrinkage_conditional(
        [[0]], [[True]], hyperprior=WFMMShrinkageHyperprior(1, 1e-30, 1, 1)
    )
    assert tiny.probability_shape_b.item() == 1e-30


@pytest.mark.parametrize("mode", [0, 0.01, 0.2, 0.5, 0.99, 1])
@pytest.mark.parametrize("variance", [0.01, 0.06, 0.08])
def test_independent_mode_and_variance_equations(mode, variance):
    p = wfmm_mode_shrinkage_hyperprior(mode, [0.1, 1, 10], probability_variance=variance)
    a, b = p.probability_shape_a, p.probability_shape_b
    np.testing.assert_allclose((a - 1) / (a + b - 2), mode, atol=1e-14)
    np.testing.assert_allclose(beta_distribution.var(a, b), variance, rtol=1e-10)
    np.testing.assert_allclose(p.slab_scale / (p.slab_shape + 1), [0.1, 1, 10])
    np.testing.assert_allclose(invgamma.var(p.slab_shape, scale=p.slab_scale), 1000, rtol=1e-7)
    assert np.all(p.slab_shape > 2)


@pytest.mark.parametrize(
    "field,value",
    [("probability_shape_a", 0), ("slab_scale", np.inf), ("slab_shape", True), ("update_mask", 1)],
)
def test_prior_validation(field, value):
    args = dict(probability_shape_a=2, probability_shape_b=3, slab_shape=4, slab_scale=3)
    args[field] = value
    with pytest.raises(ValueError):
        WFMMShrinkageHyperprior(**args)


def test_masked_updates_preserve_exact_fixed_chain_and_rng():
    prior = WFMMShrinkageHyperprior(2, 3, 4, 3, False)
    a, b = fit(), fit(prior)
    for name in ("coefficients", "inclusion_indicators", "log_likelihood"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    np.testing.assert_array_equal(b.inclusion_probabilities, 0.4)
    np.testing.assert_array_equal(b.slab_variances, 1)
    assert a.inclusion_probabilities is None


def test_partition_masks_and_thinning_exact_stream():
    prior = WFMMShrinkageHyperprior(2, 3, 4, 3, [False, True])
    kwargs = dict(coefficient_partition=[2, 9])
    dense = fit(prior, draws=48, **kwargs)
    sparse = fit(prior, thin=3, **kwargs)
    for name in (
        "coefficients",
        "inclusion_indicators",
        "inclusion_probabilities",
        "slab_variances",
    ):
        np.testing.assert_array_equal(getattr(sparse, name), getattr(dense, name)[:, 2::3])
    np.testing.assert_array_equal(sparse.inclusion_probabilities[..., 0], 0.4)
    np.testing.assert_array_equal(sparse.slab_variances[..., 0], 1)
    assert np.std(sparse.inclusion_probabilities[..., 1]) > 0.05
    assert not sparse.slab_variances.flags.writeable


def test_independently_integrated_two_coefficient_posterior():
    # Enumerate the four indicator patterns and integrate the shared slab variance.
    patterns = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    weights, tau_numerators = [], []
    for pattern in patterns:
        count = sum(pattern)

        def density(t):
            return invgamma.pdf(t, 4, scale=3) * np.prod(
                norm.pdf([1.5, -0.2], scale=np.sqrt(0.25 + pattern * t))
            )

        prior_weight = np.exp(betaln(2 + count, 3 + 2 - count) - betaln(2, 3))
        weights.append(prior_weight * quad(density, 0, np.inf, epsabs=1e-11)[0])
        tau_numerators.append(
            prior_weight * quad(lambda t: t * density(t), 0, np.inf, epsabs=1e-11)[0]
        )
    total = sum(weights)
    probabilities = np.array(weights) / total
    expected_gamma = probabilities @ patterns
    expected_pi = probabilities @ ((2 + patterns.sum(axis=1)) / 7)
    expected_tau = sum(tau_numerators) / total
    result = fit(WFMMShrinkageHyperprior(2, 3, 4, 3), draws=4000, warmup=800)
    np.testing.assert_allclose(
        result.inclusion_indicators.mean(axis=(0, 1))[0], expected_gamma, atol=0.035
    )
    assert abs(result.inclusion_probabilities.mean() - expected_pi) < 0.025
    assert abs(result.slab_variances.mean() / expected_tau - 1) < 0.08


def test_saved_study_replay_and_historical_default():
    y = np.array(
        [
            [1, 4, 2, 3],
            [2, 2, 3, 1],
            [4, 3, 1, 5],
            [5, 6, 4, 2],
            [6, 3, 7, 4],
            [3, 1, 2, 6],
            [2, 5, 5, 3],
            [4, 2, 6, 2],
        ],
        dtype=float,
    )
    study = WFMMStudy(
        y,
        np.ones((8, 1)),
        draws=8,
        warmup=3,
        shrinkage_hyperprior=WFMMShrinkageHyperprior(2, 3, 4, 3),
    )
    restored = read_wfmm_study(study.to_json())
    a, b = run_wfmm_study(study), run_wfmm_study(restored)
    assert a.report_json() == b.report_json()
    report = json.loads(a.report_json())["shrinkage_hyperparameters"]
    assert report["prior_parameterization"] == "explicit_beta_and_inverse_gamma_shape_scale"
    np.testing.assert_allclose(
        report["pi"]["mean"], a.fit.inclusion_probabilities.mean(axis=(0, 1))
    )
    old = json.loads(replace(study, shrinkage_hyperprior=None).to_json())
    del old["shrinkage_hyperprior"]
    assert read_wfmm_study(json.dumps(old)).shrinkage_hyperprior is None
    bad = json.loads(study.to_json())
    bad["shrinkage_hyperprior"]["unknown"] = 3
    with pytest.raises(ValueError):
        read_wfmm_study(json.dumps(bad))


def test_nonconstant_starting_partition_rejected():
    with pytest.raises(ValueError, match="constant"):
        fit(WFMMShrinkageHyperprior(2, 3, 4, 3), prior=WFMMPrior([[0.2, 0.4]], 1))


def test_uninformative_design_recovers_prior_marginals():
    result = fit(
        WFMMShrinkageHyperprior(2, 3, 4, 6), fixed_design=np.zeros((4, 1)), draws=4000, warmup=500
    )
    assert abs(result.inclusion_probabilities.mean() - 0.4) < 0.025
    assert abs(np.mean(1 / result.slab_variances) - 4 / 6) < 0.035
    assert abs(result.inclusion_indicators.mean() - 0.4) < 0.035


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(probability_variance=1 / 12),
        dict(probability_variance=0),
        dict(slab_variance=-1),
        dict(slab_mode=0),
    ],
)
def test_mode_calibration_domain(kwargs):
    args = dict(probability_mode=0.2, slab_mode=1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        wfmm_mode_shrinkage_hyperprior(**args)


def test_all_spike_start_can_leave_boundary():
    result = fit(WFMMShrinkageHyperprior(2, 3, 4, 3), prior=WFMMPrior(0, 0), draws=100)
    assert np.any(result.inclusion_indicators)
    assert np.all(result.slab_variances > 0)


def test_thinned_random_effects_work_budget(monkeypatch):
    import mdanderson_stats.wfmm_model as model

    # Main marginal likelihood is small (N=2); sampling 20 random levels dominates.
    monkeypatch.setattr(model, "_MAX_WORK", 500_000)
    with pytest.raises(ValueError, match="random-effect posterior"):
        fit(
            coefficients=np.ones((2, 1)),
            fixed_design=np.ones((2, 1)),
            random_design=np.ones((2, 20)),
            random_variance=1,
            sample_random_effects=True,
            thin=5,
        )
