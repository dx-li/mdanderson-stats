"""Complete initializer references and proper prior/sampler handoff checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import fit_wfmm_coefficients, wfmm_basis, wfmm_summarize
from mdanderson_stats.wfmm_initialization import initialize_wfmm

CASES = json.loads(
    (Path(__file__).parent / "fixtures/wfmm-native-initialization-pipeline.json").read_text()
)["cases"]


def inputs(case):
    model = case["model"]
    counts = np.array(model.get("m", []), int)
    return dict(
        coefficients=case["output"]["D"],
        fixed_design=model["X"],
        random_design=model.get("Z"),
        random_strata=np.repeat(np.arange(len(counts)), counts) if counts.size else None,
        residual_strata=np.argmax(model["C"], axis=1),
        coefficient_partition=np.repeat(
            np.arange(len(case["partition_sizes"])), case["partition_sizes"]
        ),
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case"])
def test_source_complete_initializer_chain_and_numerical_difference_limits(case):
    settings = {b: case["settings"][a] for a, b in [("maxO", "max_odds")] if a in case["settings"]}
    if "nj_nosmooth" in case["settings"]:
        settings["unsmoothed_partitions"] = int(case["settings"]["nj_nosmooth"])
    result = initialize_wfmm(**inputs(case), profile="legacy_3_1", native_eb_settings=settings)
    original = case["output"]
    prior = result.prior
    np.testing.assert_allclose(
        result.initial_shrunken_coefficients, case["traces"][0]["beta"], atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(
        result.initial_shrinkage.group_inclusion_probability, original["pi_MOM"], atol=1e-10
    )
    np.testing.assert_allclose(
        result.initial_shrinkage.group_slab_to_sampling_variance, original["tau_MOM"], atol=1e-5
    )
    # The native optimizer stops at its own coarse tolerances. These are bounded
    # differences against recorded outputs, not binary/trajectory identity.
    variance = np.vstack((result.conditional.random_variance, result.conditional.residual_variance))
    np.testing.assert_allclose(variance, original["omega_MLE"], atol=0.0011, rtol=0)
    np.testing.assert_allclose(
        result.conditional_sampling_variances, original["Vbetastar_ns"], atol=0.00022, rtol=0
    )
    np.testing.assert_allclose(
        result.final_shrinkage.group_inclusion_probability, original["pi_MLE"], atol=1.1e-5, rtol=0
    )
    np.testing.assert_allclose(
        result.final_shrinkage.group_slab_to_sampling_variance,
        original["tau_MLE"],
        atol=0.006,
        rtol=0,
    )
    h = result.conditional.random_variance.shape[0]
    shape = np.vstack((prior.random_shape, prior.residual_shape)) if h else prior.residual_shape
    scale = np.vstack((prior.random_scale, prior.residual_scale)) if h else prior.residual_scale
    np.testing.assert_allclose(shape, original["prior_omega_a"], rtol=1e-14)
    np.testing.assert_allclose(scale, shape * variance, rtol=1e-14)
    assert result.ready_for_sampling == (not result.holds)
    if case["case"] == "floor_freeze":
        assert "native_floor_freeze_blocks_likelihood_improvement" in result.holds
    if case["case"] == "zero_signal":
        assert "initial_empirical_bayes_not_converged" in result.holds
        assert "final_empirical_bayes_not_converged" in result.holds


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case"])
def test_stable_priors_and_positive_scale_handoff(case):
    result = initialize_wfmm(**inputs(case))
    assert result.profile == "stable"
    assert not result.conditional.frozen.any()
    assert not result.conditional.artificial_lower.any()
    assert result.ready_for_sampling
    prior = result.prior
    assert np.all(prior.residual_shape > 0) and np.all(prior.residual_scale > 0)
    assert np.all(result.conditional.proposal_sd > 0)
    assert np.isfinite(result.conditional.proposal_sd).all()
    if case["case"] == "zero_signal":
        np.testing.assert_array_equal(prior.inclusion_probability, 0)
        np.testing.assert_array_equal(prior.slab_variance, 0)


def test_prior_sampler_reconstruction_workflow():
    case = CASES[2]
    kwargs = inputs(case)
    init = initialize_wfmm(**kwargs)
    assert init.ready_for_sampling
    fit = fit_wfmm_coefficients(
        **kwargs,
        prior=init.prior,
        random_variance=init.conditional.random_variance,
        residual_variance=init.conditional.residual_variance,
        proposal_sd=(init.conditional.proposal_sd[:1], init.conditional.proposal_sd[1:]),
        draws=32,
        warmup=16,
        chains=2,
        rng=np.random.default_rng(177),
    )
    summary = wfmm_summarize(fit.coefficients, wfmm_basis(4, transform="identity"))
    assert summary.mean.shape == (1, 4)
    assert np.isfinite(summary.mean).all()
    assert np.all(fit.random_variances > 0) and np.all(fit.residual_variances > 0)
    # This is a workflow smoke check, not a posterior precision/clinical claim.


def test_holds_for_inadequate_iteration_budgets():
    result = initialize_wfmm(
        **inputs(CASES[1]),
        moment_iterations=1,
        optimizer_iterations=1,
        empirical_bayes_iterations=1,
    )
    assert not result.ready_for_sampling
    assert "moment_not_converged" in result.holds
    assert "initial_empirical_bayes_not_converged" in result.holds
    assert "conditional_optimizer_not_converged" in result.holds


@pytest.mark.parametrize(
    "kwargs",
    [
        {"profile": "other"},
        {"native_eb_settings": {"min_ratio": 10}},
        {"profile": "legacy_3_1", "native_eb_settings": {"unknown": 1}},
        {"profile": "legacy_3_1", "native_eb_settings": {"unsmoothed_partitions": True}},
        {"delta_omega": 0},
        {"delta_omega": True},
        {"empirical_bayes_iterations": False},
    ],
)
def test_invalid_controls_rejected(kwargs):
    with pytest.raises(ValueError):
        initialize_wfmm(**inputs(CASES[0]), **kwargs)
