import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.u2oet_decision import U2OETCriteria
from mdanderson_stats.u2oet_gao_adaptive_precision import _adaptive_resource_plan
from mdanderson_stats.u2oet_gao_fit import u2oet_gao_parameter_names
from mdanderson_stats.u2oet_gao_simulation import simulate_u2oet_gao_trial
from mdanderson_stats.u2oet_scenario import u2oet_scenario
from mdanderson_stats.u2oet_simulation import U2OETAdaptiveSettings


def _inputs(*, free: bool = False):
    scenario = u2oet_scenario(
        np.broadcast_to([0.5, 0.5], (2, 2, 2)),
        np.broadcast_to([0.5, 0.5], (2, 2, 2)),
        association=0.0,
    )
    mean = np.zeros(len(u2oet_gao_parameter_names(2, 2)))
    sd = np.zeros_like(mean)
    if free:
        sd[0] = 0.2
    return scenario, mean, sd


def _trial_kwargs(scenario, mean, sd):
    return dict(
        doses1=[1, 2],
        doses2=[1, 2],
        scenario=scenario,
        utility=[[10, 0], [100, 40]],
        prior_mean=mean,
        prior_sd=sd,
        initial=(0, 0),
        criteria=U2OETCriteria(1, 1, min_efficacy=0, max_toxicity=1),
        max_patients=3,
        cohort_size=2,
        efficacy_window=(5, 5),
        toxicity_window=(5, 5),
        arrival_times=[0, 1, 2],
        data_uniforms=np.zeros((3, 4)),
        draws=8,
        warmup=0,
        chains=2,
    )


def test_adaptive_gao_trial_caches_pending_snapshot_and_refits_final_counts():
    scenario, mean, sd = _inputs(free=True)
    trial = simulate_u2oet_gao_trial(
        **_trial_kwargs(scenario, mean, sd),
        adaptive_precision=U2OETAdaptiveSettings(
            target_mcse_ratio=0.05,
            initial_draws=512,
            max_draws_per_chain=4096,
            batch_draws=512,
            max_total_work=100_000_000,
        ),
        rng=np.random.default_rng(601),
    )
    assert trial.posterior_fits == 2
    assert trial.decisions[0].precision_target_met is True
    assert trial.decisions[1].precision_target_met is True
    assert (
        trial.decisions[0].precision_draws_per_chain == trial.decisions[1].precision_draws_per_chain
    )
    assert_allclose(trial.decisions[0].corner_mcse_ratio, trial.decisions[1].corner_mcse_ratio)
    assert trial.final_precision_target_met is True
    assert trial.final_precision_draws_per_chain is not None
    assert np.all(trial.final_corner_mcse_ratio <= 0.05)
    assert not trial.final_corner_mcse_ratio.flags.writeable
    assert trial.patients.complete.sum() == 3
    assert trial.patients.toxicity_only.sum() == 0
    assert trial.likelihood_evaluations > 0 and trial.likelihood_work_units > 0
    assert '"adaptive_precision"' in trial.design_json
    assert not trial.decisions[0].corner_mcse_ratio.flags.writeable


def test_adaptive_gao_trial_rejects_unmet_target_before_decision():
    scenario, mean, sd = _inputs()
    kwargs = _trial_kwargs(scenario, mean, sd)
    kwargs["max_patients"] = 1
    kwargs["data_uniforms"] = np.zeros((1, 4))
    kwargs["arrival_times"] = [0]
    with pytest.raises(ArithmeticError, match="no decision was made"):
        simulate_u2oet_gao_trial(
            **kwargs,
            adaptive_precision=U2OETAdaptiveSettings(
                target_mcse_ratio=0.01,
                initial_draws=8,
                max_draws_per_chain=8,
                batch_draws=8,
                max_total_work=100_000,
            ),
            rng=np.random.default_rng(602),
        )


def test_adaptive_gao_trial_minimum_budget_rejection_preserves_rng():
    scenario, mean, sd = _inputs(free=True)
    rng = np.random.default_rng(603)
    state = rng.bit_generator.state
    with pytest.raises(ValueError, match="minimum adaptive GAO trial evaluations"):
        simulate_u2oet_gao_trial(
            **_trial_kwargs(scenario, mean, sd),
            max_likelihood_evaluations=779,
            adaptive_precision=U2OETAdaptiveSettings(
                target_mcse_ratio=0.05,
                initial_draws=128,
                max_draws_per_chain=128,
                batch_draws=32,
                max_total_work=100_000,
            ),
            rng=rng,
        )
    assert rng.bit_generator.state == state


def test_adaptive_gao_trial_runtime_budget_is_cumulative():
    scenario, mean, sd = _inputs(free=True)
    kwargs = _trial_kwargs(scenario, mean, sd)
    kwargs["max_patients"] = 2
    kwargs["data_uniforms"] = np.zeros((2, 4))
    kwargs["arrival_times"] = [0, 1]
    kwargs["efficacy_window"] = (5, 5)
    kwargs["toxicity_window"] = (5, 5)
    plan = _adaptive_resource_plan(
        joint_shape=scenario.joint.shape,
        dimension=mean.size,
        chains=2,
        free=True,
        warmup=0,
        initial_draws=4096,
        max_draws_per_chain=4096,
        batch_draws=512,
    )
    with pytest.raises(ArithmeticError, match="budget exhausted"):
        simulate_u2oet_gao_trial(
            **kwargs,
            max_likelihood_evaluations=plan.minimum_evaluations * 2,
            max_work=plan.minimum_work * 2,
            adaptive_precision=U2OETAdaptiveSettings(
                target_mcse_ratio=0.05,
                initial_draws=4096,
                max_draws_per_chain=4096,
                batch_draws=512,
                max_total_work=plan.minimum_work * 2,
            ),
            rng=np.random.default_rng(604),
        )


def test_none_and_omitted_adaptive_mode_keep_exact_fixed_result():
    scenario, mean, sd = _inputs()
    kwargs = _trial_kwargs(scenario, mean, sd)
    implicit = simulate_u2oet_gao_trial(rng=np.random.default_rng(605), **kwargs)
    explicit = simulate_u2oet_gao_trial(
        rng=np.random.default_rng(605), adaptive_precision=None, **kwargs
    )
    assert implicit.design_json == explicit.design_json
    assert implicit.likelihood_evaluations == explicit.likelihood_evaluations
    assert implicit.likelihood_work_units == explicit.likelihood_work_units
    assert implicit.posterior_fits == explicit.posterior_fits
    assert_allclose(implicit.patients.records, explicit.patients.records)
    assert_allclose(implicit.final_posterior.mean_utility, explicit.final_posterior.mean_utility)
