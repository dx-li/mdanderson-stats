import numpy as np
import pytest

from mdanderson_stats.u2oet_decision import U2OETCriteria
from mdanderson_stats.u2oet_gao_fit import u2oet_gao_parameter_names
from mdanderson_stats.u2oet_gao_simulation import simulate_u2oet_gao_trial
from mdanderson_stats.u2oet_scenario import u2oet_scenario
from mdanderson_stats.u2oet_summary import summarize_u2oet_trials


def _inputs():
    scenario = u2oet_scenario(
        np.broadcast_to([0.5, 0.5], (2, 2, 2)),
        np.broadcast_to([0.5, 0.5], (2, 2, 2)),
        association=0.0,
    )
    names = u2oet_gao_parameter_names(2, 2)
    mean = np.zeros(len(names))
    return scenario, mean, np.zeros_like(mean)


def test_calendar_uses_available_toxicity_and_refits_only_when_counts_change():
    scenario, mean, sd = _inputs()
    uniforms = np.zeros((3, 4))
    uniforms[:, 2:] = 0.0
    trial = simulate_u2oet_gao_trial(
        [1, 2],
        [1, 2],
        scenario,
        [[10, 0], [100, 40]],
        prior_mean=mean,
        prior_sd=sd,
        initial=(0, 0),
        max_patients=3,
        cohort_size=2,
        efficacy_window=(1, 1),
        toxicity_window=(0.2, 0.2),
        arrival_times=[0, 0.5, 1.5],
        data_uniforms=uniforms,
        criteria=U2OETCriteria(1, 1, min_efficacy=0, max_toxicity=1),
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(41),
    )
    assert trial.decisions[0].complete_patients == 0
    assert trial.decisions[0].toxicity_only_patients == 1
    assert trial.decisions[0].ignored_patients == 0
    assert trial.decisions[1].complete_patients == 2
    assert trial.patients.complete.sum() == 3
    assert trial.posterior_fits == 3
    assert trial.likelihood_evaluations == 6
    assert trial.likelihood_work_units == 6 * scenario.joint.size
    assert trial.replay_only
    with pytest.raises(ValueError, match="not independent OC"):
        summarize_u2oet_trials([trial])


def test_same_original_seed_replays_generated_calendar_and_data():
    scenario, mean, sd = _inputs()
    kwargs = dict(
        doses1=[1, 2],
        doses2=[1, 2],
        scenario=scenario,
        utility=[[10, 0], [100, 40]],
        prior_mean=mean,
        prior_sd=sd,
        initial=(0, 0),
        max_patients=3,
        cohort_size=2,
        efficacy_window=(0.5, 1),
        toxicity_window=(0.1, 0.2),
        criteria=U2OETCriteria(1, 1, min_efficacy=0, max_toxicity=1),
        draws=8,
        warmup=0,
        chains=2,
    )
    one = simulate_u2oet_gao_trial(rng=np.random.default_rng(71), **kwargs)
    two = simulate_u2oet_gao_trial(rng=np.random.default_rng(71), **kwargs)
    np.testing.assert_array_equal(one.planned_arrival_times, two.planned_arrival_times)
    np.testing.assert_array_equal(one.data_uniforms, two.data_uniforms)
    np.testing.assert_array_equal(one.patients.records, two.patients.records)
    np.testing.assert_array_equal(
        one.final_posterior.mean_utility, two.final_posterior.mean_utility
    )
    assert one.data_seed == two.data_seed and one.posterior_seed == two.posterior_seed
    assert not one.replay_only


def test_global_budget_preflight_does_not_consume_rng():
    scenario, mean, sd = _inputs()
    sd[0] = 0.1
    rng = np.random.default_rng(14)
    state = rng.bit_generator.state
    with pytest.raises(ValueError, match="minimum GAO fit"):
        simulate_u2oet_gao_trial(
            [1, 2],
            [1, 2],
            scenario,
            [[10, 0], [100, 40]],
            prior_mean=mean,
            prior_sd=sd,
            initial=(0, 0),
            max_patients=2,
            criteria=U2OETCriteria(1, 1, min_efficacy=0, max_toxicity=1),
            draws=8,
            warmup=1,
            chains=2,
            max_likelihood_evaluations=3,
            rng=rng,
        )
    assert rng.bit_generator.state == state


def test_generated_positive_delay_that_rounds_away_is_rejected():
    scenario, mean, sd = _inputs()
    with pytest.raises(ArithmeticError, match="positive outcome delays"):
        simulate_u2oet_gao_trial(
            [1, 2],
            [1, 2],
            scenario,
            [[10, 0], [100, 40]],
            prior_mean=mean,
            prior_sd=sd,
            initial=(0, 0),
            max_patients=2,
            mean_interarrival=1e20,
            criteria=U2OETCriteria(1, 1, min_efficacy=0, max_toxicity=1),
            efficacy_window=(1, 1),
            toxicity_window=(1, 1),
            draws=8,
            warmup=0,
            chains=2,
            rng=np.random.default_rng(4),
        )
