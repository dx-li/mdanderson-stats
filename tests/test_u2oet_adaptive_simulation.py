import json

import numpy as np
import pytest

import mdanderson_stats.u2oet_simulation as simulation
from mdanderson_stats.u2oet_decision import U2OETCriteria
from mdanderson_stats.u2oet_fit import u2oet_parameter_names
from mdanderson_stats.u2oet_scenario import u2oet_scenario


def _setup():
    names = u2oet_parameter_names(2, 2, model="cmi")
    mean = np.zeros(len(names) - 1)
    sd = np.full(mean.size, 1e-10)
    for i, name in enumerate(names[:-1]):
        if ".slope." in name:
            mean[i] = 1
    mean[names.index("toxicity.intercept.1")] = -1000
    sd[0] = 1
    scenario = u2oet_scenario(
        np.broadcast_to([0, 1], (2, 2, 2)),
        np.broadcast_to([1, 0], (2, 2, 2)),
    )
    return names, mean, sd, scenario


def _settings():
    return simulation.U2OETAdaptiveSettings(
        target_mcse_ratio=0.05,
        initial_draws=512,
        max_draws_per_chain=4096,
        batch_draws=512,
        max_total_work=2_000_000_000,
    )


def test_calendar_reuses_adaptive_fit_and_records_compact_precision():
    _, mean, sd, scenario = _setup()
    trial = simulation.simulate_u2oet_trial(
        [1, 2],
        [1, 2],
        scenario,
        [[0, 0], [1, 1]],
        prior_mean=mean,
        prior_sd=sd,
        initial=(0, 0),
        criteria=U2OETCriteria(1, 1, min_efficacy=0, max_toxicity=1),
        max_patients=4,
        cohort_size=4,
        efficacy_window=(1000, 1000),
        toxicity_window=(1000, 1000),
        mean_interarrival=1,
        draws=100_000,
        warmup=64,
        chains=2,
        coordinate_updates=False,
        model="cmi",
        adaptive_precision=_settings(),
        rng=np.random.default_rng(8801),
    )
    assert trial.posterior_fits == 2
    assert trial.decisions
    assert all(d.precision_target_met is True for d in trial.decisions)
    assert all(d.precision_draws_per_chain >= 512 for d in trial.decisions)
    for decision in trial.decisions:
        assert decision.corner_mcse_ratio.shape == (2, 4)
        assert not decision.corner_mcse_ratio.flags.writeable
        assert np.all(decision.corner_mcse_ratio <= 0.05)
    assert trial.final_precision_target_met is True
    assert trial.final_precision_draws_per_chain >= 512
    assert trial.final_corner_mcse_ratio.shape == (2, 4)
    assert np.all(trial.final_corner_mcse_ratio <= 0.05)
    design = json.loads(trial.design_json)
    assert design["adaptive_precision"] == {
        "target_mcse_ratio": 0.05,
        "initial_draws": 512,
        "max_draws_per_chain": 4096,
        "batch_draws": 512,
        "max_total_work": 2_000_000_000,
    }


def test_calendar_fails_closed_when_real_adaptive_fit_misses_cap():
    _, mean, sd, scenario = _setup()
    settings = simulation.U2OETAdaptiveSettings(
        target_mcse_ratio=0.001,
        initial_draws=8,
        max_draws_per_chain=8,
        batch_draws=8,
        max_total_work=2_000_000,
    )
    with pytest.raises(ArithmeticError, match="no decision was made"):
        simulation.simulate_u2oet_trial(
            [1, 2],
            [1, 2],
            scenario,
            [[10, 0], [100, 40]],
            prior_mean=mean,
            prior_sd=sd,
            initial=(0, 0),
            criteria=U2OETCriteria(1, 1, min_efficacy=0, max_toxicity=1),
            max_patients=2,
            model="cmi",
            efficacy_window=(1000, 1000),
            toxicity_window=(1000, 1000),
            mean_interarrival=1,
            warmup=0,
            chains=2,
            coordinate_updates=False,
            adaptive_precision=settings,
            rng=np.random.default_rng(8802),
        )


def test_trial_work_rejection_precedes_rng_split():
    _, mean, sd, scenario = _setup()
    settings = simulation.U2OETAdaptiveSettings(
        target_mcse_ratio=0.02,
        initial_draws=8,
        max_draws_per_chain=8,
        max_total_work=1,
    )
    rng = np.random.default_rng(8803)
    state = rng.bit_generator.state
    with pytest.raises(ValueError, match="max_total_work"):
        simulation.simulate_u2oet_trial(
            [1, 2],
            [1, 2],
            scenario,
            [[10, 0], [100, 40]],
            prior_mean=mean,
            prior_sd=sd,
            initial=(0, 0),
            criteria=U2OETCriteria(1, 1, min_efficacy=0, max_toxicity=1),
            max_patients=2,
            model="cmi",
            warmup=0,
            chains=2,
            coordinate_updates=False,
            adaptive_precision=settings,
            rng=rng,
        )
    assert rng.bit_generator.state == state


def test_adaptive_warmup_limit_is_checked_before_rng_split():
    _, mean, sd, scenario = _setup()
    settings = simulation.U2OETAdaptiveSettings(
        target_mcse_ratio=0.02,
        initial_draws=10_001,
        max_draws_per_chain=10_001,
    )
    rng = np.random.default_rng(8804)
    state = rng.bit_generator.state
    with pytest.raises(ValueError, match="warmup must not exceed 10000"):
        simulation.simulate_u2oet_trial(
            [1, 2],
            [1, 2],
            scenario,
            [[10, 0], [100, 40]],
            prior_mean=mean,
            prior_sd=sd,
            initial=(0, 0),
            criteria=U2OETCriteria(1, 1, min_efficacy=0, max_toxicity=1),
            max_patients=2,
            draws=100_000,
            warmup=10_001,
            chains=2,
            model="cmi",
            adaptive_precision=settings,
            rng=rng,
        )
    assert rng.bit_generator.state == state
