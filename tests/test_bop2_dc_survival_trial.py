import copy

import numpy as np
import pytest

from mdanderson_stats.bop2_dc_survival import bop2_dc_survival_design
from mdanderson_stats.bop2_dc_survival_trial import (
    run_bop2_dc_survival_trial,
    simulate_bop2_dc_survival,
)


def test_replay_uses_as_of_censoring_and_relative_final_followup():
    design = bop2_dc_survival_design(
        4,
        lrv=1.0,
        cmv=2.0,
        lambda_lrv=1e-12,
        lambda_cmv=1e-12,
        prior_shape=2.0,
        prior_scale=1.0,
        looks=[2, 4],
    )
    replay = run_bop2_dc_survival_trial(
        design,
        [1e12, 1e12 + 0.5, 1e12 + 1.0, 1e12 + 1.5],
        [np.inf, np.inf, 0.1, np.inf],
        final_followup=0.25,
    )

    assert replay.enrolled == 4
    assert replay.events == 1
    assert replay.calendar_times.tolist() == [1e12 + 0.5, 1e12 + 1.75]
    assert replay.states[0].total_time.item() == 0.5
    assert replay.total_time == pytest.approx(3.35)
    assert replay.duration == 1e12 + 1.75
    assert replay.decision == "final_go"
    assert replay.decision == replay.states[-1].decision.item()


def test_first_interim_futility_stops_replay_at_that_look():
    design = bop2_dc_survival_design(
        4,
        lrv=1.0,
        cmv=2.0,
        lambda_lrv=0.99,
        lambda_cmv=0.99,
        gamma_lrv=0.0,
        gamma_cmv=0.0,
        prior_shape=2.0,
        prior_scale=1.0,
        looks=[2, 4],
    )
    replay = run_bop2_dc_survival_trial(
        design, [0.0, 1.0, 2.0, 3.0], [np.inf] * 4, final_followup=1.0
    )

    assert replay.enrolled == 2
    assert replay.decision == "stop_no_go"
    assert replay.calendar_times.tolist() == [1.0]
    assert replay.total_time == 1.0


def test_simulation_is_replayable_and_summaries_conserve_trials():
    design = bop2_dc_survival_design(
        4,
        lrv=1.0,
        cmv=2.0,
        lambda_lrv=0.6,
        lambda_cmv=0.3,
        prior_shape=1.0,
        prior_scale=1.0,
        looks=[2, 4],
    )
    first = simulate_bop2_dc_survival(
        design,
        3.0,
        accrual_rate=1.0,
        final_followup=2.0,
        n_trials=24,
        arrival="poisson",
        rng=178,
    )
    replay = simulate_bop2_dc_survival(
        design,
        3.0,
        accrual_rate=1.0,
        final_followup=2.0,
        n_trials=24,
        arrival="poisson",
        rng=first.rng_seed,
    )

    assert int(first.decision_count.sum()) == first.trials
    assert first.decision_probability.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(first.decision_probability, replay.decision_probability)
    np.testing.assert_array_equal(first.decision, replay.decision)
    np.testing.assert_array_equal(first.sample_size, replay.sample_size)
    np.testing.assert_allclose(first.total_time, replay.total_time)
    np.testing.assert_allclose(
        first.decision_mcse,
        np.sqrt(first.decision_probability * (1 - first.decision_probability) / first.trials),
    )


def test_path_budget_fails_before_advancing_supplied_generator():
    design = bop2_dc_survival_design(1000, lrv=1, cmv=2, looks=[1000])
    generator = np.random.default_rng(91)
    state_before = copy.deepcopy(generator.bit_generator.state)

    with pytest.raises(ValueError, match="path cell budget"):
        simulate_bop2_dc_survival(
            design,
            3.0,
            accrual_rate=1.0,
            final_followup=1.0,
            n_trials=2000,
            rng=generator,
        )
    assert generator.bit_generator.state == state_before
