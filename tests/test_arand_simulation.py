import numpy as np
import pytest

from mdanderson_stats.arand_calendar import ArandControllerPolicy
from mdanderson_stats.arand_simulation import (
    ArandSimulationConfig,
    simulate_arand,
    simulate_arand_trial,
)


def _policy(precedence="duration_wins"):
    return ArandControllerPolicy(
        floor_transform="mixture",
        ranking_scope="all_arms",
        trigger_order=("futility", "suspension", "early_winner"),
        duration_minimum_precedence=precedence,
        multiple_winner="lowest_arm_index",
        all_suspended_action="stop",
        same_time_order="arrival_before_analysis",
    )


def _config(*, max_n=1, duration=None, minimum=1, precedence="duration_wins"):
    return ArandSimulationConfig(
        prior=[[1.0, 1.0]],
        family="binary",
        analysis_times=[],
        max_enrollment=max_n,
        max_duration=duration,
        final_followup=0.0,
        minimum_enrollment=minimum,
        policy=_policy(precedence),
        binary_window=1.0,
    )


def test_single_arm_aggregate_matches_replayed_seeds_and_one_trial_mcse_is_undefined():
    config = _config()
    aggregate = simulate_arand(
        config, [0.5], accrual_rate=2.0, max_candidate_arrivals=1,
        trials=1, seed=902,
    )
    replay = simulate_arand_trial(
        config, aggregate.scenario_parameters, accrual_rate=2.0,
        max_candidate_arrivals=1, seed=int(aggregate.trial_seeds[0]),
    )

    assert aggregate.final_selected_count.tolist() == [int(replay.final_winner == 0)]
    assert aggregate.no_winner_count == int(replay.final_winner is None)
    assert aggregate.mean_patients_per_arm.tolist() == [1.0]
    assert aggregate.mean_decision_duration == replay.decision_duration
    assert np.isnan(aggregate.selected_mcse[0])
    assert np.isnan(aggregate.decision_duration_mcse)


def test_candidate_cap_cannot_silently_truncate_duration_trials():
    for precedence in ("duration_wins", "minimum_enrollment_wins"):
        config = _config(
            max_n=None,
            duration=10_000.0,
            minimum=1,
            precedence=precedence,
        )
        with pytest.raises(RuntimeError, match="candidate-arrival budget exhausted"):
            simulate_arand_trial(
                config, [0.5], accrual_rate=10.0,
                max_candidate_arrivals=2, seed=37,
            )
