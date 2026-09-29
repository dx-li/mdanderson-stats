import numpy as np
import pytest

from mdanderson_stats.multc_core import multc_lean_design
from mdanderson_stats.multc_simulation import (
    MultcSimulationConfig,
    _draw_response_delays,
    simulate_multc,
    simulate_multc_trial,
)


def _config(*, window=2.0, toxicity_delay=0.5, response_timing="clip_at_window"):
    design = multc_lean_design(
        4,
        (1.0, 1.0),
        (1.0, 1.0),
        historical_response=0.5,
        historical_toxicity=0.5,
        response_cutoff=1.0,
        toxicity_cutoff=1.0,
        pretrial_check=False,
    )
    return MultcSimulationConfig(design, window, toxicity_delay, response_timing)


def test_trial_preserves_paired_truth_and_explicit_endpoint_timing():
    config = _config()
    result = simulate_multc_trial(
        config,
        [0.0, 1.0, 0.0, 0.0],
        accrual_rate=2.0,
        seed=81,
    )

    np.testing.assert_array_equal(result.outcomes, [[1, 0]] * 4)
    assert result.arrival_times[0] == 0.0
    assert np.all(
        result.response_available_times - result.arrival_times <= config.response_window
    )
    np.testing.assert_array_equal(
        result.toxicity_available_times, result.arrival_times + config.toxicity_delay
    )
    assert result.decision == "cap_complete"


def test_aggregate_seeds_replay_and_single_trial_mcse_is_undefined():
    config = _config()
    aggregate = simulate_multc(
        config,
        [0.25, 0.25, 0.25, 0.25],
        accrual_rate=1.0,
        trials=1,
        seed=405,
    )
    replay = simulate_multc_trial(
        config,
        aggregate.joint_probabilities,
        accrual_rate=1.0,
        seed=int(aggregate.trial_seeds[0]),
    )

    assert aggregate.mean_enrolled == replay.enrolled
    assert aggregate.mean_responses == replay.responses
    assert aggregate.mean_toxicities == replay.toxicities
    assert np.isnan(aggregate.duration_mcse)
    assert np.isnan(aggregate.enrolled_mcse)
    assert aggregate.decisions == (replay.decision,)
    assert aggregate.decision_probability.tolist() == [1.0]


def test_fixed_enrollment_duration_matches_poisson_gap_expectation():
    config = _config(window=2.0, toxicity_delay=0.5)
    trials = 2_000
    rate = 2.0
    aggregate = simulate_multc(
        config,
        [0.0, 0.0, 0.0, 1.0],
        accrual_rate=rate,
        trials=trials,
        seed=2026,
    )

    expected_duration = (config.design.max_subjects - 1) / rate + config.response_window
    expected_mcse = np.sqrt((config.design.max_subjects - 1) / rate**2 / trials)
    assert abs(aggregate.mean_duration - expected_duration) < 5 * aggregate.duration_mcse
    assert aggregate.duration_mcse == pytest.approx(expected_mcse, rel=0.2)


def test_joint_category_probabilities_and_resource_preflight_are_validated():
    config = _config()
    with pytest.raises(ValueError, match="sum to one"):
        simulate_multc_trial(config, [0.2, 0.2, 0.2, 0.2], accrual_rate=1.0, seed=2)
    with pytest.raises(ValueError, match="worst-case estimate"):
        simulate_multc(
            config,
            [0.25] * 4,
            accrual_rate=1.0,
            trials=2,
            seed=2,
            max_total_work=1,
        )


def test_response_delay_sampling_scales_dimensionless_draws_safely():
    class FixedRng:
        def __init__(self, value):
            self.value = value

        def random(self, size):
            return np.full(size, self.value)

        def exponential(self, size):
            return np.full(size, self.value)

    for window in (1e-300, 1e300):
        delay = _draw_response_delays(
            np.random.default_rng(9),
            np.ones(8, dtype=np.int8),
            window,
            "conditional_truncated_exponential",
        )
        assert np.all(np.isfinite(delay))
        assert np.all(delay >= 0)
        assert np.all(delay <= window)

    smallest_window = np.nextafter(0.0, 1.0)
    zero_draw = _draw_response_delays(
        FixedRng(0.0), np.ones(1, dtype=np.int8), smallest_window,
        "conditional_truncated_exponential",
    )
    assert zero_draw[0] == 0.0
    with pytest.raises(ArithmeticError, match="below floating-point resolution"):
        _draw_response_delays(
            FixedRng(0.5), np.ones(1, dtype=np.int8), smallest_window,
            "conditional_truncated_exponential",
        )
