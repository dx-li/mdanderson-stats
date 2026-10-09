import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import betainc

from mdanderson_stats.multc99 import Multc99Event, multc99_design
from mdanderson_stats.multc99_calibration import (
    multc99_precision_sample_size,
    multc99_prior_from_interval,
)
from mdanderson_stats.multc99_randomized import (
    run_multc99_randomized_trial,
    simulate_multc99_randomized,
)

_NATIVE = json.loads((Path(__file__).parent / "fixtures/multc99-native-reference.json").read_text())


@pytest.mark.parametrize("case", _NATIVE["calibration"])
def test_prior_elicitation_matches_native_and_attains_interval_mass(case):
    result = multc99_prior_from_interval(
        case["means"], [1, 0, 1, 0], width=case["width"], coverage=case["coverage"]
    )
    np.testing.assert_allclose(result.dirichlet_parameters, case["alpha"], rtol=1e-7, atol=1e-6)
    alpha = result.dirichlet_parameters[[0, 2]].sum()
    beta = result.dirichlet_parameters[[1, 3]].sum()
    lo, hi = result.interval
    assert abs(betainc(alpha, beta, hi) - betainc(alpha, beta, lo) - case["coverage"]) < 1e-9
    np.testing.assert_allclose(result.dirichlet_parameters / result.concentration, case["means"])
    assert not result.dirichlet_parameters.flags.writeable


@pytest.mark.parametrize("case", _NATIVE["precision"])
def test_precision_planning_matches_native_rounded_sample_size(case):
    design = multc99_design([2, 3], [7, 13], [Multc99Event("e", (1, 0))], max_subjects=20)
    result = multc99_precision_sample_size(
        design,
        "e",
        target_posterior_mean=case["mean"],
        width=case["width"],
        coverage=case["coverage"],
    )
    assert result.sample_size == case["sample_size"]
    assert result.interval_width <= case["width"]
    assert 0 <= result.event_count <= result.sample_size


def test_unbracketed_prior_and_unattainable_precision_targets_raise():
    with pytest.raises(ValueError, match="not bracketed"):
        multc99_prior_from_interval([0.01, 0.99], [1, 0], width=0.5, coverage=0.1)
    design = multc99_design([2, 3], [7, 13], [Multc99Event("e", (1, 0))], max_subjects=20)
    with pytest.raises(ValueError, match="within search_cap"):
        multc99_precision_sample_size(
            design, "e", target_posterior_mean=0.4, width=0.001, coverage=0.95, search_cap=10
        )


def _randomized_design(maximum=8, *, stop=True):
    return multc99_design(
        [1, 1],
        [1, 1],
        [
            Multc99Event(
                "success", (1, 0), lower_cutoff=0.3 if stop else None, event_type="efficacy"
            )
        ],
        max_subjects=maximum,
    )


def test_stopped_arm_reassignment_and_final_patient_are_retained():
    design = _randomized_design()
    schedule = [0, 1] * 4
    tapes = [[1] * 8, [0] * 8]
    without = run_multc99_randomized_trial(design, schedule, tapes, target_event="success")
    with_reassignment = run_multc99_randomized_trial(
        design,
        schedule,
        tapes,
        target_event="success",
        reassign=True,
    )
    np.testing.assert_array_equal(without.arm_sample_sizes, [2, 4])
    np.testing.assert_array_equal(with_reassignment.arm_sample_sizes, [2, 6])
    np.testing.assert_array_equal(with_reassignment.elementary_counts, [[0, 2], [6, 0]])
    assert without.selected_arm == with_reassignment.selected_arm == 1
    np.testing.assert_array_equal(with_reassignment.terminated_arms, [True, False])
    assert with_reassignment.allocation_sequence.size == 8


def test_all_arms_terminated_and_correct_conditional_target_denominator():
    design = _randomized_design()
    trial = run_multc99_randomized_trial(
        design, [0, 1] * 4, [[1] * 8] * 2, target_event="success", reassign=True
    )
    assert trial.selected_arm is None
    np.testing.assert_array_equal(trial.arm_sample_sizes, [2, 2])
    conditional = multc99_design(
        [1, 1, 5],
        [1, 1, 3],
        [Multc99Event("e", (1, 0, 0), conditioning_definition=(1, 1, 0), event_type="efficacy")],
        max_subjects=4,
    )
    result = run_multc99_randomized_trial(
        conditional, [0, 1, 0, 1], [[2, 2, 2, 2], [0, 1, 2, 2]], target_event="e"
    )
    np.testing.assert_allclose(result.target_posterior_means, [0.5, 0.5])


def test_three_way_ties_are_uniform_and_simulation_is_replayable():
    design = _randomized_design(6, stop=False)
    result = simulate_multc99_randomized(
        design, [[0, 1]] * 3, target_event="success", trials=1000, seed=37
    )
    assert result.selection_probability[0] == 0
    assert np.all(abs(result.selection_probability[1:] - 1 / 3) < 4 * result.selection_mcse[1:])
    np.testing.assert_array_equal(result.arm_sample_sizes, np.full((1000, 3), 2))
    np.testing.assert_array_equal(result.elementary_counts.sum(axis=2), result.arm_sample_sizes)
    small = simulate_multc99_randomized(
        design, [[0.3, 0.7], [0.7, 0.3]], target_event="success", trials=10, seed=23
    )
    replay = simulate_multc99_randomized(
        design, [[0.3, 0.7], [0.7, 0.3]], target_event="success", trials=10, seed=23
    )
    np.testing.assert_array_equal(small.selected_arms, replay.selected_arms)
    np.testing.assert_array_equal(small.elementary_counts, replay.elementary_counts)
    assert not small.elementary_counts.flags.writeable


def test_target_direction_and_randomized_shapes_are_validated():
    design = _randomized_design(4)
    with pytest.raises(ValueError):
        run_multc99_randomized_trial(design, [0, 1], [[0] * 4] * 2, target_event="success")
    with pytest.raises(ValueError):
        simulate_multc99_randomized(design, [[0, 0.5]] * 2, target_event="success")
    other = multc99_design([1, 1], [1, 1], [Multc99Event("e", (1, 0))], max_subjects=4)
    with pytest.raises(ValueError, match="requires explicit"):
        simulate_multc99_randomized(other, [[0, 1]] * 2, target_event="e")
