import numpy as np
import pytest

from mdanderson_stats.boin_combination import BOINCombDesign
from mdanderson_stats.boin_waterfall_simulation import simulate_boin_waterfall
from mdanderson_stats.boin_waterfall_trial import run_boin_waterfall_trial


def _uniform_tape(events: list[int], total: int, cohort_size: int = 3) -> np.ndarray:
    tape = np.full(total, 0.9)
    for index, count in enumerate(events):
        tape[index * cohort_size : (index + 1) * cohort_size] = np.r_[
            np.full(count, 0.1), np.full(cohort_size - count, 0.9)
        ]
    return tape


def test_replay_runs_staircase_and_special_same_row_subtrial() -> None:
    result = run_boin_waterfall_trial(
        BOINCombDesign(target=0.3),
        np.full((2, 3), 0.3),
        [4, 2],
        _uniform_tape([0, 0, 0, 0, 0, 0], 18),
    )
    np.testing.assert_array_equal(result.patients, [[3, 0, 6], [3, 3, 3]])
    np.testing.assert_array_equal(result.toxicities, np.zeros((2, 3), dtype=int))
    assert [item.dose for item in result.cohort_history] == [
        (1, 1),
        (2, 1),
        (2, 2),
        (2, 3),
        (1, 3),
        (1, 3),
    ]
    assert result.selected_contour == ((1, 3), (2, 3))
    assert result.total_patients == 18


def test_safety_stopped_subtrial_keeps_actual_counts_and_flags_bad_continuity() -> None:
    result = run_boin_waterfall_trial(
        BOINCombDesign(target=0.3),
        np.full((2, 3), 0.3),
        [4, 2],
        _uniform_tape([0, 0, 0, 0, 3, 3], 18),
    )
    np.testing.assert_array_equal(result.patients, [[3, 3, 3], [3, 3, 3]])
    np.testing.assert_array_equal(result.toxicities, [[0, 3, 3], [0, 0, 0]])
    assert result.total_patients == 18
    assert result.total_toxicities == 6
    assert result.subtrials[-1].stop_reason == "stop_safety"
    assert result.source_contour[0] == (1, 2)
    assert result.selected_contour[0] is None
    assert result.continuity_blocked[0]
    for i, pair in enumerate(result.selected_contour):
        if pair is not None:
            assert result.patients[i, pair[1] - 1] > 0
            assert not result.eliminated[i, pair[1] - 1]


def test_final_extra_safe_prior_and_serial_simulation_summaries() -> None:
    strict = BOINCombDesign(target=0.3, extra_safe=True, safety_offset=0.35)
    stopped = run_boin_waterfall_trial(
        strict,
        np.full((2, 3), 0.3),
        [1, 1],
        _uniform_tape([1], 6),
    )
    assert stopped.subtrials[0].stop_reason == "stop_safety"
    assert stopped.total_patients == 3
    assert stopped.selected_contour == (None, None)

    first = simulate_boin_waterfall(
        BOINCombDesign(target=0.3), np.zeros((2, 3)), [2, 1], trials=4, rng=12
    )
    second = simulate_boin_waterfall(
        BOINCombDesign(target=0.3), np.zeros((2, 3)), [2, 1], trials=4, rng=12
    )
    np.testing.assert_array_equal(first.patients, second.patients)
    np.testing.assert_array_equal(first.selection_probability, second.selection_probability)
    np.testing.assert_array_equal(first.total_patients, first.patients.sum(axis=(1, 2)))
    np.testing.assert_allclose(
        first.selection_probability.sum(axis=1) + first.no_selection_probability, 1
    )
    assert len(first.histories) == 4

    generator = np.random.default_rng(4)
    state = generator.bit_generator.state
    with pytest.raises(ValueError, match="bounded"):
        simulate_boin_waterfall(
            BOINCombDesign(target=0.3),
            np.zeros((2, 3)),
            [2, 1],
            trials=100_000,
            rng=generator,
        )
    assert generator.bit_generator.state == state
