import csv
from pathlib import Path

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


def test_replay_runs_staircase_and_earlier_row_subtrial() -> None:
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


def test_titration_assigns_staircase_and_tops_up_terminal_cell() -> None:
    space = [(1, 1), (2, 1), (2, 2), (2, 3)]
    tape = np.full(9, 0.9)  # 4 reserved staircase draws, then 5 cohort draws.
    result = run_boin_waterfall_trial(
        BOINCombDesign(target=0.3),
        np.zeros((2, 3)),
        [1, 1],
        tape,
        cohort_size=3,
        start_dose=4,  # Native titration restarts at the first staircase position.
        titration=True,
    )
    assert result.titration_end_reason == "upper_dose_no_dlt"
    assert result.titration_endpoint == (2, 3)
    np.testing.assert_array_equal(result.titration_patients, [[1, 0, 0], [1, 1, 1]])
    assert [item.dose for item in result.cohort_history[:4]] == space
    assert all(
        item.phase == "titration" and item.patients == 1 for item in result.cohort_history[:4]
    )
    top_up = result.cohort_history[4]
    assert (top_up.phase, top_up.dose, top_up.patients, top_up.toxicities) == (
        "titration_topup",
        (2, 3),
        2,
        0,
    )
    assert result.total_patients == int(result.patients.sum())
    assert result.total_toxicities == int(result.toxicities.sum())


def test_titration_stops_staircase_on_first_dlt_and_cohort_size_one_disables_it() -> None:
    design = BOINCombDesign(target=0.3)
    probabilities = np.full((2, 3), 0.1)
    tape = np.full(15, 0.9)
    tape[0] = 0.0
    result = run_boin_waterfall_trial(
        design, probabilities, [2, 2], tape, cohort_size=3, titration=True
    )
    assert result.titration_end_reason == "first_dlt"
    assert result.titration_endpoint == (1, 1)
    np.testing.assert_array_equal(result.titration_patients, [[1, 0, 0], [0, 0, 0]])
    assert result.cohort_history[0].phase == "titration"
    assert result.cohort_history[1].phase == "titration_topup"
    assert result.cohort_history[1].dose == (1, 1)
    assert result.cohort_history[1].patients == 2

    no_titration = run_boin_waterfall_trial(
        design, probabilities, [2, 2], np.full(4, 0.9), cohort_size=1
    )
    requested_but_disabled = run_boin_waterfall_trial(
        design, probabilities, [2, 2], np.full(4, 0.9), cohort_size=1, titration=True
    )
    np.testing.assert_array_equal(no_titration.patients, requested_but_disabled.patients)
    np.testing.assert_array_equal(no_titration.toxicities, requested_but_disabled.toxicities)
    assert requested_but_disabled.titration_endpoint is None
    assert requested_but_disabled.titration_end_reason == "cohort_size_one"


def test_titration_simulation_replayable_with_actual_count_conservation() -> None:
    first = simulate_boin_waterfall(
        BOINCombDesign(target=0.3),
        np.zeros((2, 3)),
        [2, 2],
        cohort_size=3,
        trials=4,
        titration=True,
        rng=19,
    )
    second = simulate_boin_waterfall(
        BOINCombDesign(target=0.3),
        np.zeros((2, 3)),
        [2, 2],
        cohort_size=3,
        trials=4,
        titration=True,
        rng=19,
    )
    np.testing.assert_array_equal(first.patients, second.patients)
    np.testing.assert_array_equal(first.titration_patients, second.titration_patients)
    np.testing.assert_array_equal(first.total_patients, first.patients.sum(axis=(1, 2)))
    np.testing.assert_array_equal(first.total_toxicities, first.toxicities.sum(axis=(1, 2)))
    np.testing.assert_array_equal(first.titration_patients.sum(axis=(1, 2)), np.full(4, 4))
    assert first.titration_endpoint == ((2, 3),) * 4
    assert np.all(first.total_patients <= 15)  # 12 planned plus at most staircase length minus one.


def test_titration_overhead_is_preflighted_before_simulation_rng_use() -> None:
    generator = np.random.default_rng(83)
    state = generator.bit_generator.state
    with pytest.raises(ValueError, match="titration overhead"):
        simulate_boin_waterfall(
            BOINCombDesign(target=0.3),
            np.zeros((2, 3)),
            [167, 166],
            cohort_size=3,
            trials=1,
            titration=True,
            rng=generator,
        )
    assert generator.bit_generator.state == state


def test_titration_matches_independent_original_r_assignment_traces() -> None:
    fixture_dir = Path(__file__).parent / "fixtures"
    with (fixture_dir / "boin-waterfall-titration.csv").open(newline="") as stream:
        summaries = {row["case"]: row for row in csv.DictReader(stream)}
    tapes: dict[str, list[float]] = {name: [] for name in summaries}
    with (fixture_dir / "boin-waterfall-titration-tapes.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            tapes[row["case"]].append(float(row["uniform"]))
    expected_trace: dict[str, list[tuple[str, int, int, int, int]]] = {
        name: [] for name in summaries
    }
    with (fixture_dir / "boin-waterfall-titration-traces.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            expected_trace[row["case"]].append(
                (
                    row["phase"],
                    int(row["dose_a"]),
                    int(row["dose_b"]),
                    int(row["patients"]),
                    int(row["toxicities"]),
                )
            )
    expected_subtrials: dict[str, list[dict[str, str]]] = {name: [] for name in summaries}
    with (fixture_dir / "boin-waterfall-titration-subtrials.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            expected_subtrials[row["case"]].append(row)

    for name, reference in summaries.items():
        budgets = [int(value) for value in reference["budgets"].split(";")]
        cohort_size = int(reference["cohort_size"])
        result = run_boin_waterfall_trial(
            BOINCombDesign(target=0.3),
            np.full((2, 3), 0.3),
            budgets,
            np.asarray(tapes[name]),
            cohort_size=cohort_size,
            titration=True,
        )
        np.testing.assert_array_equal(
            result.patients,
            np.asarray([int(value) for value in reference["actual_patients"].split(";")]).reshape(
                2, 3
            ),
        )
        np.testing.assert_array_equal(
            result.toxicities,
            np.asarray([int(value) for value in reference["actual_toxicities"].split(";")]).reshape(
                2, 3
            ),
        )
        actual_trace = [
            (
                "titration" if item.phase == "titration" else "cohort",
                item.dose[0],
                item.dose[1],
                item.patients,
                item.toxicities,
            )
            for item in result.cohort_history
        ]
        assert actual_trace == expected_trace[name]
        assert result.total_patients == int(reference["actual_total_patients"])
        assert result.total_toxicities == int(reference["actual_total_toxicities"])
        assert result.total_patients == int(result.patients.sum())
        assert result.total_toxicities == int(result.toxicities.sum())
        if name != "first_dlt_safety":
            expected_final_mask = [
                value == "TRUE" for value in reference["native_eliminated"].split(";")
            ]
            np.testing.assert_array_equal(
                result.eliminated, np.asarray(expected_final_mask).reshape(2, 3)
            )
        snapshots = expected_subtrials[name]
        assert len(result.subtrials) == len(snapshots)
        for actual, expected in zip(result.subtrials, snapshots, strict=True):
            assert actual.start_position == int(expected["start"])
            assert actual.cohort_budget == int(expected["budget"])
            source_space = ";".join(
                str((column - 1) * 2 + row) for row, column in actual.dose_space
            )
            assert source_space == expected["space"]
            np.testing.assert_array_equal(
                actual.patients,
                np.asarray([int(value) for value in expected["patients"].split(";")]).reshape(2, 3),
            )
            np.testing.assert_array_equal(
                actual.toxicities,
                np.asarray([int(value) for value in expected["toxicities"].split(";")]).reshape(
                    2, 3
                ),
            )
            np.testing.assert_array_equal(
                actual.eliminated,
                np.asarray([int(value) for value in expected["eliminated"].split(";")])
                .reshape(2, 3)
                .astype(bool),
            )
            candidate = actual.candidate or (0, 0)
            assert candidate == (int(expected["selected_a"]), int(expected["selected_b"]))
            expected_escalation = expected["is_escalation"] in ("1", "TRUE")
            if actual.escalation_allowed is None:
                assert not expected_escalation
            else:
                assert actual.escalation_allowed == expected_escalation
