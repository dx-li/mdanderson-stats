import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import run_multc_legacy_duration

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/multc-lean-native-duration.json").read_text()
)


def arguments(case):
    inputs = case["inputs"]
    return dict(
        max_subjects=inputs["n"],
        response_stop_at=inputs["response"],
        nontoxicity_stop_at=inputs["notox"],
        joint_probabilities=inputs["prob"],
        mean_interarrival=inputs["mean"],
        response_window=inputs["window"],
        uniforms=inputs["uniform"],
        unit_exponentials=inputs["unit_exponential"],
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["case"])
def test_duration_counts_and_random_consumption_match_original_x86_instructions(case):
    trial = run_multc_legacy_duration(**arguments(case))
    native = case["native"]
    for field in (
        "sample_size",
        "responses",
        "toxicities",
        "balks",
        "uniforms_consumed",
        "exponentials_consumed",
    ):
        assert getattr(trial, field) == native[field]
    assert trial.duration == pytest.approx(native["duration"], rel=2e-15, abs=1e-14)
    assert trial.patients[0].enrollment_time == 0
    assert trial.duration == max(p.followup_time for p in trial.patients)
    assert trial.responses == sum(p.response for p in trial.patients)
    assert trial.toxicities == sum(p.toxicity for p in trial.patients)
    assert trial.uniforms_consumed == trial.sample_size


def basic(**changes):
    settings = dict(
        max_subjects=3,
        response_stop_at=[],
        nontoxicity_stop_at=[],
        joint_probabilities=[0.25] * 4,
        mean_interarrival=1.0,
        response_window=2.0,
        uniforms=[0.9] * 3,
        unit_exponentials=[0.1] * 100,
    )
    return {**settings, **changes}


def test_clipping_retains_the_original_window_mass_and_shared_followup():
    trial = run_multc_legacy_duration(
        **basic(uniforms=[0.1] * 3, unit_exponentials=[20, 0.1, 30, 0.1, 40])
    )
    assert all(patient.response and patient.toxicity for patient in trial.patients)
    np.testing.assert_allclose(
        [p.followup_time - p.enrollment_time for p in trial.patients], [2, 2, 2]
    )
    assert trial.duration == 2.2
    assert trial.decision_time == 0.2
    assert trial.decision == "cap_complete"


def test_balked_arrivals_are_discarded_instead_of_freezing_accrual():
    trial = run_multc_legacy_duration(**basic(max_subjects=6, response_stop_at=[3, 5, 7]))
    assert trial.sample_size == 3
    assert trial.balks == 19
    assert trial.decision == "stop_response"
    assert trial.duration == 2.2
    assert trial.decision_time >= trial.duration
    assert trial.patients[-1].could_stop_next
    np.testing.assert_allclose([p.enrollment_time for p in trial.patients], [0, 0.1, 0.2])


def test_stop_reason_can_be_both_and_counts_use_nontoxicities():
    trial = run_multc_legacy_duration(
        **basic(
            max_subjects=6,
            response_stop_at=[3, 5, 7],
            nontoxicity_stop_at=[3, 5, 7],
            uniforms=[0.7] * 6,
        )
    )
    assert trial.sample_size == 3
    assert trial.responses == 0
    assert trial.toxicities == 3
    assert trial.decision == "stop_both"


def test_exact_followup_arrival_does_not_balk():
    trial = run_multc_legacy_duration(
        **basic(max_subjects=6, response_stop_at=[1], unit_exponentials=[2])
    )
    assert trial.sample_size == 1
    assert trial.balks == 0
    assert trial.duration == trial.decision_time == 2


def test_native_followup_duration_can_precede_stopping_evaluation():
    trial = run_multc_legacy_duration(
        **basic(max_subjects=6, response_stop_at=[1], unit_exponentials=[10])
    )
    assert trial.sample_size == 1
    assert trial.duration == 2
    assert trial.decision_time == 10


def test_latent_response_counts_can_avoid_suspension_before_followup():
    trial = run_multc_legacy_duration(
        **basic(
            max_subjects=6,
            response_stop_at=[3, 7],
            uniforms=[0.3, 0.9, 0.9, 0.9, 0.9, 0.9],
            unit_exponentials=[20] + [0.1] * 100,
        )
    )
    assert trial.patients[0].followup_time == 2
    assert trial.patients[2].enrollment_time < 2
    assert not trial.patients[2].could_stop_next
    assert trial.balks == 0


def test_short_streams_and_clock_resolution_fail_explicitly():
    with pytest.raises(ValueError, match="uniforms exhausted"):
        run_multc_legacy_duration(**basic(uniforms=[0.9]))
    with pytest.raises(ValueError, match="unit_exponentials exhausted"):
        run_multc_legacy_duration(**basic(unit_exponentials=[]))
    with pytest.raises(ValueError, match="unit_exponentials exhausted"):
        run_multc_legacy_duration(**basic(response_stop_at=[1], unit_exponentials=[0, 0, 0]))
    with pytest.raises(ArithmeticError, match="advance the trial clock"):
        run_multc_legacy_duration(**basic(unit_exponentials=[1e100, 1e-100]))
    with pytest.raises(ArithmeticError, match="not representable"):
        run_multc_legacy_duration(**basic(mean_interarrival=1e308, unit_exponentials=[1e308]))


def test_large_clipped_response_is_not_multiplied_into_overflow():
    trial = run_multc_legacy_duration(
        **basic(uniforms=[0.1] * 3, unit_exponentials=[1e308, 0.1, 1e308, 0.1, 1e308])
    )
    assert trial.duration == 2.2


@pytest.mark.parametrize(
    "change",
    [
        {"max_subjects": 2},
        {"max_subjects": 1001},
        {"max_subjects": True},
        {"max_subjects": 3.0},
        {"response_stop_at": [0]},
        {"response_stop_at": [-1]},
        {"response_stop_at": [5]},
        {"response_stop_at": [1.5]},
        {"response_stop_at": [True]},
        {"response_stop_at": [1] * 5},
        {"nontoxicity_stop_at": [[1]]},
        {"joint_probabilities": [0.2] * 4},
        {"joint_probabilities": [-1, 1, 1, 0]},
        {"joint_probabilities": [1, 0, 0]},
        {"joint_probabilities": [float("nan"), 0, 0, 1]},
        {"mean_interarrival": 0},
        {"mean_interarrival": 1j},
        {"response_window": True},
        {"response_window": float("inf")},
        {"uniforms": [0]},
        {"uniforms": [1]},
        {"uniforms": [True]},
        {"uniforms": [[0.2]]},
        {"uniforms": [0.5] * 4},
        {"unit_exponentials": [-1]},
        {"unit_exponentials": [float("inf")]},
        {"unit_exponentials": [1j]},
        {"unit_exponentials": np.broadcast_to(1.0, (1_000_001,))},
    ],
)
def test_invalid_inputs_fail_before_replay(change):
    with pytest.raises(ValueError):
        run_multc_legacy_duration(**basic(**change))
