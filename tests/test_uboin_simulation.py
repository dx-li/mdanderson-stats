import numpy as np
import pytest

from mdanderson_stats.uboin_conduct import UBOINDesign
from mdanderson_stats.uboin_simulation import (
    simulate_uboin,
    uboin_gumbel_probabilities,
)
from mdanderson_stats.uboin_titration import uboin_stage1_titration_plan


def make_design(**kwargs: object) -> UBOINDesign:
    kwargs.setdefault("candidate_scope", "tried")
    return UBOINDesign(
        prior=np.full((2, 2), 0.25),
        utilities=np.array([[30, 0], [100, 50]]),
        **kwargs,
    )


def test_gumbel_probabilities_are_normalized_and_stable() -> None:
    result = uboin_gumbel_probabilities([0.2, 0.4], [0.3, 0.8], association=1000)
    assert result.shape == (2, 2, 2)
    assert np.all(np.isfinite(result))
    assert np.allclose(result.sum(axis=(1, 2)), 1)
    independent = uboin_gumbel_probabilities([0.2], [0.3], association=0)
    assert np.allclose(independent[0], [[0.56, 0.14], [0.24, 0.06]])


def test_all_toxic_stops_without_selection() -> None:
    probabilities = np.zeros((2, 2, 2))
    probabilities[:, 0, 1] = 1
    result = simulate_uboin(
        make_design(max_patients=12), probabilities, cohort_size=3, trials=4, seed=1
    )
    assert np.all(result.selections == 0)
    assert np.all(result.stop_reason == "stop_safety")
    assert np.all(result.joint_counts.sum(axis=(1, 2, 3)) == 3)


def test_safe_response_path_reproducible_and_partial_last_cohort() -> None:
    probabilities = np.zeros((2, 2, 2))
    probabilities[:, 1, 0] = 1
    design = make_design(max_patients=5, s1=3, s2=4)
    first = simulate_uboin(design, probabilities, cohort_size=3, trials=3, seed=4)
    second = simulate_uboin(design, probabilities, cohort_size=3, trials=3, seed=4)
    assert np.array_equal(first.joint_counts, second.joint_counts)
    assert np.all(first.joint_counts.sum(axis=(1, 2, 3)) == 5)
    assert np.all(first.stop_reason == "stop_max_patients")


def test_early_stop_is_patient_capacity_not_selection_status() -> None:
    safe = np.zeros((1, 2, 2))
    safe[0, 1, 0] = 1
    early = simulate_uboin(
        make_design(s1=3, s2=6, max_patients=12), safe, cohort_size=3, trials=1, seed=1
    )
    assert early.selections[0] == 1
    assert early.early_stop_probability == 1

    futile = np.zeros((1, 2, 2))
    futile[0, 0, 0] = 1
    at_cap = simulate_uboin(
        make_design(s1=3, s2=9, max_patients=6), futile, cohort_size=3, trials=1, seed=1
    )
    assert at_cap.selections[0] == 0
    assert at_cap.early_stop_probability == 0


def test_categorical_simulation_and_guards() -> None:
    design = UBOINDesign(
        prior=np.full((3, 3), 1 / 9),
        utilities=np.arange(9, dtype=float).reshape(3, 3),
        candidate_scope="tried",
        max_patients=3,
        s1=2,
        s2=3,
        dlt_level=2,
        response_level=2,
    )
    probabilities = np.full((1, 3, 3), 1 / 9)
    result = simulate_uboin(design, probabilities, cohort_size=2, trials=2, seed=2)
    assert result.joint_counts.shape == (2, 1, 3, 3)
    assert result.mean_toxicities.shape == (1,)
    assert result.mean_responses.shape == (1,)
    with pytest.raises(ValueError):
        simulate_uboin(design, probabilities, trials=100_000)


def test_probability_validation() -> None:
    with pytest.raises(ValueError):
        uboin_gumbel_probabilities([0.1, 0.2], [0.3])
    with pytest.raises(ValueError):
        uboin_gumbel_probabilities([1.1], [0.3])
    with pytest.raises(ValueError):
        simulate_uboin(make_design(), np.array([[[1.0000000001, 0], [0, 0]]]), trials=1)


def test_uboin_titration_replay_contract() -> None:
    common = dict(
        max_dose=4,
        toxicity_category_count=3,
        starting_dose=1,
        cohort_size=3,
        max_patients=20,
        dlt_level=2,
        grade2_toxicity_level=2,
    )
    highest = uboin_stage1_titration_plan([1, 1, 1, 1], **common)
    assert highest.titration_doses == (1, 2, 3, 4)
    assert highest.end_reason == "highest_dose_cap"
    assert highest.resume_dose == 4
    assert highest.top_up_patients == 2

    dlt = uboin_stage1_titration_plan([1, 3], **common)
    assert dlt.titration_doses == (1, 2)
    assert dlt.end_reason == "first_dlt"
    assert dlt.resume_dose == 2
    assert dlt.top_up_patients == 2

    grade2 = uboin_stage1_titration_plan([2, 1, 2], **common)
    assert grade2.titration_doses == (1, 2, 3)
    assert grade2.grade2_toxicities == 2
    assert grade2.end_reason == "second_grade2"
    assert grade2.resume_dose == 3
    assert grade2.top_up_patients == 2

    lower_cap = uboin_stage1_titration_plan([1, 1], **{**common, "titration_cap": 2})
    assert lower_cap.end_reason == "lower_cap_without_trigger"
    assert lower_cap.resume_dose == 3
    assert lower_cap.top_up_patients == 0

    clean_high_cap_from_dose3 = uboin_stage1_titration_plan(
        [1, 1], **{**common, "starting_dose": 3}
    )
    assert clean_high_cap_from_dose3.titration_doses == (3, 4)
    assert clean_high_cap_from_dose3.resume_dose == 4
    assert clean_high_cap_from_dose3.top_up_patients == 2

    budget_at_lower_cap = uboin_stage1_titration_plan(
        [1, 1], **{**common, "titration_cap": 2, "max_patients": 2}
    )
    assert budget_at_lower_cap.end_reason == "lower_cap_without_trigger"
    assert budget_at_lower_cap.resume_dose == 3
    assert budget_at_lower_cap.top_up_patients == 0

    truncated = uboin_stage1_titration_plan([1, 3], **{**common, "max_patients": 3})
    assert truncated.end_reason == "first_dlt"
    assert truncated.top_up_patients == 1

    prefix = uboin_stage1_titration_plan([1], **common)
    assert not prefix.complete
    assert prefix.next_titration_dose == 2
    with pytest.raises(ValueError, match="after the titration stop"):
        uboin_stage1_titration_plan([3, 1], **common)


def _three_category_design(*, max_patients: int, s1: int, s2: int) -> UBOINDesign:
    return UBOINDesign(
        prior=np.full((2, 3), 1 / 6),
        utilities=np.array([[0, 10, 20], [30, 60, 100]], dtype=float),
        candidate_scope="tried",
        dlt_level=2,
        s1=s1,
        s2=s2,
        max_patients=max_patients,
    )


def test_uboin_titration_simulation_topup_and_lower_cap() -> None:
    safe = np.zeros((4, 2, 3))
    safe[:, 1, 0] = 1
    highest = simulate_uboin(
        _three_category_design(max_patients=6, s1=3, s2=6),
        safe,
        cohort_size=3,
        trials=1,
        accelerated_titration=True,
        grade2_toxicity_level=2,
        seed=142,
    )
    np.testing.assert_array_equal(highest.joint_counts[0].sum(axis=(1, 2)), [1, 1, 1, 3])
    assert highest.titration_patients is not None
    assert highest.titration_patients.tolist() == [4]
    assert highest.titration_end_reason.tolist() == ["highest_dose_cap"]

    lower_cap = simulate_uboin(
        _three_category_design(max_patients=5, s1=3, s2=5),
        safe,
        cohort_size=3,
        trials=1,
        accelerated_titration=True,
        titration_cap=2,
        grade2_toxicity_level=2,
        seed=142,
    )
    np.testing.assert_array_equal(lower_cap.joint_counts[0].sum(axis=(1, 2)), [1, 1, 3, 0])
    assert lower_cap.titration_end_reason.tolist() == ["lower_cap_without_trigger"]

    budget_at_lower_cap = simulate_uboin(
        _three_category_design(max_patients=2, s1=1, s2=2),
        safe,
        cohort_size=3,
        trials=1,
        accelerated_titration=True,
        titration_cap=2,
        grade2_toxicity_level=2,
        seed=142,
    )
    np.testing.assert_array_equal(
        budget_at_lower_cap.joint_counts[0].sum(axis=(1, 2)), [1, 1, 0, 0]
    )
    assert budget_at_lower_cap.stop_reason.tolist() == ["stop_max_patients"]


def test_uboin_titration_simulation_stops_on_dlt_or_second_grade2() -> None:
    dlt_probs = np.zeros((4, 2, 3))
    dlt_probs[0, 0, 0] = 1
    dlt_probs[1:, 0, 2] = 1
    dlt = simulate_uboin(
        _three_category_design(max_patients=3, s1=3, s2=4),
        dlt_probs,
        cohort_size=3,
        trials=1,
        accelerated_titration=True,
        grade2_toxicity_level=2,
        seed=1,
    )
    np.testing.assert_array_equal(dlt.joint_counts[0].sum(axis=(1, 2)), [1, 2, 0, 0])
    assert dlt.titration_end_reason.tolist() == ["first_dlt"]

    grade2_probs = np.zeros((4, 2, 3))
    grade2_probs[:, 0, 0] = 1
    grade2_probs[0, 0] = [0, 1, 0]
    grade2_probs[2, 0] = [0, 1, 0]
    grade2 = simulate_uboin(
        _three_category_design(max_patients=5, s1=3, s2=5),
        grade2_probs,
        cohort_size=3,
        trials=1,
        accelerated_titration=True,
        grade2_toxicity_level=2,
        seed=1,
    )
    np.testing.assert_array_equal(grade2.joint_counts[0].sum(axis=(1, 2)), [1, 1, 3, 0])
    assert grade2.titration_grade2_toxicities.tolist() == [2]
    assert grade2.titration_end_reason.tolist() == ["second_grade2"]


def test_uboin_titration_noop_preserves_ordinary_random_path() -> None:
    probabilities = np.full((2, 2, 2), 0.25)
    for design, cohort in (
        (make_design(max_patients=6, s1=3, s2=6, starting_dose=2), 3),
        (make_design(max_patients=5, s1=3, s2=4), 1),
    ):
        ordinary = simulate_uboin(design, probabilities, cohort_size=cohort, trials=3, seed=142)
        requested = simulate_uboin(
            design,
            probabilities,
            cohort_size=cohort,
            trials=3,
            accelerated_titration=True,
            seed=142,
        )
        np.testing.assert_array_equal(requested.joint_counts, ordinary.joint_counts)
        np.testing.assert_array_equal(requested.selections, ordinary.selections)
        np.testing.assert_array_equal(requested.stop_reason, ordinary.stop_reason)
