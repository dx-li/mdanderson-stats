import numpy as np
import pytest

from mdanderson_stats.uboin_conduct import UBOINDesign
from mdanderson_stats.uboin_simulation import (
    simulate_uboin,
    uboin_gumbel_probabilities,
)


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
        simulate_uboin(
            make_design(), np.array([[[1.0000000001, 0], [0, 0]]]), trials=1
        )
