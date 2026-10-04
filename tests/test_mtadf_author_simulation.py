import numpy as np
import pytest

from mdanderson_stats.mtadf_author_simulation import (
    replay_mtadf_author_trial,
    simulate_mtadf_author,
)


def test_author_replay_retains_lagged_cap_and_conserves_cohorts():
    tox = np.array([[3, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    eff = np.zeros_like(tox)
    result = replay_mtadf_author_trial(tox, eff)

    assert result.admissible_count_before[0] == 4
    assert result.admissible_count_after[0] == 1
    assert result.assigned_dose[1] == 1  # one-based author dose 2
    assert result.subjects.sum() == 9
    np.testing.assert_array_equal(result.assigned_dose, [0, 1, 0])
    np.testing.assert_array_equal(result.subjects, [6, 3, 0, 0])
    assert result.stop_reason == "maximum_enrollment"


def test_author_final_rightmost_untried_tie_and_seeded_simulation():
    zeros = np.zeros((1, 4), dtype=int)
    one_cohort = replay_mtadf_author_trial(zeros, zeros)
    assert one_cohort.selected_dose == 3

    kwargs = dict(
        true_toxicity=[0.05, 0.20, 0.35],
        true_efficacy=[0.10, 0.50, 0.40],
        cohorts=4,
        cohort_size=3,
        trials=12,
    )
    first = simulate_mtadf_author(**kwargs, rng=274)
    second = simulate_mtadf_author(**kwargs, rng=274)
    np.testing.assert_array_equal(first.patients, second.patients)
    np.testing.assert_array_equal(first.toxicities, second.toxicities)
    np.testing.assert_array_equal(first.responses, second.responses)
    np.testing.assert_array_equal(first.selected_dose, second.selected_dose)
    assert np.all(first.patients.sum(axis=1) == 12)
    assert np.all(first.toxicities <= first.patients)
    assert np.all(first.responses <= first.patients)
    np.testing.assert_allclose(first.selection_probability.sum(), 1.0)

    rejected_rng = np.random.default_rng(991)
    untouched_rng = np.random.default_rng(991)
    with pytest.raises(ValueError, match="potential_cells"):
        simulate_mtadf_author(**kwargs, rng=rejected_rng, max_total_potential_cells=1)
    assert rejected_rng.random() == untouched_rng.random()
