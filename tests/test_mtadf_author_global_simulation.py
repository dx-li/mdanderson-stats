from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.mtadf_author_global_simulation import (
    replay_mtadf_author_global_trial,
    simulate_mtadf_author_global,
)


def test_observed_tape_uses_global_reference_fit_for_final_peak() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "mtadf-author-global-reference.csv"
    with fixture_path.open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["case"] == "one_dose_zero_events"]

    result = replay_mtadf_author_global_trial([0], [0], dose_count=5, cohort_size=6)
    assert result.assigned_dose.tolist() == [0]
    assert result.subjects.tolist() == [6, 0, 0, 0, 0]
    assert result.fit_count == 1
    assert result.selected_dose == 4  # rightmost equal fitted maximum
    assert result.admissible_count_before.tolist() == [5]
    assert result.admissible_count_after.tolist() == [5]
    np.testing.assert_allclose(
        result.final_fit.fitted_efficacy,
        [float(row["fitted_efficacy"]) for row in rows],
        rtol=2e-10,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        result.final_fit.coefficients,
        [float(rows[0][f"beta{index}"]) for index in range(3)],
        rtol=2e-9,
        atol=2e-10,
    )

    lagged = replay_mtadf_author_global_trial([3, 0, 0], [3, 0, 0], dose_count=5, cohort_size=3)
    assert lagged.admissible_count_after[0] == 1
    assert lagged.admissible_count_before.tolist() == [5, 1, 5]
    assert lagged.assigned_dose.tolist() == [0, 1, 0]
    assert lagged.fit_count == 3  # the rule still fits under a one-dose cap


def test_simulation_is_seed_reproducible_and_reports_source_fit_work() -> None:
    seed = np.uint64(2**63 + 2718)
    kwargs = dict(cohorts=3, cohort_size=3, trials=4, rng=seed)
    first = simulate_mtadf_author_global([0.05, 0.15, 0.3, 0.45], [0.1, 0.4, 0.65, 0.75], **kwargs)
    second = simulate_mtadf_author_global([0.05, 0.15, 0.3, 0.45], [0.1, 0.4, 0.65, 0.75], **kwargs)
    assert first.seed == int(seed)
    assert first.fit_count == 12
    assert first.nonconverged_fit_count == 0
    assert first.total_irls_iterations >= first.fit_count
    np.testing.assert_array_equal(first.patients, second.patients)
    np.testing.assert_array_equal(first.toxicities, second.toxicities)
    np.testing.assert_array_equal(first.responses, second.responses)
    np.testing.assert_array_equal(first.selected_dose, second.selected_dose)
    for trial in range(first.selected_dose.size):
        replay = replay_mtadf_author_global_trial(
            first.cohort_toxicity_history[trial],
            first.cohort_response_history[trial],
            dose_count=4,
            cohort_size=3,
        )
        np.testing.assert_array_equal(replay.assigned_dose, first.assigned_dose_history[trial])
        np.testing.assert_array_equal(replay.subjects, first.patients[trial])
        np.testing.assert_array_equal(replay.toxicities, first.toxicities[trial])
        np.testing.assert_array_equal(replay.responses, first.responses[trial])
        assert replay.selected_dose == first.selected_dose[trial]
    np.testing.assert_array_equal(first.selection_probability, second.selection_probability)
    assert not first.patients.flags.writeable
    assert not first.cohort_toxicity_history.flags.writeable
    assert np.isclose(first.selection_probability.sum(), 1.0)


def test_fit_budget_rejects_before_consuming_generator() -> None:
    rng = np.random.default_rng(91)
    control = np.random.default_rng(91)
    with pytest.raises(ValueError, match="max_total_fits"):
        simulate_mtadf_author_global(
            [0.1, 0.2],
            [0.3, 0.5],
            cohorts=3,
            trials=2,
            rng=rng,
            max_total_fits=5,
        )
    assert rng.integers(0, 2**32) == control.integers(0, 2**32)
