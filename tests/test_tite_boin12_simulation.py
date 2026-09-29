import numpy as np
import pytest

from mdanderson_stats.boin12 import BOIN12Design
from mdanderson_stats.tite_boin12_simulation import (
    simulate_tite_boin12,
    tite_boin12_gumbel_probabilities,
)


def _design() -> BOIN12Design:
    return BOIN12Design(0.35, 0.25, utilities=(100.0, 30.0, 65.0, 0.0))


def test_gumbel_conversion_preserves_margins_and_independence():
    tox = np.array([0.1, 0.4, 0.8])
    eff = np.array([0.7, 0.3, 0.6])
    cells = tite_boin12_gumbel_probabilities(tox, eff, association=0.0)
    np.testing.assert_allclose(cells[:, 0] + cells[:, 1], 1 - tox)
    np.testing.assert_allclose(cells[:, 2] + cells[:, 3], tox)
    np.testing.assert_allclose(cells[:, 0] + cells[:, 2], eff)
    np.testing.assert_allclose(cells[:, 1] + cells[:, 3], 1 - eff)
    np.testing.assert_allclose(
        cells,
        np.column_stack(((1 - tox) * eff, (1 - tox) * (1 - eff), tox * eff, tox * (1 - eff))),
    )
    assert not cells.flags.writeable


def test_fixed_complete_outcome_simulation_has_conserved_counts_and_replayable_streams():
    result = simulate_tite_boin12(
        _design(),
        [[0.0, 1.0, 0.0, 0.0]],
        2.0,
        toxicity_window=1.0,
        efficacy_window=1.0,
        cohorts=2,
        cohort_size=2,
        trials=8,
        seed=2**60 + 19,
    )

    np.testing.assert_array_equal(result.patients_by_dose.sum(axis=1), 4)
    np.testing.assert_array_equal(result.toxicities_by_dose, 0)
    np.testing.assert_array_equal(result.efficacies_by_dose, 0)
    np.testing.assert_array_equal(result.mean_patients, [4.0])
    np.testing.assert_array_equal(result.patients_mcse, [0.0])
    assert np.all(result.accrual_stop_time == 2.0)
    assert np.all(result.final_duration == 3.0)
    assert not result.trial_seed_triplets.flags.writeable

    replay = simulate_tite_boin12(
        _design(),
        [[0.0, 1.0, 0.0, 0.0]],
        2.0,
        toxicity_window=1.0,
        efficacy_window=1.0,
        cohorts=2,
        cohort_size=2,
        trials=8,
        trial_seed_triplets=result.trial_seed_triplets,
    )
    for field in (
        "selected_obd",
        "patients_by_dose",
        "toxicities_by_dose",
        "efficacies_by_dose",
        "final_duration",
        "accrual_stop_time",
    ):
        np.testing.assert_array_equal(getattr(result, field), getattr(replay, field))
    assert result.stop_reason == replay.stop_reason


def test_exponential_stream_replay_and_trial_clustered_selection_summary():
    args = dict(
        design=_design(),
        joint_probabilities=[[0.35, 0.25, 0.15, 0.25]],
        accrual_rate=0.5,
        toxicity_window=1.0,
        efficacy_window=2.0,
        cohorts=1,
        cohort_size=2,
        trials=12,
        arrival="exponential",
    )
    first = simulate_tite_boin12(**args, seed=371)
    second = simulate_tite_boin12(**args, trial_seed_triplets=first.trial_seed_triplets)
    np.testing.assert_array_equal(first.selected_obd, second.selected_obd)
    np.testing.assert_array_equal(first.final_duration, second.final_duration)
    assert np.isclose(first.selection_probability.sum(), 1.0)
    np.testing.assert_allclose(
        first.selection_probability,
        np.bincount(first.selected_obd, minlength=2) / first.selected_obd.size,
    )
    assert np.all(first.final_duration >= first.accrual_stop_time)


def test_single_trial_replay_marks_sampling_errors_undefined():
    result = simulate_tite_boin12(
        _design(),
        [[0.0, 1.0, 0.0, 0.0]],
        1.0,
        toxicity_window=1.0,
        efficacy_window=1.0,
        cohorts=1,
        cohort_size=1,
        trials=1,
        seed=81,
    )
    assert np.isnan(result.selection_mcse).all()
    assert np.isnan(result.patients_mcse).all()
    assert np.isnan(result.duration_mcse)
    assert np.isnan(result.accrual_stop_time_mcse)


def test_bda_simulation_requires_and_uses_explicit_sampler_configuration():
    result = simulate_tite_boin12(
        _design(),
        [[0.0, 1.0, 0.0, 0.0]],
        1.0,
        toxicity_window=1.0,
        efficacy_window=1.0,
        cohorts=2,
        cohort_size=1,
        trials=2,
        method="bda",
        prior_concentrations=[1.0, 1.0, 1.0, 1.0],
        draws=20,
        warmup=0,
        chains=2,
        seed=93,
    )
    assert result.patients_by_dose.sum(axis=1).tolist() == [2, 2]
    assert result.early_stop_probability == 0.0
    with pytest.raises(ValueError, match="explicit prior_concentrations"):
        simulate_tite_boin12(
            _design(),
            [[0.0, 1.0, 0.0, 0.0]],
            1.0,
            toxicity_window=1.0,
            efficacy_window=1.0,
            method="bda",
            seed=2,
        )


def test_early_safety_stops_are_aggregated_across_trials():
    design = BOIN12Design(0.1, 0.25, toxicity_cutoff=0.5)
    result = simulate_tite_boin12(
        design,
        [[0.0, 0.0, 0.0, 1.0]],
        1.0,
        toxicity_window=1.0,
        efficacy_window=1.0,
        cohorts=2,
        cohort_size=1,
        trials=4,
        seed=118,
    )
    assert result.early_stop_probability == 1.0
    assert result.early_stop_mcse == 0.0
    assert result.stop_reason == ("stop_safety",) * 4
    np.testing.assert_array_equal(result.patients_by_dose, 1)


def test_work_budget_rejects_before_any_trial_seed_or_rng_is_created(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("random stream construction must follow preflight")

    monkeypatch.setattr("mdanderson_stats.tite_boin12_simulation._trial_seeds", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    with pytest.raises(ValueError, match="total work estimate"):
        simulate_tite_boin12(
            _design(),
            [[0.0, 1.0, 0.0, 0.0]],
            1.0,
            toxicity_window=1.0,
            efficacy_window=1.0,
            cohorts=2,
            cohort_size=2,
            trials=100,
            seed=4,
            max_work=1,
        )
