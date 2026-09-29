import numpy as np
import pytest

from mdanderson_stats.iboin import IBOINDesign
from mdanderson_stats.iboin_final import select_iboin_mtd, select_iboin_trial_mtd
from mdanderson_stats.iboin_simulation import simulate_iboin, simulate_iboin_trial
from mdanderson_stats.iboin_trial import replay_iboin_trial


def test_final_rates_follow_explicit_raw_and_borrowed_equations():
    design = IBOINDesign([0.1, 0.25, 0.4], [4, 2, 3], target=0.25)
    raw = select_iboin_mtd(
        design,
        [4, 4, 4],
        [0, 2, 4],
        prior_mode="none",
        isotonic_weights="equal",
    )
    np.testing.assert_allclose(raw.raw_rate, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(raw.isotonic_rate, [0.0, 0.5, 1.0])
    borrowed = select_iboin_mtd(
        design,
        [4, 4, 4],
        [0, 2, 4],
        prior_mode="original",
        isotonic_weights="effective",
    )
    expected = (np.array([0, 2, 4]) + np.array([4, 2, 3]) * np.array([0.1, 0.25, 0.4])) / np.array(
        [8, 6, 7]
    )
    np.testing.assert_allclose(borrowed.raw_rate, expected)
    np.testing.assert_array_equal(borrowed.prior_ess_used, [4, 2, 3])
    assert borrowed.weight_mode == "effective"


def test_isotonic_fit_uses_eliminated_treated_dose_and_ties_are_explicit():
    design = IBOINDesign([0.1, 0.25, 0.4], [0, 0, 0], target=0.25, elimination_probability=0.999)
    result = select_iboin_mtd(
        design,
        [2, 2, 2],
        [0, 2, 1],
        prior_mode="none",
        isotonic_weights=[1, 10, 1],
        eliminated=np.array([False, True, False]),
        tie_policy="lowest",
    )
    # The middle treated dose participates in the fit even though it cannot be selected.
    assert result.fit_mask.tolist() == [True, True, True]
    assert result.candidate_mask.tolist() == [True, False, False]
    assert result.selected_dose == 1
    tied_low = select_iboin_mtd(
        design,
        [4, 4, 0],
        [1, 1, 0],
        prior_mode="none",
        isotonic_weights="equal",
        tie_policy="lowest",
    )
    tied_high = select_iboin_mtd(
        design,
        [4, 4, 0],
        [1, 1, 0],
        prior_mode="none",
        isotonic_weights="equal",
        tie_policy="highest",
    )
    assert tied_low.selected_dose == 1 and tied_high.selected_dose == 2


def test_extra_safe_terminal_suppresses_mtd_even_without_elimination():
    design = IBOINDesign(
        [0.1, 0.25, 0.4],
        [0, 0, 0],
        target=0.25,
        elimination_probability=0.99,
        extra_safe=True,
    )
    # Three of four DLTs crosses the stricter extra-safe cutoff but not .99.
    result = select_iboin_mtd(
        design,
        [4, 0, 0],
        [3, 0, 0],
        prior_mode="none",
        isotonic_weights="patients",
    )
    assert result.safety_stopped
    assert result.selected_dose is None
    assert not result.eliminated.any()


def test_trial_selector_requires_terminal_replay_and_honors_safety_stop():
    design = IBOINDesign([0.1, 0.25, 0.4], [0, 0, 0], target=0.25)
    pending = replay_iboin_trial(design, [], [], cohort_size=1, titration=False, max_patients=3)
    with pytest.raises(ValueError, match="terminal"):
        select_iboin_trial_mtd(design, pending, prior_mode="none", isotonic_weights="patients")
    terminal = replay_iboin_trial(
        design,
        [1, 1, 1],
        [0, 0, 0],
        cohort_size=1,
        titration=False,
        max_patients=3,
    )
    selected = select_iboin_trial_mtd(
        design, terminal, prior_mode="none", isotonic_weights="patients"
    )
    assert terminal.stop_reason == "stop_safety"
    assert selected.safety_stopped and selected.selected_dose is None


def test_single_trial_uses_shared_conduct_and_mutually_exclusive_severity():
    design = IBOINDesign([0.1, 0.25, 0.4], [0, 0, 0], target=0.25)
    result = simulate_iboin_trial(
        design,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        cohort_size=1,
        max_patients=4,
        prior_mode="none",
        isotonic_weights="equal",
        seed=42,
        titration=False,
    )
    assert result.replay.assigned_dose.tolist() == [1, 2, 3, 3]
    assert not result.replay.dlt.any() and not result.replay.grade2.any()
    assert result.selection.selected_dose == 1
    with pytest.raises(ValueError, match="sum to at most one"):
        simulate_iboin_trial(
            design,
            [0.7, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            cohort_size=1,
            max_patients=4,
            prior_mode="none",
            isotonic_weights="equal",
            seed=42,
            titration=False,
        )


def test_aggregate_simulation_replays_seeds_and_conserves_counts():
    design = IBOINDesign([0.1, 0.25, 0.4], [0, 0, 0], target=0.25)
    args = dict(
        design=design,
        grade2_probability=[0.1, 0.2, 0.1],
        dlt_probability=[0.02, 0.1, 0.25],
        cohort_size=2,
        max_patients=12,
        repetitions=8,
        prior_mode="none",
        isotonic_weights="patients",
        titration=True,
    )
    first = simulate_iboin(**args, seed=314)
    second = simulate_iboin(**args, seed=314)
    replayed = simulate_iboin(**args, trial_seeds=first.trial_seeds)
    np.testing.assert_array_equal(first.trial_seeds, second.trial_seeds)
    np.testing.assert_array_equal(first.selected_dose, second.selected_dose)
    np.testing.assert_array_equal(first.stop_reason, replayed.stop_reason)
    np.testing.assert_array_equal(first.total_patients, replayed.total_patients)
    assert first.selection_probability.sum() == 1
    assert first.stop_reason_probability.sum() == 1
    assert np.all(first.mean_grade2_by_dose + first.mean_dlt_by_dose <= first.mean_patients_by_dose)
    assert np.isfinite(first.mean_patients_mcse).all()
    assert np.isfinite(first.mean_dlt_mcse).all()
    assert np.isfinite(first.mean_grade2_mcse).all()
    assert np.isfinite(first.total_patients_mcse)


def test_simulation_validates_selection_policy_and_real_probabilities_pre_rng():
    design = IBOINDesign([0.1, 0.25, 0.4], [0, 0, 0], target=0.25)
    with pytest.raises(ValueError, match="positive at every dose"):
        simulate_iboin(
            design,
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            cohort_size=1,
            max_patients=3,
            repetitions=2,
            prior_mode="none",
            isotonic_weights=[1.0, 1.0, 0.0],
            seed=1,
        )


def test_final_selector_rejects_complex_counts_and_custom_weights():
    design = IBOINDesign([0.1, 0.25, 0.4], [0, 0, 0], target=0.25)
    with pytest.raises(ValueError, match="counts must be real"):
        select_iboin_mtd(
            design,
            np.array([2 + 0j, 2, 2]),
            [0, 0, 0],
            prior_mode="none",
            isotonic_weights="equal",
        )
    with pytest.raises(ValueError, match="weights must be real"):
        select_iboin_mtd(
            design,
            [2, 2, 2],
            [0, 0, 0],
            prior_mode="none",
            isotonic_weights=np.array([1 + 0j, 1, 1]),
        )
    with pytest.raises(ValueError, match="must be real"):
        simulate_iboin_trial(
            design,
            np.array([0.1 + 0j, 0.1, 0.1]),
            [0.1, 0.1, 0.1],
            cohort_size=1,
            max_patients=3,
            prior_mode="none",
            isotonic_weights="equal",
            seed=1,
        )


def test_single_repetition_mean_mcse_is_undefined():
    design = IBOINDesign([0.1, 0.25, 0.4], [0, 0, 0], target=0.25)
    result = simulate_iboin(
        design,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        cohort_size=1,
        max_patients=3,
        repetitions=1,
        prior_mode="none",
        isotonic_weights="equal",
        seed=3,
        titration=False,
    )
    assert np.isnan(result.mean_patients_mcse).all()
    assert np.isnan(result.mean_dlt_mcse).all()
    assert np.isnan(result.mean_grade2_mcse).all()
    assert np.isnan(result.total_patients_mcse)
