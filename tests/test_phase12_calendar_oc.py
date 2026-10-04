import numpy as np
import pytest

from mdanderson_stats.parallel_phase12_calendar import simulate_phase12_calendar
from mdanderson_stats.parallel_phase12_model import phase12_snapshot
from mdanderson_stats.phase12_calendar_oc import (
    _source_duration_summary,
    simulate_phase12_calendar_oc,
)


def test_seeded_aggregate_matches_independent_calendar_replays():
    result = simulate_phase12_calendar_oc(
        [0.0] * 6,
        [0.0] * 6,
        n_trials=2,
        seed=12031,
        max_patients=24,
        max_attempts=100,
        draws=8,
        warmup=0,
        chains=2,
        efficacy_window=84,
        toxicity_window=28,
    )
    assert result.per_trial_seeds.tolist() == [
        int(np.random.SeedSequence(12031, spawn_key=(i,)).generate_state(1, dtype=np.uint64)[0])
        for i in range(2)
    ]
    assert not result.per_trial_seeds.flags.writeable

    replay_selections = []
    replay_durations_months = []
    replay_enrollment = np.zeros(6, dtype=np.int64)
    replay_tox = np.zeros(6, dtype=np.int64)
    replay_response = np.zeros(6, dtype=np.int64)
    replay_phase_one_treated = np.zeros(6, dtype=np.int64)
    replay_phase_one_tox = np.zeros(6, dtype=np.int64)
    replay_phase_one_admissible = np.zeros(6, dtype=np.int64)
    for trial_seed in result.per_trial_seeds:
        trial = simulate_phase12_calendar(
            [0.0] * 6,
            [0.0] * 6,
            max_patients=24,
            max_attempts=100,
            draws=8,
            warmup=0,
            chains=2,
            efficacy_window=84,
            toxicity_window=28,
            rng=np.random.default_rng(int(trial_seed)),
        )
        replay_selections.append(
            trial.early_selected if trial.early_selected is not None else trial.future_selected
        )
        replay_durations_months.append(trial.stop_time * (12.0 / 365.0))
        for row in trial.records:
            dose = int(row[0])
            replay_enrollment[dose] += 1
            replay_response[dose] += int(row[2])
            replay_tox[dose] += int(row[4])
        phase_one_records = trial.records[trial.phases == 0]
        for row in phase_one_records:
            replay_phase_one_treated[int(row[0])] += 1
            replay_phase_one_tox[int(row[0])] += int(row[4])
        replay_phase_one_admissible += trial.phase_one_admissible

    expected_selected = np.zeros(6, dtype=np.int64)
    for dose in replay_selections:
        if dose is not None:
            expected_selected[dose] += 1
    np.testing.assert_array_equal(result.selected_counts, expected_selected)
    np.testing.assert_array_equal(result.treated_total, replay_enrollment)
    np.testing.assert_array_equal(result.generated_response_total, replay_response)
    np.testing.assert_array_equal(result.generated_toxicity_total, replay_tox)
    assert result.phase_one_tally_count == 2
    assert result.phase_one_tally_probability == 1.0
    np.testing.assert_allclose(result.phase_one_tally_mean_treated, replay_phase_one_treated / 2)
    np.testing.assert_allclose(result.phase_one_tally_mean_toxicities, replay_phase_one_tox / 2)
    np.testing.assert_allclose(
        result.phase_one_tally_mean_admissible, replay_phase_one_admissible / 2
    )
    assert result.selection_probability.sum() + result.no_selection_probability == pytest.approx(1)
    assert not np.isnan(result.mcse_total_enrollment)
    duration_months = np.asarray(replay_durations_months)
    np.testing.assert_allclose(result.duration_mean_months, duration_months.mean())
    np.testing.assert_allclose(
        result.duration_population_variance_months_squared, duration_months.var(ddof=0)
    )
    assert result.duration_order_indices.tolist() == [0, 0, 0, 1, 2, 2, 2]
    expected = np.sort(duration_months)[[0, 0, 0, 1]]
    np.testing.assert_allclose(result.duration_order_statistics_months[:4], expected)
    assert np.isnan(result.duration_order_statistics_months[4:]).all()
    assert not result.duration_order_indices.flags.writeable
    assert not result.duration_order_statistics_months.flags.writeable


def test_observed_endpoint_denominators_are_distinct_and_one_trial_mcse_is_undefined():
    result = simulate_phase12_calendar_oc(
        [0.0] * 6,
        [0.0] * 6,
        n_trials=1,
        seed=3,
        max_patients=18,
        max_attempts=100,
        draws=8,
        warmup=0,
        chains=2,
        efficacy_window=84,
        toxicity_window=28,
    )
    assert result.no_selection_count == 1
    assert result.no_selection_probability == 1.0
    assert np.isnan(result.no_selection_mcse)
    assert not np.isnan(result.mean_treated).any()
    # The all-zero toxicity/response design reaches a stop with outcomes still
    # pending; observed denominators therefore differ from complete generated N.
    assert int(result.observed_response_denominator.sum()) < int(result.total_enrollment)
    assert int(result.observed_toxicity_denominator.sum()) <= int(result.total_enrollment)
    assert result.mcmc_fit_count == 1
    assert result.max_mcmc_split_rhat is not None


def test_source_duration_summary_is_stable_for_large_constant_times():
    mean, variance, indices, order_statistics = _source_duration_summary(np.asarray([1e300, 1e300]))
    expected = 1e300 * (12.0 / 365.0)
    assert mean == pytest.approx(expected)
    assert variance == 0.0
    assert indices.tolist() == [0, 0, 0, 1, 2, 2, 2]
    np.testing.assert_allclose(order_statistics[:4], expected)
    assert np.isnan(order_statistics[4:]).all()


def test_native_phase_one_tally_matches_completed_toxic_stop_and_omits_incomplete_path():
    stopped = simulate_phase12_calendar_oc(
        [1.0] * 6,
        [0.0] * 6,
        n_trials=2,
        seed=8551,
        max_patients=18,
        max_attempts=100,
        draws=8,
        warmup=0,
        chains=2,
    )
    assert stopped.phase_one_tally_count == 2
    assert stopped.phase_one_toxic_stop_count == 2
    # With this arrival seed, one path's first two toxicities are observed
    # before patient three arrives, so the source closes at two patients.
    assert stopped.phase_one_tally_mean_patients == 2.5
    np.testing.assert_array_equal(stopped.phase_one_tally_mean_treated, [2.5, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(stopped.phase_one_tally_mean_toxicities, [2.5, 0, 0, 0, 0, 0])
    assert stopped.phase_one_tally_mean_admissible.sum() == 0
    assert stopped.phase_one_tally_mcse_patients == 0.5
    assert not stopped.phase_one_tally_mean_treated.flags.writeable
    assert not stopped.phase_one_tally_mcse_admissible.flags.writeable

    interrupted = simulate_phase12_calendar_oc(
        [0.0] * 6,
        [0.0] * 6,
        n_trials=2,
        seed=8552,
        max_patients=1,
        max_attempts=100,
        draws=8,
        warmup=0,
        chains=2,
    )
    assert interrupted.phase_one_tally_count == 0
    assert interrupted.phase_one_toxic_stop_count == 0
    assert interrupted.phase_one_tally_mean_patients == 0
    np.testing.assert_array_equal(interrupted.phase_one_tally_mean_treated, np.zeros(6))


def test_observed_count_and_pooled_rate_mcse_match_replayed_trial_ledgers():
    toxicity = [0.0] * 6
    efficacy = [0.1, 0.25, 0.4, 0.55, 0.7, 0.8]
    kwargs = dict(
        max_patients=24,
        max_attempts=200,
        draws=8,
        warmup=0,
        chains=2,
        efficacy_window=84,
        toxicity_window=28,
    )
    result = simulate_phase12_calendar_oc(
        toxicity,
        efficacy,
        n_trials=3,
        seed=82501,
        **kwargs,
    )
    per_trial_events = []
    per_trial_denominators = []
    for trial_seed in result.per_trial_seeds:
        trial = simulate_phase12_calendar(
            toxicity,
            efficacy,
            **kwargs,
            rng=np.random.default_rng(int(trial_seed)),
        )
        cutoff = (
            trial.final_analysis_time if trial.final_analysis_time is not None else trial.stop_time
        )
        tally = phase12_snapshot(trial.records, time=cutoff).tally
        per_trial_events.append(tally[:, 1].astype(int))
        per_trial_denominators.append(tally[:, :2].sum(axis=1).astype(int))

    events = np.asarray(per_trial_events)
    denominators = np.asarray(per_trial_denominators)
    assert int(events.sum()) > 0
    np.testing.assert_array_equal(result.observed_response_total, events.sum(axis=0))
    np.testing.assert_array_equal(result.observed_response_denominator, denominators.sum(axis=0))
    np.testing.assert_allclose(result.mean_observed_responses, events.mean(axis=0))
    np.testing.assert_allclose(
        result.mcse_observed_responses,
        events.std(axis=0, ddof=1) / np.sqrt(events.shape[0]),
    )
    pooled = np.divide(
        events.sum(axis=0),
        denominators.sum(axis=0),
        out=np.full(6, np.nan),
        where=denominators.sum(axis=0) > 0,
    )
    residuals = events - pooled[None, :] * denominators
    direct_mcse = np.sqrt(
        events.shape[0] / (events.shape[0] - 1) * np.sum(residuals**2, axis=0)
    ) / denominators.sum(axis=0)
    np.testing.assert_allclose(result.observed_response_rate, pooled, equal_nan=True)
    np.testing.assert_allclose(result.observed_response_rate_mcse, direct_mcse, equal_nan=True)


def test_importance_backend_reports_real_interim_fits_and_preflight():
    result = simulate_phase12_calendar_oc(
        [0.0] * 6,
        [0.0] * 6,
        n_trials=1,
        seed=9,
        max_patients=24,
        max_attempts=100,
        posterior_backend="importance",
        importance_max_integrations=100,
        importance_max_mode_iterations=100,
    )
    assert result.importance_fit_count >= 1
    assert result.mcmc_fit_count == 0
    assert result.importance_component_evaluations > 0
    assert result.max_importance_ratio_mcse is not None
    assert result.mean_final_analysis_time_days is not None
    assert result.final_analysis_trial_count == 1

    with pytest.raises(ValueError, match="worst-case importance component"):
        simulate_phase12_calendar_oc(
            [0.0] * 6,
            [0.0] * 6,
            n_trials=2,
            seed=9,
            max_patients=24,
            posterior_backend="importance",
            importance_max_integrations=100,
            max_total_importance_component_evaluations=100,
        )
