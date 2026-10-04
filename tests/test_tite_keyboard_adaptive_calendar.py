import numpy as np

from mdanderson_stats import KeyboardDesign
from mdanderson_stats.tite_keyboard_adaptive_calendar import TITEKeyboardAdaptiveSettings
from mdanderson_stats.tite_keyboard_simulation import simulate_tite_keyboard
from mdanderson_stats.tite_keyboard_trial import (
    TITEKeyboardAdaptiveTrial,
    run_tite_keyboard_trial,
)


def _settings(*, max_total_work: int = 100_000_000) -> TITEKeyboardAdaptiveSettings:
    # Small relaxed thresholds make these wiring tests fast; not a recommended fit setting.
    return TITEKeyboardAdaptiveSettings(
        lambda_prior=(0.5, 0.5),
        gamma_prior=(0.5, 0.5),
        chains=2,
        draws=64,
        warmup=16,
        max_fit_work=5_000_000,
        max_total_work=max_total_work,
        max_split_rhat=100.0,
        max_weight_mcse=1.0,
    )


def test_trial_uses_only_observed_event_ages_and_counts_each_fit_work() -> None:
    def replay(first_delay: float) -> TITEKeyboardAdaptiveTrial:
        delays = np.full((6, 2), np.inf)
        delays[0, 0] = first_delay
        result = run_tite_keyboard_trial(
            KeyboardDesign(),
            [1.0] * 6,
            delays,
            10.0,
            cohort_size=1,
            start_dose=1,
            pending_fraction_limit=None,
            adaptive_timing=_settings(),
            adaptive_rng=np.random.default_rng(72),
        )
        assert isinstance(result, TITEKeyboardAdaptiveTrial)
        return result

    early_event = replay(6.0)
    later_event = replay(8.0)
    early_steps = [step for step in early_event.steps if step.time < 7.0]
    later_steps = [step for step in later_event.steps if step.time < 7.0]
    assert early_steps and len(early_steps) == len(later_steps)
    for left, right in zip(early_steps, later_steps, strict=True):
        assert (left.decision.action, left.decision.next_dose) == (
            right.decision.action,
            right.decision.next_dose,
        )
        assert left.adaptive_fit is not None and right.adaptive_fit is not None
        np.testing.assert_array_equal(
            left.adaptive_fit.log_shape_mean, right.adaptive_fit.log_shape_mean
        )
        np.testing.assert_array_equal(
            left.adaptive_fit.log_shape_mcse, right.adaptive_fit.log_shape_mcse
        )
        assert left.adaptive_fit.observed_dlt_count == right.adaptive_fit.observed_dlt_count == 0
    early_fits = [step for step in early_event.steps if step.adaptive_fit is not None]
    later_fits = [step for step in later_event.steps if step.adaptive_fit is not None]
    assert any(
        step.time >= 7.0 and step.adaptive_fit.observed_dlt_count == 1 for step in early_fits
    )
    assert not any(step.time < 7.0 and step.adaptive_fit.observed_dlt_count for step in later_fits)
    assert early_event.adaptive_fit_count == len(early_fits)
    assert early_event.adaptive_work_units == sum(
        step.adaptive_fit.work_units for step in early_fits
    )
    assert all(step.adaptive_fit.diagnostics_passed for step in early_fits + later_fits)


def test_simulation_separates_and_records_replayable_random_streams() -> None:
    settings = _settings()
    first = simulate_tite_keyboard(
        KeyboardDesign(),
        [0.1, 0.2],
        10.0,
        0.3,
        cohorts=2,
        cohort_size=3,
        trials=2,
        pending_fraction_limit=None,
        rng=123,
        adaptive_timing=settings,
    )
    second = simulate_tite_keyboard(
        KeyboardDesign(),
        [0.1, 0.2],
        10.0,
        0.3,
        cohorts=2,
        cohort_size=3,
        trials=2,
        pending_fraction_limit=None,
        rng=123,
        adaptive_timing=settings,
    )
    np.testing.assert_array_equal(first.patients, second.patients)
    np.testing.assert_array_equal(first.toxicities, second.toxicities)
    np.testing.assert_array_equal(first.selected_dose, second.selected_dose)
    assert first.adaptive_fit_count > 0
    assert first.adaptive_fit_count >= 2
    assert first.adaptive_fit_count == first.adaptive_diagnostics_passed
    assert first.adaptive_work_units == second.adaptive_work_units
    assert first.adaptive_work_units <= settings.max_total_work
    assert (first.outcome_seed, first.sampler_seed) == (second.outcome_seed, second.sampler_seed)
    assert first.outcome_seed != first.sampler_seed

    cumulative_limit = _settings(max_total_work=first.adaptive_work_units - 1)
    try:
        simulate_tite_keyboard(
            KeyboardDesign(),
            [0.1, 0.2],
            10.0,
            0.3,
            cohorts=2,
            cohort_size=3,
            trials=2,
            pending_fraction_limit=None,
            rng=123,
            adaptive_timing=cumulative_limit,
        )
    except ValueError as error:
        assert "max_work" in str(error)
    else:
        raise AssertionError("a one-unit-short aggregate budget must fail before the final fit")


def test_aggregate_budget_is_shared_across_simulated_trials() -> None:
    settings = _settings(max_total_work=1)
    try:
        simulate_tite_keyboard(
            KeyboardDesign(),
            [0.1, 0.2],
            10.0,
            5.0,
            cohorts=2,
            cohort_size=3,
            trials=2,
            pending_fraction_limit=None,
            rng=5,
            adaptive_timing=settings,
        )
    except ValueError as error:
        assert "max_work" in str(error)
    else:
        raise AssertionError(
            "the first fit must fail its preflight under an insufficient aggregate budget"
        )
