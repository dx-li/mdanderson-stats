"""Independent complete-outcome desktop simulation kernel."""

from dataclasses import replace

import numpy as np
from scipy.special import betaincc


def _single_trials(
    design,
    probability,
    repetitions,
    cohorts,
    size,
    start,
    titration,
    cap,
    generator,
    profile="safe",
):
    maximum = cohorts * size
    levels = len(probability)
    patients = np.zeros((repetitions, levels), dtype=np.int64)
    toxicities = np.zeros_like(patients)
    selected = np.zeros(repetitions, dtype=np.int64)
    reasons = []
    titration = titration and size > 1
    top = min(cap + 1, levels)
    if titration and top < start:
        raise ValueError("desktop titration endpoint must not precede starting dose")
    for trial in range(repetitions):
        n, y = patients[trial], toxicities[trial]
        current = start
        first_topup = True
        excluded = np.zeros(levels, dtype=bool)
        reason = "maximum_patients"
        if titration:
            draws = generator.random(top - start + 1)
            for offset, uniform in enumerate(draws):
                current = start + offset
                n[current - 1] = 1
                y[current - 1] = uniform < probability[current - 1]
                if y[current - 1] or (profile == "safe" and n.sum() >= maximum):
                    break
        for _ in range(cohorts):
            if profile == "safe" and n.sum() >= maximum:
                break
            index = current - 1
            topup = titration and first_topup and n[index] < size
            draw_count = size - 1 if topup else size
            if topup:
                first_topup = False
            outcomes = generator.random(draw_count) < probability[index]
            remaining = maximum - int(n.sum())
            reached_capacity = not topup and 0 < remaining <= size
            allocated = (
                min(draw_count, max(0, remaining))
                if profile == "safe" or reached_capacity
                else draw_count
            )
            n[index] += allocated
            y[index] += int(outcomes[:allocated].sum())
            if reached_capacity:
                break
            analysis = design
            if profile == "desktop_legacy" and design.extra_safe:
                standard_boundary = design.boundary_table(int(n[index])).eliminate_min[-1]
                analysis = replace(design, extra_safe=False)
                if (
                    current == 1
                    and standard_boundary <= n[index]
                    and n[index] >= 3
                    and betaincc(y[index] + 1, n[index] - y[index] + 1, design.target)
                    > design.elimination_probability - design.safety_offset
                ):
                    reason = "stop_safety"
                    break
            decision = analysis.next_dose(n, y, current, eliminated=excluded)
            excluded = decision.eliminated
            if decision.next_dose is None:
                reason = decision.action
                break
            current = decision.next_dose
        final = design.select_mtd(n, y)
        selected[trial] = 0 if reason == "stop_safety" else (final.dose or 0)
        reasons.append(reason)
    return patients, toxicities, selected, tuple(reasons)


def _preflight(study, scenario_index, trials, seed, profile, max_calendar_gaps):
    from .boin_desktop_simulation import _validate_request
    from .boin_desktop_single import desktop_single_design
    from .boin_desktop_tite import desktop_tite_settings

    if study.mode not in ("single", "tite"):
        raise ValueError("univariate simulation requires a single-agent or TITE study")
    repetitions, outcome_seed = _validate_request(study, scenario_index, trials, seed, profile)
    design = desktop_single_design(study)
    if design.early_stop_patients is None or design.early_stop_patients < 6:
        raise ValueError("desktop simulation requires precision threshold of at least six")
    if (
        isinstance(max_calendar_gaps, bool)
        or not isinstance(max_calendar_gaps, int)
        or not 1 <= max_calendar_gaps <= 100000
    ):
        raise ValueError("max_calendar_gaps must be an integer in [1,100000]")
    if study.mode == "single":
        if max_calendar_gaps != 1000:
            raise ValueError("calendar gap settings require a TITE study")
        if study.titration and min(study.titration_cap + 1, study.shape[0]) < study.start_dose[0]:
            raise ValueError("desktop titration endpoint precedes the starting dose")
        return repetitions, outcome_seed, design, None
    if repetitions * max_calendar_gaps * study.shape[0] > 50_000_000:
        raise ValueError("request exceeds 50000000 bounded calendar dose-check cells")
    settings = desktop_tite_settings(study)
    probability = np.asarray(study.scenarios[scenario_index].probabilities)[:, 0]
    if settings.distribution != "uniform" and np.any(probability == 1):
        raise ValueError("nonuniform event timing requires scenario probabilities below one")
    return repetitions, outcome_seed, design, settings


def simulate_boin_desktop_univariate(
    study,
    *,
    scenario_index=0,
    trials=None,
    seed=None,
    profile="safe",
    max_calendar_gaps=1000,
):
    """Run current desktop single/TITE settings with explicit bounded calendars."""
    from .boin import _owned
    from .boin_desktop_simulation import BOINDesktopSimulation
    from .boin_desktop_tite_calendar import replay_boin_desktop_tite
    from .toxicity_timing import toxicity_time_quantile

    repetitions, outcome_seed, design, timing = _preflight(
        study,
        scenario_index,
        trials,
        seed,
        profile,
        max_calendar_gaps,
    )
    probability = np.asarray(study.scenarios[scenario_index].probabilities)[:, 0]
    generator = np.random.default_rng(outcome_seed)
    duration = actual_duration = checks = future = None
    if timing is None:
        n, y, chosen, reasons = _single_trials(
            design,
            probability,
            repetitions,
            study.maximum_patients // study.cohort_size,
            study.cohort_size,
            study.start_dose[0],
            study.titration,
            study.titration_cap,
            generator,
            profile,
        )
    else:
        n = np.zeros((repetitions, study.shape[0]), dtype=np.int64)
        y = np.zeros_like(n)
        chosen = np.zeros(repetitions, dtype=np.int64)
        duration = np.zeros(repetitions)
        actual_duration = np.zeros(repetitions)
        checks = np.zeros(repetitions, dtype=np.int64)
        future = np.zeros(repetitions, dtype=bool)
        stops = []
        for trial in range(repetitions):
            potential = toxicity_time_quantile(
                generator.random((study.maximum_patients, study.shape[0])),
                probability[None, :],
                timing.window_months,
                distribution=timing.distribution,
                late_probability=timing.late_probability,
            )
            gaps = generator.exponential(1 / timing.accrual_per_month, size=max_calendar_gaps)
            result = replay_boin_desktop_tite(
                design,
                gaps,
                potential,
                timing.window_months,
                cohort_size=study.cohort_size,
                start_dose=study.start_dose[0],
                max_pending_fraction=timing.max_pending_fraction,
                trimester_probabilities=timing.trimester_weights,
                profile=profile,
            )
            n[trial], y[trial] = result.patients, result.toxicities
            chosen[trial] = result.selected_dose or 0
            duration[trial] = result.final_time
            actual_duration[trial] = max(
                result.final_time, float(result.decision_times.max(initial=0))
            )
            checks[trial] = result.extra_checks
            future[trial] = result.used_future_stop
            stops.append(result.stop_reason)
        reasons = tuple(stops)
    selected = np.zeros((repetitions, study.shape[0], 1), dtype=bool)
    rows = np.flatnonzero(chosen > 0)
    selected[rows, chosen[rows] - 1, 0] = True
    n, y = n[:, :, None], y[:, :, None]
    return BOINDesktopSimulation(
        study,
        scenario_index,
        outcome_seed,
        profile,
        _owned(n),
        _owned(y),
        _owned(n),
        _owned(y),
        _owned(selected),
        _owned(np.zeros(repetitions, dtype=bool)),
        reasons,
        reported_duration=None if duration is None else _owned(duration),
        actual_duration=None if actual_duration is None else _owned(actual_duration),
        suspension_checks=None if checks is None else _owned(checks),
        future_stop=None if future is None else _owned(future),
        calendar_gap_budget=max_calendar_gaps if timing is not None else None,
    )
