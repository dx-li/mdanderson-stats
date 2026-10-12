"""Saved desktop comparator choices, preflighted jointly with primary simulation."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .boin import _owned
from .boin_desktop import _boolean
from .boin_desktop_batch import BOINDesktopBatch, _scenario_plan, simulate_boin_desktop_scenarios
from .boin_desktop_rolling import replay_boin_desktop_rolling_six
from .boin_desktop_simulation import BOINDesktopSimulation
from .boin_desktop_tite import desktop_tite_settings
from .three_plus_three import simulate_three_plus_three
from .toxicity_timing import toxicity_time_quantile


@dataclass(frozen=True)
class BOINDesktopWorkflow:
    primary: BOINDesktopBatch
    comparator: BOINDesktopBatch | None
    comparison_name: str | None
    expansion: bool


def _three_plus_three(probability, repetitions, cap, generator):
    finding = simulate_three_plus_three(
        probability, trials=repetitions, start_dose=1, rng=generator
    )
    n, y = finding.patients.copy(), finding.toxicities.copy()
    if cap is not None:
        for trial, dose in enumerate(finding.selected_dose):
            remaining = max(0, cap - int(n[trial].sum()))
            if dose and remaining:
                n[trial, dose - 1] += remaining
                y[trial, dose - 1] += generator.binomial(remaining, probability[dose - 1])
    return n, y, finding.selected_dose


def _comparator(study, scenario, repetitions, seed, profile, kind, expansion, gap_budget):
    probability = np.asarray(study.scenarios[scenario].probabilities)[:, 0]
    generator = np.random.default_rng(seed)
    reported = actual = pauses = unsafe_assignment = None
    unsafe = np.zeros(repetitions, dtype=bool)
    if kind == "3+3":
        n, y, chosen = _three_plus_three(
            probability,
            repetitions,
            study.maximum_patients if expansion else None,
            generator,
        )
        reasons = tuple("selected" if v else "no_selection" for v in chosen)
    else:
        timing = desktop_tite_settings(study)
        n = np.zeros((repetitions, study.shape[0]), dtype=np.int64)
        y = np.zeros_like(n)
        chosen = np.zeros(repetitions, dtype=np.int64)
        reported = np.zeros(repetitions)
        actual = np.zeros(repetitions)
        pauses = np.zeros(repetitions, dtype=np.int64)
        unsafe_assignment = np.zeros(repetitions, dtype=np.int64)
        for trial in range(repetitions):
            potential = toxicity_time_quantile(
                generator.random((6 * study.shape[0], study.shape[0])),
                probability[None, :],
                timing.window_months,
                distribution=timing.distribution,
                late_probability=timing.late_probability,
            )
            gaps = generator.exponential(1 / timing.accrual_per_month, size=gap_budget)
            try:
                result = replay_boin_desktop_rolling_six(
                    gaps,
                    potential,
                    timing.window_months,
                    cohort_size=study.cohort_size,
                    start_dose=study.start_dose[0],
                    profile=profile,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Rolling Six scenario {scenario}, seed {seed}, trial {trial}: {exc}"
                ) from exc
            n[trial], y[trial] = result.patients, result.toxicities
            chosen[trial] = result.selected_dose or 0
            reported[trial], actual[trial] = result.reported_final_time, result.actual_final_time
            pauses[trial] = result.suspension_checks
            unsafe[trial] = result.selected_eliminated
            unsafe_assignment[trial] = result.unsafe_assignments.sum()
        reasons = tuple("selected" if v else "no_selection" for v in chosen)
    selected = np.zeros((repetitions, study.shape[0], 1), dtype=bool)
    rows = np.flatnonzero(chosen > 0)
    selected[rows, chosen[rows] - 1, 0] = True
    n, y = n[:, :, None], y[:, :, None]
    return BOINDesktopSimulation(
        study,
        scenario,
        seed,
        profile,
        _owned(n),
        _owned(y),
        _owned(n),
        _owned(y),
        _owned(selected),
        _owned(unsafe),
        reasons,
        reported_duration=None if reported is None else _owned(reported),
        actual_duration=None if actual is None else _owned(actual),
        suspension_checks=None if pauses is None else _owned(pauses),
        future_stop=None if reported is None else _owned(np.zeros(repetitions, dtype=bool)),
        calendar_gap_budget=gap_budget if reported is not None else None,
        method=kind + (" with fixed-selection expansion" if expansion else ""),
        unsafe_assignment_count=None if unsafe_assignment is None else _owned(unsafe_assignment),
    )


def simulate_boin_desktop_workflow(
    study,
    *,
    scenario_indices=None,
    batches=1,
    trials=None,
    seed=None,
    profile="safe",
    max_calendar_gaps=1000,
) -> BOINDesktopWorkflow:
    """Run all requested saved scenarios and the mode's enabled native comparator.

    3+3 starts at dose one; optional expansion targets the planned maximum and
    never revises the dose-finding selection. Rolling Six has its own six-per-dose
    enrollment bound, not the BOIN planned maximum. Comparator streams use
    separately recorded seeds; no native RNG or paired-stream identity is claimed.
    """
    indices, repetitions, master, seeds = _scenario_plan(
        study,
        scenario_indices,
        batches,
        trials,
        seed,
        profile,
        max_calendar_gaps,
    )
    sim = ET.fromstring(study.xml).find("SimDat")
    assert sim is not None
    kind = None
    expansion = False
    if study.mode == "single" and _boolean(sim, "Simulate_3Plus3Design"):
        kind = "3+3"
        expansion = _boolean(sim, "Simulate_3Plus3Design_CohortExpansion")
    elif study.mode == "tite" and _boolean(sim, "Simulate_Rolling6Design"):
        kind = "Rolling Six"
        if study.cohort_size not in (1, 2, 3, 6):
            raise ValueError("Rolling Six cohort size must divide six")
    if kind is not None:
        comparator_maximum = max(6 * study.shape[0], study.maximum_patients if expansion else 0)
        runs = len(seeds)
        overhead = (
            sum(study.shape) + study.cohort_size
            if profile == "desktop_legacy" and study.titration
            else 0
        )
        if (
            repetitions * runs * (study.maximum_patients + overhead + comparator_maximum)
            > 1_000_000
        ):
            raise ValueError("joint comparison request exceeds 1000000 planned patient draws")
        if 2 * repetitions * runs * np.prod(study.shape) > 2_000_000:
            raise ValueError("joint comparison exceeds 2000000 retained trial-dose cells")
        if (
            kind == "Rolling Six"
            and 2 * repetitions * runs * max_calendar_gaps * study.shape[0] > 50_000_000
        ):
            raise ValueError("joint comparison exceeds 50000000 calendar dose-check cells")
    primary = simulate_boin_desktop_scenarios(
        study,
        scenario_indices=indices,
        batches=batches,
        trials=repetitions,
        seed=master,
        profile=profile,
        max_calendar_gaps=max_calendar_gaps,
    )
    comparator = None
    if kind is not None:
        comp_seed = (master + 2**63) % 2**64
        results = tuple(
            _comparator(
                study,
                scenario,
                repetitions,
                (comp_seed + position * batches + batch) % 2**64,
                profile,
                kind,
                expansion,
                max_calendar_gaps,
            )
            for position, scenario in enumerate(indices)
            for batch in range(batches)
        )
        comparator = BOINDesktopBatch(results, comp_seed)
    return BOINDesktopWorkflow(primary, comparator, kind, expansion)
