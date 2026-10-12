"""Scenario summaries with explicit native truth definitions and actual counts."""

import math
from dataclasses import dataclass

import numpy as np

from .boin_desktop_tite import desktop_tite_settings


@dataclass(frozen=True)
class BOINDesktopScenarioSummary:
    trials: int
    mean_reported_patients: float
    mean_actual_patients: float
    mean_reported_toxicities: float
    mean_actual_toxicities: float
    no_selection_probability: float
    recorded_safety_stop_probability: float | None
    exact_target_selection_probability: float | None
    actual_fraction_at_exact_target: float | None
    native_console_fraction_at_exact_target: float | None
    closest_target_selection_probability: float | None
    actual_fraction_at_closest_target: float | None
    reference_contour: tuple[tuple[int, int], ...]
    native_contour_selection_probability: float | None
    native_reported_contour_regions: tuple[float, float, float] | None
    native_actual_contour_regions: tuple[float, float, float] | None
    actual_fraction_at_reference_points: float | None
    actual_incomparable_fraction: float | None
    conventional_duration_approximation: float | None
    qualifications: tuple[str, ...]


def _fraction(counts, mask):
    total = float(counts.sum())
    return float(counts[mask].sum() / total) if total else None


def _region_fractions(counts, masks):
    return (
        float(counts[masks[0]].sum() / counts.sum()),
        float(counts[masks[1]].sum() / counts.sum()),
        float(counts[masks[2]].sum() / counts.sum()),
    )


def _contour_regions(probability, target):
    reference = np.zeros(probability.shape, dtype=bool)
    below = np.zeros_like(reference)
    above = np.zeros_like(reference)
    for a, row in enumerate(probability):
        b = int(np.argmin(abs(row - target)))
        if row[b] <= target + 0.05:
            reference[a, b] = True
            below[: a + 1, : b + 1] = True
            above[a:, b:] = True
    below[reference] = above[reference] = False
    return reference, below, above


def summarize_boin_desktop(batch, scenario_index=0) -> BOINDesktopScenarioSummary:
    """Pool retained trials, distinguishing native display definitions from safety.

    Native single-MTD console truth is exact target equality, while returned R
    summaries use nearest target. Native waterfall residual regions can include
    incomparable combinations, so exact contour points are reported separately.
    No-selection frequency is not relabeled an observed safety-stop frequency.
    """
    runs = tuple(r for r in batch.results if r.scenario_index == scenario_index)
    if not runs:
        raise ValueError("scenario is not present in batch")
    first = runs[0]
    study = first.study
    if any(
        r.study != study or r.profile != first.profile or r.method != first.method for r in runs
    ):
        raise ValueError("summary requires matching study, method and profile")
    trials = sum(len(r.patients) for r in runs)
    n = sum(r.patients.sum(axis=0) for r in runs)
    y = sum(r.toxicities.sum(axis=0) for r in runs)
    real = sum(r.actual_patients.sum(axis=0) for r in runs)
    real_y = sum(r.actual_toxicities.sum(axis=0) for r in runs)
    selected = np.concatenate([r.selected for r in runs])
    probability = np.asarray(study.scenarios[scenario_index].probabilities)
    empty = float(np.mean(~selected.any(axis=(1, 2))))
    stops = (
        sum(
            reason.startswith(("stop_safety", "stop_extra_safe"))
            for r in runs
            for reason in r.stop_reasons
        )
        / trials
        if first.method == "BOIN"
        else None
    )
    exact_selection = exact_fraction = native_fraction = closest_selection = closest_fraction = None
    contour: tuple[tuple[int, int], ...] = ()
    contour_selection = regions = actual_regions = points = incomparable = None
    notes = []
    if study.mode == "combination":
        exact = probability == study.target
        closest = abs(probability - study.target) == np.min(abs(probability - study.target))
        exact_selection = float(selected[:, exact].sum() / trials)
        closest_selection = float(selected[:, closest].sum() / trials)
        exact_fraction = _fraction(real, exact)
        closest_fraction = _fraction(real, closest)
        if all(
            r.legacy_reporting_denominator is not None and r.legacy_reporting_denominator > 0
            for r in runs
        ):
            native_fraction = float(
                sum(r.patients[:, exact].sum() / r.legacy_reporting_denominator for r in runs)
                / trials
            )
        notes.append(
            "Native console uses exact target equality and its final mutable capacity; "
            "returned R summaries use nearest target. Actual fractions use actual enrollment."
        )
        if native_fraction is not None and native_fraction > 1:
            notes.append(
                "The defective native denominator yields a fraction above one; it is not clipped."
            )
    elif study.mode == "contour":
        reference, below, above = _contour_regions(probability, study.target)
        contour = tuple((int(a) + 1, int(b) + 1) for a, b in np.argwhere(reference))
        contour_selection = (
            float(np.mean(np.all(selected == reference, axis=(1, 2)))) if contour else 0.0
        )
        if not contour:
            notes.append(
                "No reference contour qualifies; native correct-selection is zero and "
                "geometric allocation fractions are undefined."
            )
        elif np.any(below & above):
            notes.append("Nonmonotone reference geometry overlaps; region fractions are undefined.")
        else:
            residual = ~(below | above)
            masks = (residual, above, below)
            if n.sum():
                regions = _region_fractions(n, masks)
            if real.sum():
                actual_regions = _region_fractions(real, masks)
            points = _fraction(real, reference)
            incomparable = _fraction(real, residual & ~reference)
        notes.append(
            "Reference contour takes the first nearest-target dose in each row, "
            "provided its probability is at most target+.05. Native residual allocation "
            "includes incomparable combinations; above is geometric, not a toxicity label."
        )
    duration = None
    if study.mode == "tite" and first.method == "BOIN":
        timing = desktop_tite_settings(study)
        duration = math.ceil(float(n.sum()) / trials / study.cohort_size) * (
            (study.cohort_size - 0.5) / timing.accrual_per_month + timing.window_months
        )
        if not math.isfinite(duration):
            raise ValueError("conventional duration approximation is not representable")
        notes.append(
            "Conventional duration is the native CRM/mTPI approximation, not a simulated "
            "calendar. The original help uses a different comparator label."
        )
    return BOINDesktopScenarioSummary(
        trials,
        float(n.sum() / trials),
        float(real.sum() / trials),
        float(y.sum() / trials),
        float(real_y.sum() / trials),
        empty,
        stops,
        exact_selection,
        exact_fraction,
        native_fraction,
        closest_selection,
        closest_fraction,
        contour,
        contour_selection,
        regions,
        actual_regions,
        points,
        incomparable,
        duration,
        tuple(notes),
    )
