"""Bounded saved-study BOIN combination simulation and explicit legacy profiles."""

from dataclasses import dataclass, replace

import numpy as np

from .boin import _owned
from .boin_desktop import BOINDesktopStudy
from .boin_desktop_controller import desktop_combination_design
from .boin_desktop_decision import BOINDesktopCombDesign
from .boin_desktop_selection import select_boin_desktop_mtd
from .boin_desktop_single import desktop_single_design
from .boin_waterfall_trial import _final_contour, run_boin_waterfall_trial


@dataclass(frozen=True)
class BOINDesktopSimulation:
    study: BOINDesktopStudy
    scenario_index: int
    seed: int
    profile: str
    patients: np.ndarray
    toxicities: np.ndarray
    actual_patients: np.ndarray
    actual_toxicities: np.ndarray
    selected: np.ndarray
    selected_eliminated: np.ndarray
    stop_reasons: tuple[str, ...]
    reported_duration: np.ndarray | None = None
    actual_duration: np.ndarray | None = None
    suspension_checks: np.ndarray | None = None
    future_stop: np.ndarray | None = None
    calendar_gap_budget: int | None = None
    method: str = "BOIN"
    unsafe_assignment_count: np.ndarray | None = None

    @property
    def selection_probability(self) -> np.ndarray:
        return _owned(self.selected.mean(axis=0))

    @property
    def selection_mcse(self) -> np.ndarray:
        p = self.selection_probability
        return _owned(np.sqrt(p * (1 - p) / self.selected.shape[0]))

    @property
    def mean_patients(self) -> np.ndarray:
        return _owned(self.patients.mean(axis=0))

    @property
    def mean_toxicities(self) -> np.ndarray:
        return _owned(self.toxicities.mean(axis=0))


@dataclass(frozen=True)
class _WaterfallSimulationDesign(BOINDesktopCombDesign):
    def boundary_table(self, max_patients=150):
        table = super().boundary_table(max_patients)
        if self.run_in:
            table = replace(
                table, deescalate_min=_owned(np.where(table.patients == 6, 2, table.deescalate_min))
            )
        return table


def _single_trials(study, probability, repetitions, generator, profile):
    design = desktop_combination_design(study)
    shape = (repetitions, *study.shape)
    patients = np.zeros(shape, dtype=np.int64)
    toxicities = np.zeros_like(patients)
    actual = np.zeros_like(patients)
    selected = np.zeros(shape, dtype=bool)
    unsafe = np.zeros(repetitions, dtype=bool)
    reasons = []
    # Only the explicit legacy profile shares these defective native variables.
    legacy_first_topup = True
    legacy_maximum = study.maximum_patients
    size = study.cohort_size
    cohort_count = (study.maximum_patients + size - 1) // size
    if profile == "desktop_legacy":
        cohort_count = study.maximum_patients // size
    for trial in range(repetitions):
        n, y, real = patients[trial], toxicities[trial], actual[trial]
        dose = study.start_dose
        first_topup = True
        reason = "maximum_patients"
        if study.titration:
            cap = min(study.titration_cap, sum(study.shape) - 2)
            for move in range(cap + 1):
                a, b = dose[0] - 1, dose[1] - 1
                event = int(generator.random() < probability[a, b])
                n[a, b] += 1
                real[a, b] += 1
                y[a, b] += event
                if event or dose == study.shape or move == cap:
                    break
                if profile == "safe" and n.sum() >= study.maximum_patients:
                    break
                if a + 1 < study.shape[0] and b + 1 < study.shape[1]:
                    axis = int(generator.binomial(1, 0.5))
                    dose = (dose[0] + (axis == 0), dose[1] + (axis == 1))
                elif a + 1 < study.shape[0]:
                    dose = (dose[0] + 1, dose[1])
                else:
                    dose = (dose[0], dose[1] + 1)
            if profile == "desktop_legacy" and y.sum() == 0:
                dose = study.shape
        for _ in range(cohort_count):
            a, b = dose[0] - 1, dose[1] - 1
            if profile == "safe":
                remaining = study.maximum_patients - int(n.sum())
                if remaining <= 0:
                    break
                enrolled = min(
                    size - int(n[a, b]) if study.titration and first_topup else size, remaining
                )
                enrolled = max(0, enrolled)
                draws = generator.random(enrolled) < probability[a, b]
                n[a, b] += enrolled
                real[a, b] += enrolled
                y[a, b] += int(np.sum(draws))
                first_topup = False
            elif study.titration and legacy_first_topup and n[a, b] < size:
                legacy_first_topup = False
                draws = generator.random(size - 1) < probability[a, b]
                y[a, b] += int(np.sum(draws))
                n[a, b] += size
                real[a, b] += size - 1
            else:
                draws = generator.random(size) < probability[a, b]
                if study.titration and n.sum() < legacy_maximum <= n.sum() + size:
                    legacy_maximum -= int(n.sum())
                    enrolled = legacy_maximum
                    y[a, b] += int(np.sum(draws[:enrolled]))
                    n[a, b] += enrolled
                    real[a, b] += enrolled
                    reason = "legacy_capacity_reassignment"
                    break
                y[a, b] += int(np.sum(draws))
                n[a, b] += size
                real[a, b] += size
            decision = design.desktop_next_dose(n, y, dose, rng=generator)
            if decision.next_dose is None:
                reason = decision.action
                break
            dose = decision.next_dose
        final = select_boin_desktop_mtd(design, n, y, profile=profile, intervals=False)
        if reason == "stop_safety":
            final_dose = None
        else:
            final_dose = final.selection.dose
        if final_dose is not None:
            selected[trial, final_dose[0] - 1, final_dose[1] - 1] = True
            unsafe[trial] = final.selected_eliminated
        reasons.append(reason)
    return patients, toxicities, actual, toxicities.copy(), selected, unsafe, tuple(reasons)


def _contour_trials(study, probability, repetitions, generator, profile):
    source = desktop_combination_design(study)
    design = _WaterfallSimulationDesign(
        target=source.target,
        safe_probability=source.safe_probability,
        toxic_probability=source.toxic_probability,
        elimination_probability=source.elimination_probability,
        extra_safe=source.extra_safe,
        safety_offset=source.safety_offset,
        early_stop_patients=source.early_stop_patients,
        run_in=source.run_in,
    )
    patients = np.zeros((repetitions, *study.shape), dtype=np.int64)
    toxicities = np.zeros_like(patients)
    selected = np.zeros(patients.shape, dtype=bool)
    reasons = []
    actual = np.zeros_like(patients)
    actual_tox = np.zeros_like(patients)
    unsafe = np.zeros(repetitions, dtype=bool)
    cap = min(study.titration_cap, sum(study.shape) - 2)
    prelude = cap + 1 if study.titration and study.cohort_size > 1 else 0
    maximum = sum(study.subtrial_maximums)
    for trial in range(repetitions):
        tape = generator.random(maximum + max(0, prelude - 1))
        result = run_boin_waterfall_trial(
            design,
            probability,
            np.array(study.subtrial_maximums) // study.cohort_size,
            tape,
            cohort_size=study.cohort_size,
            early_stop_patients=int(design.early_stop_patients or 100),
            titration=study.titration,
            max_titration_escalations=cap if study.titration else None,
            extra_safe_profile="native_boundary_gate" if profile == "desktop_legacy" else "always",
            limit_total_patients=profile == "safe",
        )
        actual[trial], actual_tox[trial] = result.patients, result.toxicities
        if profile == "desktop_legacy":
            assert result.legacy_reported_patients is not None
            assert result.legacy_reported_toxicities is not None
            assert result.legacy_reported_eliminated is not None
            patients[trial] = result.legacy_reported_patients
            toxicities[trial] = result.legacy_reported_toxicities
            _, contour, _, _, _ = _final_contour(
                design, patients[trial], toxicities[trial], result.legacy_reported_eliminated
            )
            unsafe[trial] = any(
                dose is not None and result.legacy_reported_eliminated[dose[0] - 1, dose[1] - 1]
                for dose in contour
            )
        else:
            patients[trial] = result.patients
            toxicities[trial] = result.toxicities
            contour = result.selected_contour
        for dose in contour:
            if dose is not None:
                selected[trial, dose[0] - 1, dose[1] - 1] = True
        reasons.append(result.stop_reason)
    return (
        patients,
        toxicities,
        actual,
        actual_tox,
        selected,
        unsafe,
        tuple(reasons),
    )


def _validate_request(study, scenario_index, trials, seed, profile):
    if study.mode not in ("combination", "contour", "single", "tite") or study.patients:
        raise ValueError("simulation requires an empty supported desktop study")
    if profile not in ("safe", "desktop_legacy"):
        raise ValueError("profile must be safe or desktop_legacy")
    if (
        isinstance(scenario_index, bool)
        or not isinstance(scenario_index, int)
        or not 0 <= scenario_index < len(study.scenarios)
    ):
        raise ValueError("scenario_index must identify a saved scenario")
    repetitions = study.repetitions if trials is None else trials
    outcome_seed = study.seed if seed is None else seed
    for value, name, maximum in ((repetitions, "trials", 10000), (outcome_seed, "seed", 2**64 - 1)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not 0 <= value <= maximum
        ):
            raise ValueError(f"{name} must be an integer in [0,{maximum}]")
    if repetitions < 1 or repetitions * np.prod(study.shape) > 2_000_000:
        raise ValueError("require positive trials and at most 2000000 retained trial-dose cells")
    maximum = sum(study.subtrial_maximums) if study.mode == "contour" else study.maximum_patients
    if (
        maximum < 1
        or maximum
        + (
            sum(study.shape) + study.cohort_size
            if profile == "desktop_legacy" and study.titration
            else 0
        )
        > 1000
    ):
        raise ValueError("planned enrollment plus legacy overhead exceeds the 1000-patient bound")
    if (
        profile == "desktop_legacy"
        and study.mode == "combination"
        and (
            study.maximum_patients < study.cohort_size or study.maximum_patients % study.cohort_size
        )
    ):
        raise ValueError("legacy simulation requires a positive whole-cohort maximum")
    if repetitions * maximum > 1_000_000:
        raise ValueError("request exceeds 1000000 planned patient draws")
    if study.mode in ("single", "tite"):
        if study.maximum_patients % study.cohort_size:
            raise ValueError("single-agent simulation requires a whole-cohort planned maximum")
        desktop_single_design(study)
    else:
        desktop_combination_design(study)
    return int(repetitions), int(outcome_seed)


def simulate_boin_desktop_combination(
    study: BOINDesktopStudy,
    *,
    scenario_index: int = 0,
    trials: int | None = None,
    seed: int | None = None,
    profile: str = "safe",
) -> BOINDesktopSimulation:
    """Run one saved combination scenario with bounded output and explicit profiles.

    Safe simulation preserves assigned-patient counts and planned total capacity.
    Legacy single-MTD simulation reproduces the endpoint jump, phantom top-up and
    cross-replication state errors, retaining actual counts separately. Legacy
    waterfall retains source overhead and extra-safe gating. These profiles are
    comparisons, not interchangeable operating-characteristic estimates.
    """
    if study.mode not in ("combination", "contour"):
        raise ValueError("combination simulation requires a combination/contour study")
    repetitions, outcome_seed = _validate_request(study, scenario_index, trials, seed, profile)
    probability = np.asarray(study.scenarios[scenario_index].probabilities)
    generator = np.random.default_rng(outcome_seed)
    run = _contour_trials if study.mode == "contour" else _single_trials
    n, y, actual, actual_tox, selected, unsafe, reasons = run(
        study, probability, int(repetitions), generator, profile
    )
    return BOINDesktopSimulation(
        study,
        scenario_index,
        int(outcome_seed),
        profile,
        _owned(n),
        _owned(y),
        _owned(actual),
        _owned(actual_tox),
        _owned(selected),
        _owned(unsafe),
        reasons,
    )


def simulate_boin_desktop_study(
    study: BOINDesktopStudy,
    *,
    scenario_index: int = 0,
    trials: int | None = None,
    seed: int | None = None,
    profile: str = "safe",
    max_calendar_gaps: int = 1000,
) -> BOINDesktopSimulation:
    """Dispatch saved single-agent, TITE, combination or contour simulation.

    The calendar gap budget is explicit; exhaustion raises instead of dropping
    trials or silently changing the arrival model. Native RNG identity is not
    claimed. Comparison designs are handled separately.
    """
    if study.mode in ("single", "tite"):
        from .boin_desktop_univariate_simulation import simulate_boin_desktop_univariate

        return simulate_boin_desktop_univariate(
            study,
            scenario_index=scenario_index,
            trials=trials,
            seed=seed,
            profile=profile,
            max_calendar_gaps=max_calendar_gaps,
        )
    if max_calendar_gaps != 1000:
        raise ValueError("calendar gap settings require a TITE study")
    return simulate_boin_desktop_combination(
        study,
        scenario_index=scenario_index,
        trials=trials,
        seed=seed,
        profile=profile,
    )
