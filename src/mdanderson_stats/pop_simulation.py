"""Memory-bounded simulation for the PoP phase-I design."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import scalar
from .pop_design import PoPDesign, _allocation, _integer


@dataclass(frozen=True)
class PoPSimulation:
    selections: NDArray[np.int64]
    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    early_stop: NDArray[np.bool_]
    stop_reason: NDArray[np.str_]
    selection_probability: NDArray[np.float64]
    selection_mcse: NDArray[np.float64]
    mean_patients: NDArray[np.float64]
    mean_toxicities: NDArray[np.float64]
    risk_under: float
    risk_over: float
    early_stop_probability: float
    early_stop_mcse: float
    risk_under_mcse: float
    risk_over_mcse: float


def simulate_pop(
    design: PoPDesign,
    skeleton: ArrayLike,
    *,
    total_patients: int,
    cohort_size: int = 3,
    trials: int = 1000,
    start_dose: int = 1,
    titration: bool = True,
    earlyterm: bool = True,
    risk_cutoff: float = 0.8,
    seed: int | None = 123,
) -> PoPSimulation:
    """Simulate complete-outcome PoP trials without allocating potential outcomes."""
    if not isinstance(design, PoPDesign):
        raise ValueError("design must be a PoPDesign")
    p = np.asarray(skeleton, dtype=float)
    if not isinstance(titration, (bool, np.bool_)) or not isinstance(earlyterm, (bool, np.bool_)):
        raise ValueError("titration and earlyterm must be boolean")
    if (
        p.ndim != 1
        or p.size < 2
        or p.size > 100
        or np.any(~np.isfinite(p))
        or np.any((p < 0) | (p > 1))
        or np.any(np.diff(p) < 0)
    ):
        raise ValueError("skeleton must be a nondecreasing probability vector with 2..100 doses")
    total_patients = _integer(total_patients, "total_patients", 1, 1000)
    cohort_size = _integer(cohort_size, "cohort_size", 1, 4)
    trials = _integer(trials, "trials", 1, 100000)
    start_dose = _integer(start_dose, "start_dose", 1, p.size)
    risk_cutoff = scalar(risk_cutoff, "risk_cutoff")
    if not 0 <= risk_cutoff <= 1:
        raise ValueError("risk_cutoff must be in [0,1]")
    if trials * total_patients > 100000 or trials * p.size > 100000:
        raise ValueError("trials and allocation size exceed simulation limits")
    rng = np.random.default_rng(seed)
    boundaries = design.boundaries(total_patients, cohort_size=1)
    selections: NDArray[np.int64] = np.zeros(trials, dtype=np.int64)
    patient_counts = np.zeros((trials, p.size), dtype=np.int64)
    toxicity_counts = np.zeros_like(patient_counts)
    early: NDArray[np.bool_] = np.zeros(trials, dtype=bool)
    reasons: NDArray[np.str_] = np.full(trials, "completed", dtype="U16")
    true_mtd = int(np.argmin(np.abs(p - design.target)))

    for trial in range(trials):
        n = np.zeros(p.size, dtype=np.int64)
        y = np.zeros(p.size, dtype=np.int64)
        under = np.zeros(p.size, dtype=bool)
        over = np.zeros(p.size, dtype=bool)
        dose = int(start_dose)
        remaining = total_patients
        if titration:
            while remaining:
                dlt = int(rng.binomial(1, p[dose - 1]))
                n[dose - 1] += 1
                y[dose - 1] += dlt
                remaining -= 1
                if dlt or dose == p.size:
                    if dlt:
                        if y[dose - 1] <= boundaries.escalate_max[0]:
                            direction = 1
                            dose = max(1, min(p.size, dose + direction))
                        elif y[dose - 1] >= boundaries.deescalate_min[0]:
                            direction = -1
                            dose = max(1, min(p.size, dose + direction))
                    break
                dose += 1
        while remaining:
            take = min(cohort_size, remaining)
            dlt = int(rng.binomial(take, p[dose - 1]))
            n[dose - 1] += take
            y[dose - 1] += dlt
            remaining -= take
            j = dose - 1
            k = int(n[j]) - 1
            action, next_dose, under, over = _allocation(
                j,
                under,
                over,
                bool(earlyterm and y[j] <= boundaries.exclude_under_max[k]),
                bool(earlyterm and y[j] >= boundaries.exclude_over_min[k]),
                bool(y[j] <= boundaries.escalate_max[k]),
                bool(y[j] >= boundaries.deescalate_min[k]),
            )
            if action == "stop":
                early[trial] = True
                reasons[trial] = "all_excluded"
                break
            if next_dose is None:
                raise ArithmeticError("non-stopping PoP decision has no next dose")
            dose = next_dose
        selected = design.select_mtd(n, y).dose
        selections[trial] = 0 if selected is None else selected
        patient_counts[trial] = n
        toxicity_counts[trial] = y

    probs = np.bincount(selections, minlength=p.size + 1).astype(float) / trials
    mcse = np.sqrt(probs * (1 - probs) / trials)
    below = patient_counts[:, :true_mtd].sum(axis=1) if true_mtd else np.zeros(trials)
    above = (
        patient_counts[:, true_mtd + 1 :].sum(axis=1) if true_mtd + 1 < p.size else np.zeros(trials)
    )
    risk_under_value = float(np.mean(below > risk_cutoff * total_patients))
    risk_over_value = float(np.mean(above > risk_cutoff * total_patients))
    early_value = float(np.mean(early))
    for array in (selections, patient_counts, toxicity_counts, early, reasons, probs, mcse):
        array.setflags(write=False)
    mean_patients = patient_counts.mean(axis=0)
    mean_toxicities = toxicity_counts.mean(axis=0)
    mean_patients.setflags(write=False)
    mean_toxicities.setflags(write=False)
    return PoPSimulation(
        selections,
        patient_counts,
        toxicity_counts,
        early,
        reasons,
        probs,
        mcse,
        mean_patients,
        mean_toxicities,
        risk_under_value,
        risk_over_value,
        early_value,
        float(np.sqrt(early_value * (1 - early_value) / trials)),
        float(np.sqrt(risk_under_value * (1 - risk_under_value) / trials)),
        float(np.sqrt(risk_over_value * (1 - risk_over_value) / trials)),
    )
