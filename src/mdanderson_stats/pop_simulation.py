"""Memory-bounded simulation for the PoP phase-I design."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .pop_design import PoPDesign, predictive_bayes_factor


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


def simulate_pop(
    design: PoPDesign,
    skeleton: NDArray[np.float64],
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
    if int(total_patients) != total_patients or not 1 <= total_patients <= 1000:
        raise ValueError("total_patients must be in [1,1000]")
    if int(cohort_size) != cohort_size or not 1 <= cohort_size <= 4:
        raise ValueError("cohort_size must be in [1,4]")
    if (
        int(trials) != trials
        or not 1 <= trials
        or trials * total_patients > 100000
        or trials * p.size > 100000
    ):
        raise ValueError("trials and allocation size exceed simulation limits")
    if not 0 <= risk_cutoff <= 1 or int(start_dose) != start_dose or not 1 <= start_dose <= p.size:
        raise ValueError("invalid start_dose or risk_cutoff")
    trials, total_patients, cohort_size = int(trials), int(total_patients), int(cohort_size)
    rng = np.random.default_rng(seed)
    selections = np.zeros(trials, dtype=np.int64)
    patient_counts = np.zeros((trials, p.size), dtype=np.int64)
    toxicity_counts = np.zeros_like(patient_counts)
    early = np.zeros(trials, dtype=bool)
    reasons = np.full(trials, "completed", dtype="U16")
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
                        rate = y[dose - 1] / n[dose - 1]
                        bf = float(predictive_bayes_factor(design.target, n[dose - 1], y[dose - 1]))
                        if bf < design.cutoff:
                            direction = 1 if rate < design.target else -1
                            dose = max(1, min(p.size, dose + direction))
                    break
                dose += 1
        while remaining:
            take = min(cohort_size, remaining)
            dlt = int(rng.binomial(take, p[dose - 1]))
            n[dose - 1] += take
            y[dose - 1] += dlt
            remaining -= take
            decision = design.decision(
                dose, n, y, excluded_under=under, excluded_over=over, earlyterm=earlyterm
            )
            under, over = decision.excluded_under, decision.excluded_over
            if decision.action == "stop_safety":
                early[trial] = True
                reasons[trial] = "all_excluded"
                break
            if decision.next_dose is None:
                raise RuntimeError("non-stopping PoP decision has no next dose")
            dose = decision.next_dose
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
    return PoPSimulation(
        selections,
        patient_counts,
        toxicity_counts,
        early,
        reasons,
        probs,
        mcse,
        patient_counts.mean(axis=0),
        toxicity_counts.mean(axis=0),
        float(np.mean(below > risk_cutoff * total_patients)),
        float(np.mean(above > risk_cutoff * total_patients)),
    )
