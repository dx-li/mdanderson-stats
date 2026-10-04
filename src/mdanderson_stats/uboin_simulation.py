"""Bounded simulation helpers for complete-outcome U-BOIN designs."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite, scalar
from .uboin_conduct import UBOINDecision, UBOINDesign
from .uboin_titration import UBOINTitrationPlan, uboin_stage1_titration_plan


def _readonly(value: ArrayLike, dtype: np.dtype | type = np.float64) -> NDArray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _integer(value: int, name: str, low: int, high: int) -> int:
    candidate = scalar(value, name)
    if candidate != np.floor(candidate) or not low <= candidate <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(candidate)


def uboin_gumbel_probabilities(
    toxicity: ArrayLike,
    efficacy: ArrayLike,
    *,
    association: float = 0.2,
) -> FloatArray:
    """Return joint Bernoulli cells from the U-BOIN Gumbel association model.

    Rows are efficacy ``0,1`` and columns are toxicity ``0,1``.  The
    factored form remains finite for association values such as ``+/-1000``.
    """
    tox_shape = np.shape(toxicity)
    eff_shape = np.shape(efficacy)
    if len(tox_shape) != 1 or len(eff_shape) != 1 or tox_shape != eff_shape:
        raise ValueError("toxicity and efficacy must be same-shaped one-dimensional vectors")
    if not 1 <= tox_shape[0] <= 100:
        raise ValueError("toxicity and efficacy must contain 1..100 doses")
    tox = finite(toxicity, "toxicity")
    eff = finite(efficacy, "efficacy")
    if np.any((tox < 0) | (tox > 1)) or np.any((eff < 0) | (eff > 1)):
        raise ValueError("toxicity and efficacy probabilities must lie in [0,1]")
    association = scalar(association, "association")
    rho = np.tanh(association / 2)
    p00 = (1 - eff) * (1 - tox) * (1 + eff * tox * rho)
    p01 = (1 - eff) * tox * (1 - eff * (1 - tox) * rho)
    p10 = eff * (1 - tox) * (1 - (1 - eff) * tox * rho)
    p11 = eff * tox * (1 + (1 - eff) * (1 - tox) * rho)
    result = np.stack((p00, p01, p10, p11), axis=1).reshape(-1, 2, 2)
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ArithmeticError("Gumbel probabilities are not finite nonnegative values")
    return _readonly(result)


@dataclass(frozen=True)
class UBOINSimulation:
    selections: NDArray[np.int64]
    joint_counts: NDArray[np.int64]
    stop_reason: NDArray[np.str_]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    mean_patients: FloatArray
    mean_toxicities: FloatArray
    mean_responses: FloatArray
    early_stop_probability: float
    early_stop_mcse: float
    titration_patients: NDArray[np.int64] | None = None
    titration_grade2_toxicities: NDArray[np.int64] | None = None
    titration_end_dose: NDArray[np.int64] | None = None
    titration_end_reason: NDArray[np.str_] | None = None


def simulate_uboin(
    design: UBOINDesign,
    joint_probabilities: ArrayLike,
    *,
    cohort_size: int = 3,
    trials: int = 1000,
    seed: int | None = None,
    accelerated_titration: bool = False,
    titration_cap: int | None = None,
    grade2_toxicity_level: int | None = None,
) -> UBOINSimulation:
    """Simulate complete categorical outcomes under a U-BOIN conduct design.

    A cohort is sampled at the current dose, then the controller is called.
    Thus a cohort can overshoot ``s1`` or ``s2`` by at most ``cohort_size-1``.
    Optional source-style accelerated titration samples one patient at a time
    through its Stage-I prelude, then tops up the terminal singleton or starts
    a full cohort above a clean lower cap. No patient-level history is retained.
    It requires a one-based ``grade2_toxicity_level`` that maps to a category
    below the design's DLT split; binary toxicity outcomes cannot distinguish
    grade 2 from DLT and therefore cannot use this option.
    """
    if not isinstance(design, UBOINDesign):
        raise TypeError("design must be a UBOINDesign")
    try:
        shape = np.shape(joint_probabilities)
    except (TypeError, ValueError) as exc:
        raise ValueError("joint_probabilities must have shape (D,E,T)") from exc
    if len(shape) != 3:
        raise ValueError("joint_probabilities must have shape (D,E,T)")
    d, e, t = shape
    prior = np.asarray(design.prior)
    if not 1 <= d <= 100 or (e, t) != prior.shape[-2:]:
        raise ValueError("joint_probabilities dimensions must match design")
    probabilities = finite(joint_probabilities, "joint_probabilities")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("joint_probabilities entries must lie in [0,1]")
    row_sums = probabilities.sum(axis=(1, 2))
    if not np.all(np.isclose(row_sums, 1.0, rtol=0, atol=1e-12)):
        raise ValueError("each dose joint probability table must sum to one")
    probabilities = probabilities / row_sums[:, None, None]
    cohort = _integer(cohort_size, "cohort_size", 1, 100)
    repetitions = _integer(trials, "trials", 1, 100_000)
    if not isinstance(accelerated_titration, (bool, np.bool_)):
        raise ValueError("accelerated_titration must be boolean")
    if repetitions * design.max_patients > 100_000:
        raise ValueError("trials * max_patients must be at most 100,000")
    if repetitions * d * e * t > 100_000:
        raise ValueError("trials * D * E * T must be at most 100,000")
    if repetitions * design.max_patients * d > 1_000_000:
        raise ValueError("trials * max_patients * D must be at most 1,000,000")
    # Validate all controller dimensions and scalar design settings before the
    # simulation loop, including a valid zero-data starting decision.
    empty = np.zeros((d, e, t), dtype=np.int64)
    design.decision(empty, current_dose=design.starting_dose, stage=1)

    if not accelerated_titration and (
        titration_cap is not None or grade2_toxicity_level is not None
    ):
        raise ValueError(
            "titration_cap and grade2_toxicity_level require accelerated_titration=True"
        )
    use_titration = bool(accelerated_titration) and cohort > 1 and design.starting_dose < d
    initial_plan: UBOINTitrationPlan | None = None
    cap_value = d
    if accelerated_titration:
        cap_value = _integer(
            d if titration_cap is None else titration_cap,
            "titration_cap",
            design.starting_dose,
            d,
        )
        if use_titration and grade2_toxicity_level is not None:
            grade2_level = _integer(grade2_toxicity_level, "grade2_toxicity_level", 2, t)
            if grade2_level - 1 >= design.dlt_level:
                raise ValueError(
                    "grade2_toxicity_level must identify a category below the DLT split"
                )
        elif use_titration:
            raise ValueError(
                "accelerated_titration requires an explicit grade2_toxicity_level"
            )
        initial_plan = uboin_stage1_titration_plan(
            [],
            max_dose=d,
            toxicity_category_count=t,
            starting_dose=design.starting_dose,
            cohort_size=cohort,
            max_patients=design.max_patients,
            dlt_level=design.dlt_level,
            grade2_toxicity_level=grade2_toxicity_level if use_titration else None,
            titration_cap=cap_value,
        )

    rng = np.random.default_rng(seed)
    all_counts = np.zeros((repetitions, d, e, t), dtype=np.int64)
    selections: NDArray[np.int64] = np.zeros(repetitions, dtype=np.int64)
    reasons: NDArray[np.str_] = np.empty(repetitions, dtype="U32")
    patients = np.zeros((repetitions, d), dtype=np.float64)
    toxicities = np.zeros((repetitions, d), dtype=np.float64)
    responses = np.zeros((repetitions, d), dtype=np.float64)
    titration_patients = np.zeros(repetitions, dtype=np.int64)
    titration_grade2 = np.zeros(repetitions, dtype=np.int64)
    titration_end_dose = np.zeros(repetitions, dtype=np.int64)
    default_titration_reason = "not_requested"
    if accelerated_titration and cohort == 1:
        default_titration_reason = "cohort_size_one"
    elif accelerated_titration and design.starting_dose == d:
        default_titration_reason = "start_at_highest_dose"
    titration_end_reason = np.full(repetitions, default_titration_reason, dtype="U32")

    for trial in range(repetitions):
        counts = np.zeros((d, e, t), dtype=np.int64)
        current = design.starting_dose
        stage = 1
        eliminated = None
        terminal: UBOINDecision | None = None

        if use_titration:
            if initial_plan is None or grade2_toxicity_level is None:
                raise RuntimeError("missing preflighted U-BOIN titration contract")
            categories: list[int] = []
            plan = initial_plan
            while not plan.complete:
                if plan.next_titration_dose is None:
                    raise RuntimeError("incomplete U-BOIN titration plan has no next dose")
                dose = plan.next_titration_dose
                draw = rng.multinomial(1, probabilities[dose - 1].reshape(-1))
                counts[dose - 1] += draw.reshape(e, t)
                categories.append(
                    int(np.flatnonzero(draw.reshape(e, t).sum(axis=0))[0]) + 1
                )
                plan = uboin_stage1_titration_plan(
                    categories,
                    max_dose=d,
                    toxicity_category_count=t,
                    starting_dose=design.starting_dose,
                    cohort_size=cohort,
                    max_patients=design.max_patients,
                    dlt_level=design.dlt_level,
                    grade2_toxicity_level=grade2_toxicity_level,
                    titration_cap=cap_value,
                )
            titration_patients[trial] = len(plan.titration_doses)
            titration_grade2[trial] = plan.grade2_toxicities
            titration_end_reason[trial] = plan.end_reason
            if plan.end_dose is not None:
                titration_end_dose[trial] = plan.end_dose
            current = plan.resume_dose
            if (
                plan.end_reason == "lower_cap_without_trigger"
                and int(counts.sum()) >= design.max_patients
                and plan.end_dose is not None
            ):
                current = plan.end_dose

            if plan.top_up_patients:
                top_up = min(plan.top_up_patients, design.max_patients - int(counts.sum()))
                if top_up > 0:
                    draw = rng.multinomial(top_up, probabilities[current - 1].reshape(-1))
                    counts[current - 1] += draw.reshape(e, t)

            # A toxicity/highest-cap exit completes the terminal dose's first
            # cohort with its singleton plus the prescribed m-1 patients.
            # A clean lower cap instead starts an untouched full cohort above it.
            if plan.titration_doses and plan.end_reason != "lower_cap_without_trigger":
                terminal = design.decision(
                    counts, current_dose=current, stage=stage, eliminated=eliminated
                )
                if not np.any(terminal.allocation_probabilities):
                    reasons[trial] = terminal.action
                    selections[trial] = terminal.selected_dose or 0
                else:
                    allocation = terminal.allocation_probabilities
                    if np.count_nonzero(allocation) == 1:
                        current = int(np.flatnonzero(allocation == 1)[0] + 1)
                    else:
                        current = int(rng.choice(d, p=allocation)) + 1
                    stage = terminal.stage
                    eliminated = terminal.eliminated

        while True:
            if terminal is not None and not np.any(terminal.allocation_probabilities):
                break
            total = int(counts.sum())
            size = min(cohort, design.max_patients - total)
            if size <= 0:
                result = design.decision(
                    counts, current_dose=current, stage=stage, eliminated=eliminated
                )
                reasons[trial] = result.action
                selections[trial] = result.selected_dose or 0
                break
            draw = rng.multinomial(size, probabilities[current - 1].reshape(-1))
            counts[current - 1] += draw.reshape(e, t)
            result = design.decision(
                counts, current_dose=current, stage=stage, eliminated=eliminated
            )
            if not np.any(result.allocation_probabilities):
                reasons[trial] = result.action
                selections[trial] = result.selected_dose or 0
                break
            allocation = result.allocation_probabilities
            if np.count_nonzero(allocation) == 1:
                current = int(np.flatnonzero(allocation == 1)[0] + 1)
            else:
                current = int(rng.choice(d, p=allocation)) + 1
            stage = result.stage
            eliminated = result.eliminated
        all_counts[trial] = counts
        patients[trial] = counts.sum(axis=(1, 2))
        toxicities[trial] = counts[:, :, design.dlt_level :].sum(axis=(1, 2))
        responses[trial] = counts[:, design.response_level :, :].sum(axis=(1, 2))

    selection_probability = np.bincount(selections, minlength=d + 1).astype(float) / repetitions
    selection_mcse = np.sqrt(selection_probability * (1 - selection_probability) / repetitions)
    early = patients.sum(axis=1) < design.max_patients
    early_probability = float(np.mean(early))
    early_mcse = float(np.sqrt(early_probability * (1 - early_probability) / repetitions))
    return UBOINSimulation(
        _readonly(selections, dtype=np.int64),
        _readonly(all_counts, dtype=np.int64),
        _readonly(reasons, dtype=np.str_),
        _readonly(selection_probability),
        _readonly(selection_mcse),
        _readonly(patients.mean(axis=0)),
        _readonly(toxicities.mean(axis=0)),
        _readonly(responses.mean(axis=0)),
        early_probability,
        early_mcse,
        _readonly(titration_patients, dtype=np.int64),
        _readonly(titration_grade2, dtype=np.int64),
        _readonly(titration_end_dose, dtype=np.int64),
        _readonly(titration_end_reason, dtype=np.str_),
    )
