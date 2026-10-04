"""Source-defined single-patient Stage-I titration for U-BOIN."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import count, scalar


@dataclass(frozen=True)
class UBOINTitrationPlan:
    """Dose path and cohort handoff for an observed titration prefix.

    Dose numbers and toxicity categories are one-based. ``dlt_level`` is the
    existing U-BOIN split index: categories at or above that zero-based index
    count as DLT. A grade-2 category must lie below this split.
    """

    titration_doses: tuple[int, ...]
    grade2_toxicities: int
    end_reason: str
    end_dose: int | None
    resume_dose: int
    top_up_patients: int
    complete: bool
    next_titration_dose: int | None


def _integer(value: int, name: str, low: int, high: int) -> int:
    candidate = scalar(value, name)
    if candidate != np.floor(candidate) or not low <= candidate <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(candidate)


def _category_vector(value: ArrayLike, maximum_length: int) -> NDArray[np.float64]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError("toxicity_categories must be a one-dimensional vector")
        if value.size > maximum_length:
            raise ValueError("toxicity_categories exceed the titration path limit")
    else:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("toxicity_categories must be a one-dimensional sequence")
        if len(value) > maximum_length:
            raise ValueError("toxicity_categories exceed the titration path limit")
        if any(not np.isscalar(item) or np.iscomplexobj(item) for item in value):
            raise ValueError("toxicity_categories must be a one-dimensional real vector")
    categories = count(value, "toxicity_categories")
    if categories.ndim != 1:
        raise ValueError("toxicity_categories must be a one-dimensional vector")
    return categories


def uboin_stage1_titration_plan(
    toxicity_categories: ArrayLike,
    *,
    max_dose: int,
    toxicity_category_count: int,
    starting_dose: int,
    cohort_size: int,
    max_patients: int,
    dlt_level: int,
    grade2_toxicity_level: int | None,
    titration_cap: int | None = None,
) -> UBOINTitrationPlan:
    """Replay the U-BOIN singleton dose path from observed toxicity categories.

    ``toxicity_categories`` is the observed one-based category for each
    singleton, in the order patients were treated. The path visits each dose
    from ``starting_dose`` upward at most once. Supply a prefix to receive an
    incomplete plan with the next dose that still needs one singleton result.
    Once the source exit rule is reached, extra category values are rejected.

    A DLT or second grade-2 toxicity (or a clean arrival at the highest dose)
    tops the final singleton up with ``cohort_size - 1`` patients at that dose,
    truncated by ``max_patients``. Reaching a lower cap without either
    toxicity trigger starts a full cohort at the next dose and adds no patients
    at the cap dose. Cohort size one and a highest-dose starting point disable
    titration as specified by the guide.
    """
    maximum = _integer(max_dose, "max_dose", 1, 100)
    category_count = _integer(toxicity_category_count, "toxicity_category_count", 2, 3)
    start = _integer(starting_dose, "starting_dose", 1, maximum)
    size = _integer(cohort_size, "cohort_size", 1, 100)
    patient_limit = _integer(max_patients, "max_patients", 1, 1000)
    dlt = _integer(dlt_level, "dlt_level", 1, 2)
    cap = maximum if titration_cap is None else _integer(
        titration_cap, "titration_cap", start, maximum
    )
    if dlt >= category_count:
        raise ValueError("dlt_level must be a valid split index for toxicity categories")

    if size == 1 or start == maximum:
        categories = _category_vector(toxicity_categories, 0)
        if categories.size != 0:
            raise ValueError("a disabled titration plan must not include patient outcomes")
        reason = "cohort_size_one" if size == 1 else "start_at_highest_dose"
        return UBOINTitrationPlan((), 0, reason, None, start, 0, True, None)

    if grade2_toxicity_level is None:
        raise ValueError("accelerated titration requires an explicit grade-2 toxicity category")
    grade2_level = _integer(
        grade2_toxicity_level, "grade2_toxicity_level", 2, category_count
    )
    if grade2_level - 1 >= dlt:
        raise ValueError("grade-2 toxicity category must be below the DLT category split")

    maximum_path = min(cap - start + 1, patient_limit)
    categories = _category_vector(toxicity_categories, maximum_path)
    if np.any((categories < 1) | (categories > category_count)):
        raise ValueError(
            "toxicity_categories must be one-based categories within toxicity_category_count"
        )

    grade2_count = 0
    path: list[int] = []
    for offset, raw_category in enumerate(categories):
        dose = start + offset
        category = int(raw_category)
        path.append(dose)
        grade2_count += int(category == grade2_level)
        is_dlt = category - 1 >= dlt
        if is_dlt or grade2_count >= 2:
            reason = "first_dlt" if is_dlt else "second_grade2"
            top_up = min(size - 1, patient_limit - len(path))
            plan = UBOINTitrationPlan(
                tuple(path), grade2_count, reason, dose, dose, top_up, True, None
            )
            if offset + 1 != categories.size:
                raise ValueError("toxicity_categories contain outcomes after the titration stop")
            return plan
        if dose == cap:
            if cap == maximum:
                reason, resume, top_up = "highest_dose_cap", dose, min(
                    size - 1, patient_limit - len(path)
                )
            else:
                reason, resume, top_up = "lower_cap_without_trigger", dose + 1, 0
            plan = UBOINTitrationPlan(
                tuple(path), grade2_count, reason, dose, resume, top_up, True, None
            )
            if offset + 1 != categories.size:
                raise ValueError("toxicity_categories contain outcomes after the titration cap")
            return plan
        if len(path) == patient_limit:
            plan = UBOINTitrationPlan(
                tuple(path), grade2_count, "max_patients", dose, dose, 0, True, None
            )
            if offset + 1 != categories.size:
                raise ValueError("toxicity_categories exceed max_patients")
            return plan

    next_dose = start + len(path)
    if next_dose > cap or len(path) >= patient_limit:
        raise ValueError("titration plan could not determine an exit")
    return UBOINTitrationPlan(
        tuple(path), grade2_count, "continue", path[-1] if path else None,
        next_dose, 0, False, next_dose
    )
