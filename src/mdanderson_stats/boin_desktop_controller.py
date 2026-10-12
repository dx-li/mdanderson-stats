"""BOIN desktop patient/cohort controller with explicit capacity compatibility."""

import hashlib
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .boin_combination import BOINCombDecision, BOINCombSelection
from .boin_desktop import BOINDesktopStudy, _boolean, _integer, _number, desktop_bound_probabilities
from .boin_desktop_contour import desktop_contour_decision
from .boin_desktop_decision import BOINDesktopCombDesign
from .boin_desktop_selection import BOINDesktopSelection, select_boin_desktop_mtd


@dataclass(frozen=True)
class BOINDesktopRecommendation:
    status: str
    cohort: int
    dose: tuple[int, int] | None
    reason: str
    pending_patients: int
    selection: BOINCombSelection | None = None
    selection_details: BOINDesktopSelection | None = None
    capacity_profile: str = "safe"
    selection_profile: str = "safe"
    seed: int = 0
    study_sha256: str = ""


def desktop_combination_design(study: BOINDesktopStudy) -> BOINDesktopCombDesign:
    """Use the saved numeric bounds and safety inputs; retain inactive metadata."""
    if study.mode not in ("combination", "contour"):
        raise ValueError("a BOIN combination/contour study is required")
    design = ET.fromstring(study.xml).find("SimDat/TrialDesign")
    assert design is not None
    safe, toxic = desktop_bound_probabilities(study)
    return BOINDesktopCombDesign(
        target=study.target,
        safe_probability=safe,
        toxic_probability=toxic,
        elimination_probability=_number(design, "EliminateDoseThreshold"),
        extra_safe=_boolean(design, "ExtraSafe"),
        safety_offset=_number(design, "Offset"),
        early_stop_patients=_integer(design, "EarlyStop", 1, 1000),
        run_in=study.run_in,
    )


def recommend_boin_desktop(
    study: BOINDesktopStudy,
    *,
    seed: int,
    capacity_profile: str = "safe",
    selection_profile: str = "safe",
    analyze: Callable[[BOINDesktopStudy], BOINCombDecision] | None = None,
    select: Callable[[BOINDesktopStudy], BOINCombSelection] | None = None,
) -> BOINDesktopRecommendation:
    """Recommend a patient dose from the managed desktop cohort convention.

    Any pending outcome blocks analysis at a completed cohort or final capacity.
    Pending patients do not prohibit completing the current cohort. The original
    outer controller's first-cohort precedence can bypass capacity; only the
    explicit ``desktop_legacy`` profile preserves that defect. No patients are
    added by this function. Supplied services isolate controller validation.
    """
    if study.mode not in ("combination", "contour"):
        raise ValueError("managed combination conduct requires a combination/contour study")
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or not 0 <= seed < 2**64
    ):
        raise ValueError("seed must be an integer in [0,2**64)")
    if capacity_profile not in ("safe", "desktop_legacy"):
        raise ValueError("capacity_profile must be safe or desktop_legacy")
    if selection_profile not in ("safe", "desktop_legacy"):
        raise ValueError("selection_profile must be safe or desktop_legacy")
    selection_details = None
    patients = study.patients
    pending = sum(p.outcome == 2 for p in patients)
    cohort = patients[-1].cohort if patients else 1
    trailing = 0
    for patient in reversed(patients):
        if patient.cohort != cohort:
            break
        trailing += 1
    new_cohort = trailing >= study.cohort_size
    if new_cohort:
        cohort += 1

    def result(status, dose, reason, selection=None):
        return BOINDesktopRecommendation(
            status,
            cohort,
            dose,
            reason,
            pending,
            selection,
            selection_details,
            capacity_profile,
            selection_profile,
            int(seed),
            hashlib.sha256(study.xml.encode()).hexdigest(),
        )

    def final_selection():
        nonlocal selection_details
        if select is not None:
            return select(study)
        n, y, _ = study.counts()
        selection_details = select_boin_desktop_mtd(
            desktop_combination_design(study),
            n,
            y,
            contour=study.mode == "contour",
            profile=selection_profile,
        )
        return selection_details.selection

    capacity = (
        sum(study.subtrial_maximums)
        if study.mode == "contour" and study.subtrial_maximums
        else study.maximum_patients
    )
    at_capacity = len(patients) >= capacity
    if at_capacity and capacity_profile == "safe":
        if pending:
            return result("wait", None, "pending_at_capacity")
        return result("complete", None, "capacity", final_selection())
    if cohort == 1:
        return result("go", study.start_dose, "first_cohort")
    if at_capacity and study.mode == "combination":
        if pending:
            return result("wait", None, "pending_at_capacity")
        return result("complete", None, "capacity", final_selection())
    if not new_cohort:
        return result("go", patients[-1].dose, "complete_current_cohort")
    if pending:
        return result("wait", None, "pending_at_cohort_boundary")
    if analyze is not None:
        decision = analyze(study)
    else:
        n, y, _ = study.counts()
        design = desktop_combination_design(study)
        if study.mode == "contour":
            decision = desktop_contour_decision(
                design,
                n,
                y,
                patients[-1].dose,
                study.subtrial_maximums,
                compatibility_profile=capacity_profile,
            )
        else:
            decision = design.desktop_next_dose(n, y, patients[-1].dose, rng=seed)
    if decision.next_dose is not None:
        return result("go", decision.next_dose, decision.action)
    if decision.action == "error":
        return result("error", None, "native_invalid_dose")
    if decision.action == "stop_precision":
        return result("converged", None, decision.action, final_selection())
    if decision.action == "complete":
        return result("complete", None, decision.action, final_selection())
    return result("stop_early", None, decision.action)
