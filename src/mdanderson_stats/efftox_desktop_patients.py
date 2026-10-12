"""Validated current EffTox patient-file edits with semantic metadata retention."""

import xml.etree.ElementTree as ET

from .efftox_desktop import EffToxDesktopStudy, read_efftox_desktop_xml
from .efftox_desktop_controller import Analysis, recommend_efftox_desktop
from .efftox_model import _integer


def update_efftox_desktop_patient(
    study: EffToxDesktopStudy,
    patient_id: int,
    *,
    outcome: int | None = None,
    new_patient_id: int | None = None,
    administered_dose_index: int | None = None,
    begin_date: str | None = None,
) -> EffToxDesktopStudy:
    """Edit recorded observations without changing calculated dose or cohort.

    Outcome -1 clears both outcomes; None leaves the existing value unchanged.
    Dates are literal identifiers. Unknown patient/study XML fields survive.
    """
    identifier = _integer(patient_id, "patient_id", 0, 2**31 - 1)
    root = ET.fromstring(study.xml)
    nodes = [
        n
        for n in root.findall("Cohorts/Cohort/Patient")
        if int(n.findtext("ID", "-1")) == identifier
    ]
    if len(nodes) != 1:
        raise ValueError("patient_id must identify exactly one existing patient")
    updates: dict[str, str] = {}
    if new_patient_id is not None:
        updates["ID"] = str(_integer(new_patient_id, "new_patient_id", 0, 2**31 - 1))
    if outcome is not None:
        updates["Outcome"] = str(_integer(outcome, "outcome", -1, 3))
    if administered_dose_index is not None:
        updates["DoseAdminIndex"] = str(
            _integer(administered_dose_index, "administered_dose_index", 0, len(study.doses) - 1)
        )
    if begin_date is not None:
        if not isinstance(begin_date, str) or not begin_date.strip():
            raise ValueError("begin_date must be a nonempty literal string")
        updates["BeginDate"] = begin_date
    for name, value in updates.items():
        node = nodes[0].find(name)
        assert node is not None
        node.text = value
    return read_efftox_desktop_xml(ET.tostring(root, encoding="unicode"))


def add_efftox_desktop_patient(
    study: EffToxDesktopStudy,
    analyze: Analysis,
    *,
    patient_id: int,
    begin_date: str,
    administered_dose_index: int | None = None,
    outcome: int = -1,
    max_analyses: int = 1000,
) -> EffToxDesktopStudy:
    """Recompute GO before appending one patient; always enforce actual capacity.

    Preserves prescribed and administered dose separately. No legacy profile can
    bypass capacity or manual shutdown. The analysis service must be explicit.
    """
    identifier = _integer(patient_id, "patient_id", 0, 2**31 - 1)
    code = _integer(outcome, "outcome", -1, 3)
    if not isinstance(begin_date, str) or not begin_date.strip():
        raise ValueError("begin_date must be a nonempty literal string")
    if any(p.patient_id == identifier for p in study.patients):
        raise ValueError("patient_id already exists")
    if len(study.patients) >= study.max_patients:
        raise ValueError("patient capacity has been reached")
    root = ET.fromstring(study.xml)
    if root.findtext("TrialIsShutDown") != "0":
        raise ValueError("the saved trial is shut down")
    admin = (
        None
        if administered_dose_index is None
        else _integer(administered_dose_index, "administered_dose_index", 0, len(study.doses) - 1)
    )
    if any(len(g.findall("Patient")) > study.cohort_size for g in root.findall("Cohorts/Cohort")):
        raise ValueError("cannot enroll into a history containing an overfull cohort")
    recommendation = recommend_efftox_desktop(study, analyze, max_analyses=max_analyses)
    if recommendation.status != "go" or recommendation.dose_index is None:
        raise ValueError(f"cannot add a patient: controller returned {recommendation.status}")
    cohorts = root.find("Cohorts")
    if cohorts is None:
        cohorts = ET.SubElement(root, "Cohorts")
    groups = cohorts.findall("Cohort")
    if recommendation.new_cohort:
        cohort = ET.SubElement(cohorts, "Cohort")
    else:
        cohort = next(g for g in reversed(groups) if g.findall("Patient"))
    patient = ET.SubElement(cohort, "Patient")
    for name, value in {
        "ID": identifier,
        "BeginDate": begin_date,
        "DoseCalcedIndex": recommendation.dose_index,
        "DoseAdminIndex": recommendation.dose_index if admin is None else admin,
        "Outcome": code,
    }.items():
        ET.SubElement(patient, name).text = str(value)
    return read_efftox_desktop_xml(ET.tostring(root, encoding="unicode"))


def delete_last_efftox_desktop_patient(study: EffToxDesktopStudy) -> EffToxDesktopStudy:
    """Remove the final recorded patient and its cohort if that cohort becomes empty."""
    root = ET.fromstring(study.xml)
    cohorts = root.find("Cohorts")
    if cohorts is None or not study.patients:
        raise ValueError("there is no recorded patient to delete")
    groups = [g for g in cohorts.findall("Cohort") if g.findall("Patient")]
    cohort = groups[-1]
    cohort.remove(cohort.findall("Patient")[-1])
    if not cohort.findall("Patient"):
        cohorts.remove(cohort)
    return read_efftox_desktop_xml(ET.tostring(root, encoding="unicode"))


def clear_efftox_desktop_patients(study: EffToxDesktopStudy) -> EffToxDesktopStudy:
    """Return an empty patient history, retaining design/scenarios/other metadata."""
    root = ET.fromstring(study.xml)
    cohorts = root.find("Cohorts")
    if cohorts is not None:
        for group in cohorts.findall("Cohort"):
            cohorts.remove(group)
    return read_efftox_desktop_xml(ET.tostring(root, encoding="unicode"))
