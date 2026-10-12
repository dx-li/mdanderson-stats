"""Reversible patient-file edits for recovered BOIN combination conduct."""

import xml.etree.ElementTree as ET
from datetime import date

from .boin_desktop import BOINDesktopStudy, read_boin_desktop_project
from .boin_desktop_controller import recommend_boin_desktop

_CODES = ("tslNO", "tslYES", "tslUNKNOWN")


def _integer(value: int, name: str, low: int = 0, high: int = 2**31 - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return value


def _date(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("new dates must be ISO calendar-date strings")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ValueError("new dates must be ISO calendar-date strings") from exc


def _save(root: ET.Element) -> BOINDesktopStudy:
    return read_boin_desktop_project(ET.tostring(root, encoding="unicode"))


def add_boin_desktop_patient(
    study: BOINDesktopStudy,
    *,
    patient_id: int,
    date_entered: str,
    seed: int,
    outcome: int = 2,
) -> BOINDesktopStudy:
    """Recompute the recommendation and append one patient only on safe GO.

    New dates use ISO syntax so original locale-dependent short dates need not
    be guessed. Existing date strings and unrecognized XML metadata survive.
    """
    identifier = _integer(patient_id, "patient_id")
    code = _integer(outcome, "outcome", high=2)
    entered = _date(date_entered)
    if any(p.patient_id == identifier for p in study.patients):
        raise ValueError("patient_id already exists")
    maximum = sum(study.subtrial_maximums) if study.mode == "contour" else study.maximum_patients
    if len(study.patients) >= maximum:
        raise ValueError("patient capacity has been reached")
    recommendation = recommend_boin_desktop(study, seed=seed)
    if recommendation.status != "go" or recommendation.dose is None:
        raise ValueError(f"patient addition requires GO, received {recommendation.status}")
    fingerprint = max((p.fingerprint for p in study.patients), default=0) + 1
    _integer(fingerprint, "new fingerprint", low=-(2**31))
    root = ET.fromstring(study.xml)
    container = root.find("SimDat/Patients")
    assert container is not None
    node = ET.SubElement(container, "Patient")
    a, b = recommendation.dose
    fields = {
        "PID": identifier,
        "Fingerprint": fingerprint,
        "Cohort": recommendation.cohort,
        "DoseIndexA": a - 1,
        "DoseLevelB": b,
        "DoseIndexB": b - 1,
        "OutcomeCode": _CODES[code],
        "DateEntered": entered,
        "DateUpdated": entered,
    }
    for name, value in fields.items():
        ET.SubElement(node, name).text = str(value)
    return _save(root)


def update_boin_desktop_patient(
    study: BOINDesktopStudy,
    patient_id: int,
    *,
    date_updated: str,
    outcome: int | None = None,
    new_patient_id: int | None = None,
    date_entered: str | None = None,
) -> BOINDesktopStudy:
    """Update outcomes/identifiers/dates while retaining treatment, cohort and fingerprint."""
    identifier = _integer(patient_id, "patient_id")
    updated = _date(date_updated)
    root = ET.fromstring(study.xml)
    nodes = [
        n
        for n in root.findall("SimDat/Patients/Patient")
        if float(n.findtext("PID", "nan")) == identifier
    ]
    if len(nodes) != 1:
        raise ValueError("patient_id must identify exactly one existing patient")
    fields = {"DateUpdated": updated}
    if outcome is not None:
        fields["OutcomeCode"] = _CODES[_integer(outcome, "outcome", high=2)]
    if new_patient_id is not None:
        fields["PID"] = str(_integer(new_patient_id, "new_patient_id"))
    if date_entered is not None:
        fields["DateEntered"] = _date(date_entered)
    for name, value in fields.items():
        child = nodes[0].find(name)
        assert child is not None
        child.text = value
    return _save(root)


def delete_last_boin_desktop_patient(study: BOINDesktopStudy) -> BOINDesktopStudy:
    root = ET.fromstring(study.xml)
    container = root.find("SimDat/Patients")
    assert container is not None
    nodes = container.findall("Patient")
    if not nodes:
        raise ValueError("no patient is available to delete")
    container.remove(nodes[-1])
    return _save(root)


def clear_boin_desktop_patients(study: BOINDesktopStudy) -> BOINDesktopStudy:
    root = ET.fromstring(study.xml)
    container = root.find("SimDat/Patients")
    assert container is not None
    for child in container.findall("Patient"):
        container.remove(child)
    return _save(root)


def review_boin_desktop_patient(study: BOINDesktopStudy, patient_id: int, *, seed: int):
    """Return the preceding saved rows and their freshly computed safe decision.

    Native row review uses current outcomes of preceding patients, excluding the
    selected row and all subsequent rows. This is a recalculation, not a recovery
    of the original decision or of outcomes known on the original entry date.
    Random ties use the explicitly supplied seed. The original file is unchanged.
    """
    identifier = _integer(patient_id, "patient_id")
    matches = [i for i, patient in enumerate(study.patients) if patient.patient_id == identifier]
    if len(matches) != 1:
        raise ValueError("patient_id must identify exactly one existing patient")
    root = ET.fromstring(study.xml)
    container = root.find("SimDat/Patients")
    assert container is not None
    for node in container.findall("Patient")[matches[0] :]:
        container.remove(node)
    prefix = _save(root)
    return prefix, recommend_boin_desktop(prefix, seed=seed)
