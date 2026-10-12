"""Explicit-date CRM Suite patient files, editing, and as-of observations.

Native short-date strings require a caller-selected date order. Pending records
stop contributing follow-up at their last assessment; elapsed wall time alone
never turns an unassessed patient into a completed nontoxic outcome.
"""

from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from datetime import date, datetime
from html import escape
from typing import Literal

import numpy as np

from ._cdflib import _freeze
from .crm_calendar import CRMCalendarSnapshot, _freeze_int
from .crm_desktop import CRMDesktopStudy, _envelope, _integer, _text, read_crm_desktop_html

DateOrder = Literal["mdy", "dmy", "ymd"]
_CODES = {"tslFALSE": 0, "tslTRUE": 1, "tslUNKNOWN": -1}


def _date(text: str, order: DateOrder) -> date:
    if order not in ("mdy", "dmy", "ymd"):
        raise ValueError("date_order must explicitly be mdy, dmy, or ymd")
    if not re.fullmatch(r"\d{1,4}([/-])\d{1,2}\1\d{1,4}", text):
        raise ValueError("dates require numeric components and a four-digit year")
    parts = [int(v) for v in re.split(r"[/-]", text)]
    values = dict(zip(order, parts, strict=True))
    year_text = re.split(r"[/-]", text)[order.index("y")]
    if len(year_text) != 4:
        raise ValueError("dates require a four-digit year; century guessing is unsupported")
    return date(values["y"], values["m"], values["d"])


def _date_text(value: date, order: DateOrder) -> str:
    values = {"m": str(value.month), "d": str(value.day), "y": f"{value.year:04d}"}
    return "/".join(values[key] for key in order)


def _whole(value: int, name: str, low: int, high: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")


@dataclass(frozen=True)
class CRMDesktopPatient:
    patient_id: int
    fingerprint: int
    cohort: int
    dose: int
    treatment_date: date
    assessment_date: date
    outcome: Literal[-1, 0, 1]
    dose_description: str = ""

    def __post_init__(self) -> None:
        _whole(self.patient_id, "patient_id", -(2**31), 2**31 - 1)
        _whole(self.fingerprint, "fingerprint", -(2**31), 2**31 - 1)
        _whole(self.cohort, "cohort", 0, 200)
        _whole(self.dose, "dose", 0, 19)
        _whole(self.outcome, "outcome", -1, 1)
        for value in (self.treatment_date, self.assessment_date):
            if not isinstance(value, date) or isinstance(value, datetime):
                raise ValueError("patient dates must be date objects without time-of-day")
        if self.assessment_date < self.treatment_date:
            raise ValueError("assessment must not precede treatment")
        if not isinstance(self.dose_description, str) or len(self.dose_description) > 10000:
            raise ValueError("dose_description must be bounded text")


def _settings(study: CRMDesktopStudy) -> tuple[int, int, int]:
    model = ET.fromstring(study.xml).find("CRMSimulatorData/Model")
    assert model is not None
    return (
        _integer(model, "EvalTime", 1, 2_147_483_647),
        _integer(model, "CohortSize", 1, 4),
        _integer(model, "MaxPatients", 1, 200),
    )


@dataclass(frozen=True)
class CRMDesktopHistory:
    study: CRMDesktopStudy
    patients: tuple[CRMDesktopPatient, ...]
    date_order: DateOrder

    def __post_init__(self) -> None:
        _date("2026/1/1" if self.date_order == "ymd" else "1/1/2026", self.date_order)
        if not isinstance(self.study, CRMDesktopStudy):
            raise TypeError("study must be a CRMDesktopStudy")
        object.__setattr__(self, "study", read_crm_desktop_html(self.study.to_html()))
        parents = ET.fromstring(self.study.xml).findall("Patients")
        if len(parents) > 1:
            raise ValueError("duplicate Patients sections")
        for node in [] if not parents else parents[0]:
            tags = [child.tag for child in node]
            if node.tag != "Patient" or len(tags) != len(set(tags)):
                raise ValueError("unsupported or duplicate patient fields")
        object.__setattr__(self, "patients", tuple(self.patients))
        window, _, maximum = _settings(self.study)
        if len(self.patients) > maximum:
            raise ValueError("patient history exceeds MaxPatients")
        ids, fingerprints = set(), set()
        previous = date.min
        for p in self.patients:
            if not isinstance(p, CRMDesktopPatient):
                raise TypeError("patients must contain CRMDesktopPatient records")
            if p.patient_id in ids or p.fingerprint in fingerprints:
                raise ValueError("patient IDs and fingerprints must be unique")
            ids.add(p.patient_id)
            fingerprints.add(p.fingerprint)
            if p.dose >= len(self.study.dose_labels) or p.treatment_date < previous:
                raise ValueError("require valid dose indices and chronological treatment dates")
            previous = p.treatment_date
            elapsed = (p.assessment_date - p.treatment_date).days
            if (
                (p.outcome == -1 and elapsed >= window)
                or (p.outcome == 1 and elapsed > window)
                or (p.outcome == 0 and elapsed < window)
            ):
                raise ValueError("outcome and assessment date contradict the toxicity window")

    def with_patients(self, patients: tuple[CRMDesktopPatient, ...]) -> CRMDesktopHistory:
        """Return a validated edited history with the same saved design."""
        return replace(self, patients=patients)

    def to_html(self) -> str:
        """Save patient edits in the native XML envelope, retaining unknown fields."""
        root = ET.fromstring(self.study.xml)
        parents = root.findall("Patients")
        if len(parents) > 1:
            raise ValueError("duplicate Patients sections")
        parent = parents[0] if parents else ET.SubElement(root, "Patients")
        old = {node.findtext("PID"): node for node in parent.findall("Patient")}
        parent[:] = []
        for patient in self.patients:
            previous = old.get(str(patient.patient_id))
            node = copy.deepcopy(previous) if previous is not None else ET.Element("Patient")
            fields = {
                "PID": str(patient.patient_id),
                "Fingerprint": str(patient.fingerprint),
                "Cohort": str(patient.cohort),
                "DoseIndex": str(patient.dose),
                "OutcomeCode": next(k for k, v in _CODES.items() if v == patient.outcome),
                "DoseDescription": patient.dose_description,
                "DateEntered": _date_text(patient.treatment_date, self.date_order),
                "DateUpdated": _date_text(patient.assessment_date, self.date_order),
            }
            for key, value in fields.items():
                child = node.find(key)
                if child is None:
                    child = ET.SubElement(node, key)
                child.text = value
            parent.append(node)
        rows = [
            "<h1>CRM Suite patient history</h1>",
            f"<p>Date order: {self.date_order}; zero-based dose indices.</p>",
            "<table><tr><th>Patient</th><th>Dose</th><th>Treatment</th>"
            "<th>Assessment</th><th>Outcome</th></tr>",
        ]
        for p in self.patients:
            values = (
                p.patient_id,
                p.dose,
                p.treatment_date,
                p.assessment_date,
                {-1: "pending", 0: "no DLT", 1: "DLT"}[p.outcome],
            )
            rows.append("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in values) + "</tr>")
        rows.append("</table>")
        return _envelope("\n".join(rows), ET.tostring(root, encoding="unicode"))


def read_crm_desktop_history(document: str, *, date_order: DateOrder) -> CRMDesktopHistory:
    """Read native patient records with explicit date order, never locale guessing."""
    study = read_crm_desktop_html(document)
    root = ET.fromstring(study.xml)
    parent = root.find("Patients")
    patients = []
    for node in [] if parent is None else parent:
        tags = [child.tag for child in node]
        if node.tag != "Patient" or len(tags) != len(set(tags)):
            raise ValueError("unsupported or duplicate patient fields")
        code = _text(node, "OutcomeCode")
        if code not in _CODES:
            raise ValueError("unsupported native outcome code")
        patients.append(
            CRMDesktopPatient(
                _integer(node, "PID", -(2**31), 2**31 - 1),
                _integer(node, "Fingerprint", -(2**31), 2**31 - 1),
                _integer(node, "Cohort", 0, 200),
                _integer(node, "DoseIndex", 0, len(study.dose_labels) - 1),
                _date(_text(node, "DateEntered"), date_order),
                _date(_text(node, "DateUpdated"), date_order),
                _CODES[code],  # type: ignore[arg-type]
                node.findtext("DoseDescription", ""),
            )
        )
    return CRMDesktopHistory(study, tuple(patients), date_order)


def crm_desktop_snapshot(history: CRMDesktopHistory, *, at: date) -> CRMCalendarSnapshot:
    """Original day-resolution censor ordering: event, window, decision, update.

    Future-treated patients are excluded. A pending record's last assessment is
    an actual censoring bound, even when the decision date is much later.
    """
    if not isinstance(history, CRMDesktopHistory):
        raise TypeError("history must be a CRMDesktopHistory")
    if not isinstance(at, date) or isinstance(at, datetime):
        raise ValueError("at must be a date without time-of-day")
    window, _, _ = _settings(history.study)
    rows, doses, outcomes, times = [], [], [], []
    origin = min(history.patients[0].treatment_date, at) if history.patients else at
    for i, patient in enumerate(history.patients):
        if patient.treatment_date > at:
            continue
        updated = (patient.assessment_date - patient.treatment_date).days
        age = (at - patient.treatment_date).days
        event = updated if patient.outcome == 1 else window + 1
        followup, kind = min((event, 1), (window, 2), (age, 3), (updated, 4))
        rows.append(i)
        doses.append(patient.dose)
        outcomes.append(1 if kind == 1 else 0 if kind == 2 else -1)
        times.append(followup)
    dose_values = np.asarray(doses, dtype=np.int64)
    outcome_values = np.asarray(outcomes, dtype=np.int64)
    counts = [
        np.bincount(dose_values[mask], minlength=len(history.study.dose_labels))
        for mask in (
            np.ones(len(doses), dtype=bool),
            outcome_values != -1,
            outcome_values == 1,
            outcome_values == -1,
        )
    ]
    return CRMCalendarSnapshot(
        _freeze_int(np.asarray(rows, dtype=np.int64)),
        _freeze_int(dose_values),
        _freeze_int(outcome_values),
        _freeze(np.asarray(times, dtype=float)),
        _freeze_int(counts[0]),
        _freeze_int(counts[1]),
        _freeze_int(counts[2]),
        _freeze_int(counts[3]),
        float(window),
        float((at - origin).days),
    )
