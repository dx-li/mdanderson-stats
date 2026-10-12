"""Original patient serialization and outcome/censor instruction references."""

import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from test_crm_desktop import document

from mdanderson_stats.crm_desktop import _envelope, read_crm_desktop_html
from mdanderson_stats.crm_desktop_history import (
    CRMDesktopHistory,
    CRMDesktopPatient,
    crm_desktop_snapshot,
    read_crm_desktop_history,
)

ROOT = Path(__file__).resolve().parents[1] / "research/fixtures/crm-native"
BASE = date(2026, 10, 1)


def study():
    return read_crm_desktop_html(
        document()
        .replace("<EvalTime>1</EvalTime>", "<EvalTime>6</EvalTime>")
        .replace("<MaxPatients>2</MaxPatients>", "<MaxPatients>30</MaxPatients>")
    )


@pytest.mark.parametrize(
    "case", [json.loads(row) for row in (ROOT / "patient-outcomes.jsonl").read_text().splitlines()]
)
def test_original_outcome_and_censor_instructions(case):
    arrival, window, updated, toxicity_delay, at = case["input"]
    outcome = 1 if toxicity_delay <= window else 0 if updated >= window else -1
    patient = CRMDesktopPatient(
        1,
        10,
        1,
        0,
        BASE + timedelta(days=arrival),
        BASE + timedelta(days=arrival + updated),
        outcome,
    )
    history = CRMDesktopHistory(study(), (patient,), "mdy")
    snap = crm_desktop_snapshot(history, at=BASE + timedelta(days=at))
    assert snap.outcomes.tolist() == [-1 if case["outcome"] == 2 else case["outcome"]]
    assert snap.times.tolist() == [case["followup"]]


@pytest.mark.parametrize("culture,order", [("en-US", "mdy"), ("en-GB", "dmy")])
def test_original_managed_patient_round_trip(culture, order):
    source = (ROOT / f"{culture}.xml").read_bytes()
    assert source == (ROOT / f"{culture}-roundtrip.xml").read_bytes()
    root = ET.fromstring(study().xml)
    patient = ET.fromstring(source)
    ET.SubElement(patient, "UnknownPatientSetting").text = "preserved"
    root.find("Patients").append(patient)
    original = _envelope("", ET.tostring(root, encoding="unicode"))
    history = read_crm_desktop_history(original, date_order=order)
    assert history.patients[0].treatment_date == date(2026, 10, 2)
    assert history.patients[0].assessment_date == date(2026, 10, 5)
    assert history.patients[0].outcome == -1
    restored = read_crm_desktop_history(history.to_html(), date_order=order)
    assert restored.patients == history.patients
    assert (
        ET.fromstring(restored.study.xml).findtext("Patients/Patient/UnknownPatientSetting")
        == "preserved"
    )
    updated = replace(history.patients[0], outcome=1, assessment_date=date(2026, 10, 6))
    edited = history.with_patients((updated,))
    reloaded = read_crm_desktop_history(edited.to_html(), date_order=order)
    assert reloaded.patients == (updated,)
    assert ET.fromstring(reloaded.study.xml).findtext("PreservedSetting") == "untouched"


def test_pending_assessment_never_auto_completes_and_dates_can_be_retrospective():
    patient = CRMDesktopPatient(1, 10, 1, 0, BASE, BASE + timedelta(days=2), -1)
    history = CRMDesktopHistory(study(), (patient,), "ymd")
    later = crm_desktop_snapshot(history, at=BASE + timedelta(days=100))
    assert later.outcomes.tolist() == [-1]
    assert later.times.tolist() == [2]
    assert later.pending_counts.tolist() == [1, 0]
    before = crm_desktop_snapshot(history, at=BASE - timedelta(days=1))
    assert before.doses.size == 0
    np.testing.assert_array_equal(before.observed_counts, [0, 0])
    with pytest.raises(ValueError):
        later.times[0] = 99


@pytest.mark.parametrize("outcome,elapsed", [(-1, 6), (0, 5), (1, 7)])
def test_invalid_outcome_assessment_pair(outcome, elapsed):
    patient = CRMDesktopPatient(1, 10, 1, 0, BASE, BASE + timedelta(days=elapsed), outcome)
    with pytest.raises(ValueError, match="toxicity window"):
        CRMDesktopHistory(study(), (patient,), "mdy")


def test_duplicate_identifiers_and_ambiguous_date_rejected():
    p = CRMDesktopPatient(1, 10, 1, 0, BASE, BASE, -1)
    with pytest.raises(ValueError, match="unique"):
        CRMDesktopHistory(study(), (p, p), "mdy")
    history = CRMDesktopHistory(study(), (p,), "mdy")
    with pytest.raises(ValueError, match="four-digit"):
        read_crm_desktop_history(
            history.to_html().replace("10/1/2026", "10/1/26"), date_order="mdy"
        )
    with pytest.raises(ValueError, match="date_order"):
        read_crm_desktop_history(history.to_html(), date_order="auto")
