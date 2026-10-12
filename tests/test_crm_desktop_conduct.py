"""Conduct workflow, original cohort probes and independent posterior decisions."""

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from test_crm_desktop import document
from test_crm_desktop_history import BASE, study

from mdanderson_stats import fit_bmacrm
from mdanderson_stats.bmacrm_decision import bmacrm_decision
from mdanderson_stats.crm_desktop import read_crm_desktop_html
from mdanderson_stats.crm_desktop_conduct import _incomplete_cohort, recommend_crm_desktop
from mdanderson_stats.crm_desktop_history import (
    CRMDesktopHistory,
    CRMDesktopPatient,
    read_crm_desktop_history,
)

CASES = [
    json.loads(row)
    for row in (
        Path(__file__).resolve().parents[1] / "research/fixtures/crm-native/cohort-control.jsonl"
    )
    .read_text()
    .splitlines()
]


@pytest.mark.parametrize("case", [c for c in CASES if "cohorts" in c])
def test_original_native_cohort_helper(case):
    assert _incomplete_cohort(tuple(case["cohorts"]), case["size"]) == case["incomplete"]


def test_complete_patient_workflow_and_final_posterior():
    history = CRMDesktopHistory(study(), (), "mdy")
    start = recommend_crm_desktop(history, at=BASE, seed=100)
    assert (start.action, start.dose, start.can_add_patient) == ("start", 0, True)
    patient = CRMDesktopPatient(1, 11, 1, 0, BASE, BASE, -1)
    history = history.with_patients((patient,))
    historical = recommend_crm_desktop(history, at=BASE - timedelta(days=1), seed=100)
    assert historical.action == "start" and not historical.can_add_patient
    observed = replace(patient, outcome=0, assessment_date=BASE + timedelta(days=6))
    history = history.with_patients((observed,))
    next_ = recommend_crm_desktop(history, at=BASE + timedelta(days=6), seed=100)
    expected = fit_bmacrm(history.study.skeletons, [0, 0], [1, 0], target=0.25)
    np.testing.assert_allclose(next_.analysis.posterior.dose_mean, expected.dose_mean, atol=1e-14)
    assert next_.can_add_patient
    saved = read_crm_desktop_history(next_.to_html(), date_order="mdy")
    assert saved.patients == history.patients
    finalstudy = read_crm_desktop_html(
        history.study.to_html().replace(
            "<MaxPatients>30</MaxPatients>", "<MaxPatients>1</MaxPatients>"
        )
    )
    final = recommend_crm_desktop(
        CRMDesktopHistory(finalstudy, (observed,), "mdy"), at=observed.assessment_date, seed=100
    )
    assert final.action == "select_mtd" and not final.can_add_patient
    assert final.analysis.decision.high_uncertainty


@pytest.mark.parametrize("case", [c for c in CASES if "da_within_cohort_safety" in c])
def test_original_da_within_cohort_safety_disposition(case):
    cutoff = 0 if case["da_within_cohort_safety"] else 1
    text = (
        document(True)
        .replace("<CohortSize>1</CohortSize>", "<CohortSize>3</CohortSize>")
        .replace("<MaxPatients>2</MaxPatients>", "<MaxPatients>6</MaxPatients>")
        .replace("<EvalTime>1</EvalTime>", "<EvalTime>6</EvalTime>")
        .replace(
            "<UpperLimitOnProbToxAtLowestDose>.9</UpperLimitOnProbToxAtLowestDose>",
            f"<UpperLimitOnProbToxAtLowestDose>{cutoff}</UpperLimitOnProbToxAtLowestDose>",
        )
    )
    patient = CRMDesktopPatient(1, 11, 1, 0, BASE, BASE + timedelta(days=1), -1)
    history = CRMDesktopHistory(read_crm_desktop_html(text), (patient,), "mdy")
    decision = recommend_crm_desktop(
        history, at=patient.assessment_date, seed=12, draws=8, warmup=0, chains=2
    )
    assert decision.within_cohort
    assert (0 if decision.action == "treat" else 1) == case["action"]
    assert decision.dose == (0 if case["action"] == 0 else None)
    assert "Parameter split R-hat" in decision.to_html()


def test_capacity_with_pending_da_uses_complete_outcome_lookahead():
    text = document(True).replace("<MaxPatients>2</MaxPatients>", "<MaxPatients>1</MaxPatients>")
    patient = CRMDesktopPatient(1, 11, 1, 0, BASE, BASE, -1)
    history = CRMDesktopHistory(read_crm_desktop_html(text), (patient,), "ymd")
    decision = recommend_crm_desktop(history, at=BASE, seed=12, max_completions=1)
    assert decision.analysis.routing == "lookahead"
    assert decision.action == "wait"
    assert decision.analysis.decision.reason == "work_limit"
    assert not decision.can_add_patient


def test_raw_gate_is_an_explicit_separate_controller_choice():
    posterior = fit_bmacrm([0.01, 0.05, 0.25], [1, 0, 0], [2, 0, 0], target=0.25, prior_sd=0.1)
    ordinary = bmacrm_decision(posterior, current_dose=0, safety_cutoff=1)
    ungated = bmacrm_decision(posterior, current_dose=0, safety_cutoff=1, apply_raw_rate=False)
    assert ordinary.dose == 0 and ordinary.raw_rate_limited
    assert ungated.dose == 1 and not ungated.raw_rate_limited
    with pytest.raises(ValueError, match="apply_raw_rate"):
        bmacrm_decision(posterior, current_dose=0, apply_raw_rate="no")


@pytest.mark.parametrize(
    "case",
    [
        json.loads(row)
        for row in (
            Path(__file__).resolve().parents[1]
            / "research/fixtures/crm-native/managed-cohort-control.jsonl"
        )
        .read_text()
        .splitlines()
    ],
)
def test_original_managed_replay_gate(case):
    from mdanderson_stats.crm_desktop_conduct import _cohort_plan

    action = {0: "treat", 1: "wait", 2: "stop", 3: "select_mtd"}[case["status"]]
    cohort, prefix = _cohort_plan(tuple(case["cohorts"]), case["size"], action)
    assert cohort == case["next_cohort"]
    assert (prefix is not None) == case["replay"]
    assert prefix == case["prefix"]


def test_managed_wrapper_recalculates_cohort_entry_and_adds_pending_patient():
    design = read_crm_desktop_html(
        study()
        .to_html()
        .replace("<CohortSize>1</CohortSize>", "<CohortSize>3</CohortSize>")
        .replace(
            "<UpperLimitOnProbToxAtLowestDose>.9</UpperLimitOnProbToxAtLowestDose>",
            "<UpperLimitOnProbToxAtLowestDose>1</UpperLimitOnProbToxAtLowestDose>",
        )
    )
    preceding = tuple(
        CRMDesktopPatient(
            i + 1, i + 11, 1, 0, BASE + timedelta(days=i), BASE + timedelta(days=i + 6), 0
        )
        for i in range(3)
    )
    history = CRMDesktopHistory(design, preceding, "mdy")
    entry = recommend_crm_desktop(history, at=BASE + timedelta(days=9), seed=5)
    added = entry.add_patient(patient_id=4, fingerprint=14)
    assert added.patients[-1].outcome == -1
    assert added.patients[-1].cohort == 2
    assert added.patients[-1].dose == entry.dose
    updated = replace(added.patients[-1], outcome=1, assessment_date=BASE + timedelta(days=10))
    revised = added.with_patients(preceding + (updated,))
    decision = recommend_crm_desktop(revised, at=BASE + timedelta(days=10), seed=5)
    assert decision.within_cohort and decision.cohort_entry_analysis is not None
    assert decision.dose == entry.dose
    np.testing.assert_array_equal(decision.cohort_entry_analysis.snapshot.toxicities, [0, 0])
    np.testing.assert_array_equal(
        decision.analysis.snapshot.toxicities, np.bincount([entry.dose], minlength=2)
    )
    assert "Cohort-entry analysis" in decision.to_html()
    assert "Trial design and simulation settings" in decision.to_html()
    assert (
        decision.evaluations
        == decision.analysis.evaluations + decision.cohort_entry_analysis.evaluations
    )


def test_current_safety_stop_precedes_cohort_rewind():
    design = read_crm_desktop_html(
        study()
        .to_html()
        .replace("<CohortSize>1</CohortSize>", "<CohortSize>3</CohortSize>")
        .replace(
            "<UpperLimitOnProbToxAtLowestDose>.9</UpperLimitOnProbToxAtLowestDose>",
            "<UpperLimitOnProbToxAtLowestDose>0</UpperLimitOnProbToxAtLowestDose>",
        )
    )
    patient = CRMDesktopPatient(1, 11, 1, 0, BASE, BASE + timedelta(days=1), 1)
    history = CRMDesktopHistory(design, (patient,), "mdy")
    stopped = recommend_crm_desktop(history, at=patient.assessment_date, seed=12)
    assert stopped.action == "stop" and not stopped.can_add_patient
    assert stopped.cohort_entry_analysis is None
    with pytest.raises(ValueError, match="does not allow"):
        stopped.add_patient(patient_id=2, fingerprint=12)


def test_final_pending_consensus_preserves_three_subject_uncertainty():
    saved = read_crm_desktop_html(
        document()
        .replace("<double>.3</double>", "<double>.1</double>")
        .replace("<MaxPatients>2</MaxPatients>", "<MaxPatients>1</MaxPatients>")
        .replace(
            "<UpperLimitOnProbToxAtLowestDose>.9</UpperLimitOnProbToxAtLowestDose>",
            "<UpperLimitOnProbToxAtLowestDose>1</UpperLimitOnProbToxAtLowestDose>",
        )
    )
    pending = CRMDesktopPatient(1, 11, 1, 0, BASE, BASE, -1)
    result = recommend_crm_desktop(CRMDesktopHistory(saved, (pending,), "mdy"), at=BASE, seed=5)
    assert result.action == "select_mtd" and result.dose == 0
    assert result.analysis.decision.reason == "invariant"
    assert all(d.high_uncertainty for d in result.analysis.decision.decisions)
    assert result.analysis.decision.high_uncertainty and result.high_uncertainty
    assert "High uncertainty" in result.explanation
    assert "Final-dose high uncertainty: True" in result.to_html()
