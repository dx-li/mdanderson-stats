"""Original executed control-flow fixtures plus independent sufficient statistics."""

import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_efftox_desktop import document

from mdanderson_stats.efftox_desktop import read_efftox_desktop_xml
from mdanderson_stats.efftox_desktop_controller import (
    _cohort_prefix,
    _completions,
    _pending_consistent,
    _status,
    recommend_efftox_desktop,
)
from mdanderson_stats.efftox_desktop_decision import EffToxDesktopDecision

FIXTURES = Path(__file__).resolve().parents[1] / "research/fixtures/efftox-native"


@pytest.mark.parametrize("case", json.loads((FIXTURES / "current-cohort-prefix.json").read_text()))
def test_original_cohort_prefix(case):
    assert _cohort_prefix(case["cohorts"], case["size"]) == (
        case["prefix"],
        case["new_cohort"],
        case["next_cohort"],
    )


@pytest.mark.parametrize("case", json.loads((FIXTURES / "current-controller.json").read_text()))
def test_original_outer_controller(case):
    status = _status(
        case["patients"],
        case["maximum"],
        not case["pending"] or case["consistent"],
        case["selection_stopped"],
    )
    assert status == ("go", "wait", "stop_early", "complete")[case["status"]]


@pytest.mark.parametrize("case", json.loads((FIXTURES / "current-pending-loop.json").read_text()))
def test_original_pending_baseline_loop(case):
    analyzed = [case["baseline"]]

    def stream():
        for completion in case["completions"]:
            analyzed.append(completion)
            yield tuple(completion)

    assert _pending_consistent(tuple(case["baseline"]), stream()) == case["consistent"]
    assert analyzed == case["analyzed"]


@pytest.mark.parametrize("pending", [(0, 0), (1, 0), (2, 1), (1, 2), (3, 2)])
def test_aggregated_completions_equal_independent_patient_enumeration(pending):
    expected = set()
    doses = [dose for dose, count in enumerate(pending) for _ in range(count)]
    for outcomes in itertools.product(range(4), repeat=sum(pending)):
        matrix = np.zeros((2, 2, 2), dtype=np.int64)
        for dose, outcome in zip(doses, outcomes, strict=True):
            matrix[dose, outcome // 2, outcome % 2] += 1
        expected.add(tuple(matrix.flat))
    observed = [
        tuple(c.flat) for c in _completions(np.zeros((2, 2, 2), dtype=np.int64), np.array(pending))
    ]
    assert len(observed) == len(set(observed))
    assert set(observed) == expected


def decision(dose):
    return EffToxDesktopDecision(dose, np.ones(2, dtype=bool), np.zeros(2), np.ones(2), np.ones(2))


def test_pending_agreement_must_include_observed_baseline_and_returns_no_dose():
    study = read_efftox_desktop_xml(document())
    seen = []

    def analyze(counts):
        seen.append(counts.copy())
        return decision(0 if counts.sum() == 2 else 1)

    result = recommend_efftox_desktop(study, analyze)
    assert result.status == "wait" and result.dose_index is None
    assert result.analyzed_patients == 3 and result.pending_patients == 1
    assert result.analyses == 2 and result.completion_count == 4
    assert seen[0].sum() == 2 and seen[1].sum() == 3
    assert seen[0][0].sum() == 0  # administered, not calculated dose


def test_work_limit_rejects_before_any_inference():
    study = read_efftox_desktop_xml(document())

    def never(_):
        pytest.fail("preflight must run first")

    result = recommend_efftox_desktop(study, never, max_analyses=4)
    assert result.status == "work_limit" and result.analyses == 0
    assert result.completion_count == 4 and result.dose_index is None


def test_nonmultiple_capacity_bug_is_only_in_explicit_legacy_profile():
    study = read_efftox_desktop_xml(document().replace("<MaxQtyPatients>6", "<MaxQtyPatients>5"))
    safe = recommend_efftox_desktop(study, lambda _: decision(1))
    legacy = recommend_efftox_desktop(
        study, lambda _: decision(1), capacity_profile="desktop_legacy"
    )
    assert safe.status == "complete" and safe.capacity_guard
    assert safe.analyzed_patients == 5 and safe.analyses == 5
    assert legacy.status == "go" and not legacy.capacity_guard
    assert legacy.analyzed_patients == 3


def test_no_pending_early_stop_and_complete():
    study = read_efftox_desktop_xml(document())
    patients = tuple(replace(p, outcome=1) for p in study.patients[:3])
    study = replace(study, patients=patients, max_patients=3)
    assert recommend_efftox_desktop(study, lambda _: decision(None)).status == "stop_early"
    result = recommend_efftox_desktop(study, lambda _: decision(0))
    assert result.status == "complete" and result.analyses == 1
    assert result.completion_count == 0
