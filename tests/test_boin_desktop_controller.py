"""Replay original managed controller with independent supplied decision services."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_boin_desktop import document

from mdanderson_stats.boin_combination import BOINCombDecision, BOINCombSelection
from mdanderson_stats.boin_desktop import BOINDesktopPatient, read_boin_desktop_project
from mdanderson_stats.boin_desktop_controller import recommend_boin_desktop

CASES = json.loads(
    (Path(__file__).parents[1] / "research/fixtures/boin-desktop/controller.json").read_text()
)["cases"]


def study_for(case):
    design = case["design"]
    return replace(
        read_boin_desktop_project(document()),
        mode="contour" if design["FindContours"] else "combination",
        cohort_size=design["CohortSize"],
        maximum_patients=design["MaximumSampleSize"],
        start_dose=(design["StartingDoseLevel_DrugA"], design["StartingDoseLevel_DrugB"]),
        patients=tuple(
            BOINDesktopPatient(
                i + 1,
                p["Cohort"],
                (p["DoseIndexA"] + 1, p["DoseIndexB"] + 1),
                p["OutcomeCode"],
                "2026-10-12",
                "2026-10-12",
                i + 1,
            )
            for i, p in enumerate(case["patients"])
        ),
    )


@pytest.mark.parametrize("case", CASES)
def test_executed_original_managed_controller(case):
    study = study_for(case)
    calls = []
    status = case["supplied_analysis_status"]

    def analyze(study):
        calls.append("contour" if study.mode == "contour" else "single")
        return BOINCombDecision(
            {0: "stay", 2: "stop_safety", 3: "complete", 5: "stop_precision"}[status],
            (2, 3) if status == 0 else None,
            np.zeros(study.shape, dtype=bool),
            np.zeros(study.shape),
        )

    def select(study):
        calls.append("final_selection")
        return BOINCombSelection(
            None,
            None,
            np.zeros(study.shape, dtype=bool),
            np.zeros(study.shape),
            np.zeros(study.shape),
        )

    actual = recommend_boin_desktop(
        study, seed=1, capacity_profile="desktop_legacy", analyze=analyze, select=select
    )
    expected = case["decision"]
    assert (
        actual.status
        == {0: "go", 1: "wait", 2: "stop_early", 3: "complete", 5: "converged"}[expected["Status"]]
    )
    assert actual.cohort == expected["Cohort"]
    if actual.status == "go":
        assert actual.dose == (expected["TreatmentIndexA"] + 1, expected["TreatmentIndexB"] + 1)
    else:
        assert actual.dose is None
    assert calls == [c for c in case["calls"] if c != "tally"]


def test_safe_capacity_does_not_repeat_native_first_cohort_precedence():
    case = next(
        c
        for c in CASES
        if c["design"]["CohortSize"] == 3
        and c["design"]["MaximumSampleSize"] == 2
        and not c["design"]["FindContours"]
        and len(c["patients"]) == 2
        and any(p["OutcomeCode"] == 2 for p in c["patients"])
    )
    study = study_for(case)
    assert recommend_boin_desktop(study, seed=1, capacity_profile="desktop_legacy").status == "go"
    assert recommend_boin_desktop(study, seed=1).status == "wait"


def test_actual_single_mtd_service_reports_interim_and_final():
    xml = document().replace(
        "</TrialDesign>",
        """<AcceptableToxicity_LowerBound>.15</AcceptableToxicity_LowerBound>
<AcceptableToxicity_UpperBound>.35</AcceptableToxicity_UpperBound><EliminateDoseThreshold>.95</EliminateDoseThreshold>
<ExtraSafe>false</ExtraSafe><Offset>.05</Offset><EarlyStop>12</EarlyStop></TrialDesign>""",
    )
    study = replace(
        read_boin_desktop_project(xml),
        patients=tuple(
            BOINDesktopPatient(i + 1, 1, (1, 1), 0, "2026-10-12", "2026-10-12", i + 1)
            for i in range(3)
        ),
    )
    result = recommend_boin_desktop(study, seed=1)
    assert result.status == "go" and result.dose in ((1, 2), (2, 1))
    completed = replace(study, maximum_patients=3)
    result = recommend_boin_desktop(completed, seed=1)
    assert result.status == "complete" and result.selection is not None
