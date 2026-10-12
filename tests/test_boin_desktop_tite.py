"""Managed STFT reference and native saved time/prior contracts."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from test_boin_desktop_single import single_study

from mdanderson_stats.boin_desktop import read_boin_desktop_project
from mdanderson_stats.boin_desktop_tite import boin_desktop_stft, desktop_tite_settings

ROWS = json.loads(
    (Path(__file__).parents[1] / "research/fixtures/boin-desktop/stft.json").read_text()
)["cases"]


def saved_tite(window=3, unit="Months", prior=(0.1, 0.2, 0.7), uniform=True, followup=(30, 45, 60)):
    root = ET.fromstring(single_study(True).xml)
    sim = root.find("SimDat")
    design = sim.find("TrialDesign")
    for field, value in {
        "DLTAssessmentWindow": window,
        "DLTAssessmentWindow_Unit": unit,
        "UseUniform_ToxPriorProbs": str(uniform).lower(),
        "AccrualRate": 1,
        "MaxPendingOutcomePercent": 50,
    }.items():
        ET.SubElement(design, field).text = str(value)
    node = ET.SubElement(design, "ToxPriorProbs")
    for value in prior:
        ET.SubElement(node, "double").text = str(value)
    node = ET.SubElement(sim, "TITE_BOIN_PendingPatientData")
    for value in followup:
        ET.SubElement(node, "double").text = str(value)
    ET.SubElement(sim, "TimeToToxDistribution").text = "eUniform"
    ET.SubElement(sim, "ProbDLTInLateHalf").text = ".5"
    return read_boin_desktop_project(ET.tostring(root, encoding="unicode"))


@pytest.mark.parametrize("row", ROWS)
def test_original_stft_il(row):
    saved = saved_tite(
        row["window_days"], "Days", row["weights"], row["uniform"], row["followup_days"]
    )
    result = boin_desktop_stft(saved)
    assert abs(result.total - row["total"]) < 1e-14
    assert not result.weights.flags.writeable


def test_pending_calculator_times_are_days_even_when_window_is_months():
    study = saved_tite(followup=(30.436666666666667, 60.873333333333335))
    settings = desktop_tite_settings(study)
    result = boin_desktop_stft(study)
    assert settings.window_months == 3 and settings.window_days == 91.31
    np.testing.assert_allclose(result.weights, [1 / 3, 2 / 3])


def test_native_simulation_normalization_is_distinct_from_calculator_probabilities():
    study = saved_tite(prior=(1, 2, 7), uniform=False)
    assert desktop_tite_settings(study).trimester_weights == (0.1, 0.2, 0.7)
    with pytest.raises(ValueError):
        boin_desktop_stft(study)


def test_inactive_timing_values_do_not_change_uniform_model():
    study = read_boin_desktop_project(
        saved_tite().xml.replace("<ProbDLTInLateHalf>.5", "<ProbDLTInLateHalf>2")
    )
    assert desktop_tite_settings(study).late_probability is None
    study = read_boin_desktop_project(
        study.xml.replace(">eUniform</TimeToToxDistribution>", ">eWeibull</TimeToToxDistribution>")
    )
    with pytest.raises(ValueError, match="late-half"):
        desktop_tite_settings(study)


def test_calculator_does_not_require_inactive_simulation_choices():
    study = read_boin_desktop_project(
        saved_tite().xml.replace("<TimeToToxDistribution>eUniform</TimeToToxDistribution>", "")
    )
    assert boin_desktop_stft(study).total > 0
    with pytest.raises(ValueError):
        desktop_tite_settings(study)


def test_unrepresentable_accrual_gap_is_rejected_before_simulation():
    study = read_boin_desktop_project(
        saved_tite().xml.replace("<AccrualRate>1", "<AccrualRate>1e-320")
    )
    with pytest.raises(ValueError, match="representable"):
        desktop_tite_settings(study)


def test_saved_followup_edit_and_reset_preserve_active_study():
    from mdanderson_stats.boin_desktop_tite import set_boin_desktop_followup

    saved = saved_tite(window=90, unit="Days")
    edited = set_boin_desktop_followup(saved, [30, 48, 75])
    assert boin_desktop_stft(edited).total == pytest.approx(1.7)
    assert edited.scenarios == saved.scenarios
    assert boin_desktop_stft(set_boin_desktop_followup(edited, [])).total == 0
    assert boin_desktop_stft(set_boin_desktop_followup(edited, [0, 90])).total == 1
    for values in ([91], [-1], [float("nan")], [[30]], [0] * 1001):
        with pytest.raises(ValueError):
            set_boin_desktop_followup(saved, values)
