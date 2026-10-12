"""Whole original single-agent wrappers plus saved study replay/preflight."""

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from test_boin_desktop_single import single_study
from test_boin_desktop_tite import saved_tite

from mdanderson_stats.boin import BOINDesign
from mdanderson_stats.boin_desktop import read_boin_desktop_project
from mdanderson_stats.boin_desktop_batch import simulate_boin_desktop_scenarios
from mdanderson_stats.boin_desktop_simulation import simulate_boin_desktop_study
from mdanderson_stats.boin_desktop_univariate_simulation import _single_trials

ROWS = list(
    csv.DictReader(
        (Path(__file__).parents[1] / "research/fixtures/boin-desktop/single-simulation.csv")
        .read_text()
        .splitlines()
    )
)


class Probe(np.random.Generator):
    def random(self, size=None, dtype=np.float64, out=None):
        return 0.1 if size is None else np.resize([0.1, 0.5, 0.9], size)


@pytest.mark.parametrize("row", ROWS)
def test_original_complete_single_agent_wrapper(row):
    target = float(row["target"])
    p = {"gradient": [0.05, 0.15, target, 0.55, 0.75], "all": [0.95] * 5, "none": [0.01] * 5}[
        row["truth"]
    ]
    design = BOINDesign(
        target=target,
        early_stop_patients=12,
        extra_safe=row["extra_safe"] == "TRUE",
        stay_at_one_of_three=row["runin"] == "TRUE" and target == 0.25,
    )
    n, y, chosen, _ = _single_trials(
        design,
        np.asarray(p),
        2,
        6,
        3,
        int(row["start"]),
        row["titration"] == "TRUE",
        int(row["cap"]),
        Probe(np.random.PCG64(1)),
        "desktop_legacy",
    )
    np.testing.assert_allclose(n.mean(axis=0), np.fromstring(row["patients"], sep=";"))
    np.testing.assert_allclose(y.mean(axis=0), np.fromstring(row["toxicities"], sep=";"))
    np.testing.assert_allclose(
        np.bincount(chosen, minlength=6)[1:] * 50, np.fromstring(row["selection"], sep=";")
    )


def scenario_study(tite):
    study = saved_tite() if tite else single_study()
    root = ET.fromstring(study.xml)
    node = root.find("SimDat/Scenarios/Scenario/TrueProbabilities")
    for child, p in zip(node, [0.1, 0.3], strict=True):
        child.text = str(p)
    return read_boin_desktop_project(ET.tostring(root, encoding="unicode"))


@pytest.mark.parametrize("tite", [False, True])
@pytest.mark.parametrize("profile", ["safe", "desktop_legacy"])
def test_saved_univariate_batch_replay(tite, profile):
    study = scenario_study(tite)
    batch = simulate_boin_desktop_scenarios(study, batches=2, trials=3, seed=651, profile=profile)
    assert len(batch.results) == 2
    for result in batch.results:
        replay = simulate_boin_desktop_study(study, trials=3, seed=result.seed, profile=profile)
        np.testing.assert_array_equal(result.patients, replay.patients)
        assert result.patients.shape == (3, 2, 1)
        assert (result.patients.sum(axis=(1, 2)) <= 12).all()
        if tite:
            np.testing.assert_array_equal(result.reported_duration, replay.reported_duration)
            assert (result.actual_duration >= result.reported_duration).all()
            if profile == "safe":
                assert not result.future_stop.any()
        else:
            assert result.reported_duration is None


def test_all_tite_scenarios_are_validated_before_running(monkeypatch):
    import mdanderson_stats.boin_desktop_batch as module

    study = scenario_study(True)
    root = ET.fromstring(study.xml)
    root.find("SimDat/TimeToToxDistribution").text = "eWeibull"
    node = root.find("SimDat/Scenarios")
    extra = ET.SubElement(node, "Scenario")
    ET.SubElement(extra, "Name").text = "Invalid Weibull endpoint"
    ET.SubElement(extra, "RowsPerScenario").text = "2"
    values = ET.SubElement(extra, "TrueProbabilities")
    for p in (0.2, 1):
        ET.SubElement(values, "double").text = str(p)
    study = read_boin_desktop_project(ET.tostring(root, encoding="unicode"))
    monkeypatch.setattr(
        module,
        "simulate_boin_desktop_study",
        lambda *a, **k: pytest.fail("started before preflight"),
    )
    with pytest.raises(ValueError, match="below one"):
        simulate_boin_desktop_scenarios(study, trials=1)


def test_calendar_work_budget_is_request_wide():
    with pytest.raises(ValueError, match="bounded calendar"):
        simulate_boin_desktop_scenarios(
            scenario_study(True), batches=10, trials=1000, max_calendar_gaps=10000
        )


def test_timing_budget_and_legacy_qualifications_survive_reports():
    from mdanderson_stats.boin_desktop_report import boin_desktop_simulation_html

    study = scenario_study(True)
    batch = simulate_boin_desktop_scenarios(
        study, trials=2, seed=77, profile="desktop_legacy", max_calendar_gaps=1001
    )
    result = batch.results[0]
    assert result.calendar_gap_budget == 1001
    replay = simulate_boin_desktop_study(
        study,
        trials=2,
        seed=result.seed,
        profile=result.profile,
        max_calendar_gaps=result.calendar_gap_budget,
    )
    np.testing.assert_array_equal(result.reported_duration, replay.reported_duration)
    report = boin_desktop_simulation_html(batch)
    assert "Arrival-gap budget" in report and "Future-outcome stops" in report
    assert "future unobserved outcomes" in report and "1001" in report
    assert "phantom patients" not in report
    assert read_boin_desktop_project(report) == study


def test_disabled_titration_does_not_validate_an_inactive_endpoint():
    design = BOINDesign(target=0.25, early_stop_patients=12)
    n, _, _, _ = _single_trials(
        design, np.zeros(5), 1, 6, 3, 4, False, 1, Probe(np.random.PCG64(1))
    )
    assert n.sum() <= 18 and not n[0, :3].any()
