"""Original comparator references and saved comparison workflow contracts."""

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from test_boin_desktop_univariate_simulation import scenario_study

from mdanderson_stats.boin_desktop import read_boin_desktop_project
from mdanderson_stats.boin_desktop_comparison import (
    _three_plus_three,
    simulate_boin_desktop_workflow,
)
from mdanderson_stats.boin_desktop_report import boin_desktop_workflow_html

ROWS = list(
    csv.DictReader(
        (Path(__file__).parents[1] / "research/fixtures/boin-desktop/three-plus-three.csv")
        .read_text()
        .splitlines()
    )
)


class Probe(np.random.Generator):
    def binomial(self, n, p, size=None):
        result = np.sum(np.resize([0.1, 0.5, 0.9], int(n))[:, None] < np.atleast_1d(p), axis=0)
        return int(result[0]) if np.ndim(p) == 0 else result


@pytest.mark.parametrize("row", ROWS)
def test_original_three_plus_three_expansion(row):
    p = {"gradient": [0.05, 0.15, 0.25, 0.55, 0.75], "all": [0.95] * 5, "none": [0.01] * 5}[
        row["truth"]
    ]
    n, y, selected = _three_plus_three(
        np.asarray(p), 2, int(row["cap"]) if row["cap"] else None, Probe(np.random.PCG64(1))
    )
    np.testing.assert_allclose(n.mean(axis=0), np.fromstring(row["patients"], sep=";"))
    np.testing.assert_allclose(y.mean(axis=0), np.fromstring(row["toxicities"], sep=";"))
    np.testing.assert_allclose(
        np.bincount(selected, minlength=6)[1:] * 50, np.fromstring(row["selection"], sep=";")
    )


def comparison_study(tite=False, enabled=True, expansion=False):
    root = ET.fromstring(scenario_study(tite).xml)
    sim = root.find("SimDat")
    for field, value in {
        "Simulate_Rolling6Design" if tite else "Simulate_3Plus3Design": enabled,
        "Simulate_3Plus3Design_CohortExpansion": expansion,
    }.items():
        node = sim.find(field)
        if node is None:
            node = ET.SubElement(sim, field)
        node.text = str(value).lower()
    return read_boin_desktop_project(ET.tostring(root, encoding="unicode"))


@pytest.mark.parametrize("tite", [False, True])
def test_saved_comparisons_replay_and_report(tite):
    study = comparison_study(tite, expansion=not tite)
    result = simulate_boin_desktop_workflow(study, batches=2, trials=5, seed=2**64 - 1)
    repeat = simulate_boin_desktop_workflow(study, batches=2, trials=5, seed=2**64 - 1)
    assert len({r.seed for b in (result.primary, result.comparator) for r in b.results}) == 4
    for r, other in zip(result.comparator.results, repeat.comparator.results, strict=True):
        np.testing.assert_array_equal(r.patients, other.patients)
        assert not r.selected_eliminated.any()
        if tite:
            assert not r.unsafe_assignment_count.any()
            assert (r.actual_duration >= r.reported_duration).all()
    report = boin_desktop_workflow_html(result)
    assert read_boin_desktop_project(report) == study
    assert report.count("<!doctype html>") == 1
    assert result.comparison_name in report
    if not tite:
        assert "never shrinks" in report


def test_disabled_comparator_ignores_inactive_expansion():
    study = comparison_study(enabled=False)
    study = read_boin_desktop_project(
        study.xml.replace(
            "<Simulate_3Plus3Design_CohortExpansion>false",
            "<Simulate_3Plus3Design_CohortExpansion>invalid",
        )
    )
    assert simulate_boin_desktop_workflow(study, trials=1).comparator is None


def test_joint_preflight_precedes_primary(monkeypatch):
    import mdanderson_stats.boin_desktop_comparison as module

    monkeypatch.setattr(
        module,
        "simulate_boin_desktop_scenarios",
        lambda *a, **k: pytest.fail("started before joint validation"),
    )
    with pytest.raises(ValueError, match="joint comparison"):
        simulate_boin_desktop_workflow(comparison_study(), batches=5, trials=10000)
