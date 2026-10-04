from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.pop_design import PoPDesign
from mdanderson_stats.pop_simulation import simulate_pop

report_module = importlib.import_module("mdanderson_stats.pop_protocol_report")


def _request(**overrides):
    values = {
        "design": PoPDesign(target=0.25),
        "scenarios": (
            report_module.PoPReportScenario("low", [0.05, 0.20, 0.45]),
            report_module.PoPReportScenario("high", [0.10, 0.30, 0.60]),
        ),
        "total_patients": 12,
        "cohort_size": 3,
        "trials": 100,
        "start_dose": 1,
        "titration": True,
        "earlyterm": True,
        "risk_cutoff": 0.5,
        "seed": 20261004,
    }
    values.update(overrides)
    return report_module.run_pop_protocol(**values)


def test_report_uses_source_simulation_metrics_for_nonexact_target_truth():
    truth = [0.10, 0.20, 0.40]
    report = _request(
        scenarios=(report_module.PoPReportScenario("near target", truth),),
        trials=40,
    )
    summary = report.scenarios[0]
    assert summary.true_mtd == 2
    assert report.risk_cutoff == 0.5
    assert len(summary.selection_probability) == 4
    assert len(summary.selection_mcse) == 4

    direct = simulate_pop(
        report.design,
        truth,
        total_patients=report.total_patients,
        cohort_size=report.cohort_size,
        trials=report.trials,
        start_dose=report.start_dose,
        titration=report.titration,
        earlyterm=report.earlyterm,
        risk_cutoff=report.risk_cutoff,
        seed=summary.seed,
    )
    np.testing.assert_array_equal(summary.selection_probability, direct.selection_probability)
    np.testing.assert_array_equal(summary.selection_mcse, direct.selection_mcse)
    np.testing.assert_array_equal(summary.mean_patients, direct.mean_patients)
    np.testing.assert_array_equal(summary.mean_toxicities, direct.mean_toxicities)
    assert summary.mean_total_patients == pytest.approx(direct.mean_patients.sum())
    assert summary.mean_total_toxicities == pytest.approx(direct.mean_toxicities.sum())
    assert summary.risk_under == direct.risk_under
    assert summary.risk_over == direct.risk_over
    assert summary.risk_under_mcse == direct.risk_under_mcse
    assert summary.risk_over_mcse == direct.risk_over_mcse
    assert summary.early_stop_probability == direct.early_stop_probability
    assert summary.early_stop_mcse == direct.early_stop_mcse


def test_full_integer_boundary_table_and_first_nearest_tie_are_reported():
    report = _request(
        scenarios=(report_module.PoPReportScenario("tie", [0.10, 0.40]),),
        total_patients=8,
        cohort_size=2,
        trials=10,
    )
    assert report.scenarios[0].true_mtd == 1
    table = report.design.boundaries(8, cohort_size=1)
    assert tuple(row[0] for row in report.boundaries) == tuple(range(1, 9))
    assert tuple(row[1] for row in report.boundaries) == tuple(table.escalate_max)
    assert tuple(row[2] for row in report.boundaries) == tuple(table.deescalate_min)
    assert tuple(row[3] for row in report.boundaries) == tuple(table.exclude_under_max)
    assert tuple(row[4] for row in report.boundaries) == tuple(table.exclude_over_min)
    html_text = report.to_html()
    assert "first dose closest to target on an exact distance tie" in html_text
    assert "Risk cutoff" in html_text
    assert "-1 or n+1" in html_text


def test_html_escapes_scenario_labels_and_notes_unimplemented_native_outputs():
    report = _request(
        scenarios=(report_module.PoPReportScenario("<dose & scenario>", [0.1, 0.2]),),
        total_patients=6,
        cohort_size=2,
        trials=8,
        titration=False,
    )
    rendered = report.to_html()
    assert "&lt;dose &amp; scenario&gt;" in rendered
    assert "no claim about it" in rendered
    assert "No interval calculation is inferred" in rendered
    assert "Risk is calculated from the first dose closest to target" in rendered


def test_report_preflights_per_scenario_workload_before_simulation(monkeypatch):
    def unexpected_simulation(*args, **kwargs):
        raise AssertionError("simulation must not begin before all bounds pass")

    monkeypatch.setattr(report_module, "simulate_pop", unexpected_simulation)
    scenarios = tuple(
        report_module.PoPReportScenario(f"scenario {index}", [0.1, 0.2]) for index in range(20)
    )
    with pytest.raises(ValueError, match="each scenario"):
        report_module.run_pop_protocol(
            PoPDesign(),
            scenarios,
            total_patients=1000,
            trials=101,
            seed=1,
        )
    with pytest.raises(ValueError, match="each scenario"):
        report_module.run_pop_protocol(
            PoPDesign(),
            (report_module.PoPReportScenario("wide", [0.01] * 100),),
            total_patients=10,
            trials=1001,
            seed=1,
        )


def test_report_rejects_invalid_truth_and_cutoff_before_simulation():
    with pytest.raises(ValueError, match="nondecreasing"):
        report_module.run_pop_protocol(
            PoPDesign(),
            (report_module.PoPReportScenario("bad", [0.1, 0.05]),),
            total_patients=6,
            seed=1,
        )
    with pytest.raises(ValueError, match="risk_cutoff"):
        report_module.run_pop_protocol(
            PoPDesign(),
            (report_module.PoPReportScenario("valid", [0.1, 0.2]),),
            total_patients=6,
            risk_cutoff=1.1,
            seed=1,
        )


def test_html_write_is_utf8_and_atomic_destination_is_replaced(tmp_path: Path):
    report = _request(
        scenarios=(report_module.PoPReportScenario("café", [0.1, 0.2]),),
        total_patients=6,
        cohort_size=2,
        trials=8,
        titration=False,
    )
    destination = tmp_path / "report.html"
    destination.write_text("old", encoding="utf-8")
    assert report.write_html(destination) == destination
    content = destination.read_text(encoding="utf-8")
    assert "café" in content
    assert content == report.to_html()
