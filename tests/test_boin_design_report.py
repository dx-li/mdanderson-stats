import importlib
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.boin import BOINDesign
from mdanderson_stats.boin_design_report import BOINReportScenario, boin_design_report
from mdanderson_stats.boin_simulation import simulate_boin
from mdanderson_stats.dose_allocation_risk import dose_allocation_risks


def test_report_captures_exact_boundaries_and_renders_summary_safely() -> None:
    escalation = float(np.nextafter(0.2, 0.0))
    deescalation = float(np.nextafter(0.4, 1.0))
    design = BOINDesign.from_boundaries(0.3, escalation, deescalation)
    scenarios = (
        BOINReportScenario("<scenario & one>", [0.1, 0.3, 0.8]),
        BOINReportScenario("second", [0.2, 0.4, 0.6]),
    )
    report = boin_design_report(
        design,
        scenarios,
        cohorts=2,
        cohort_size=3,
        trials=12,
        seed=19,
    )

    assert report.design.escalation_boundary == escalation
    assert report.design.deescalation_boundary == deescalation
    summary = report.scenarios[0]
    assert summary.allocation_risk.has_exact_target
    assert len(summary.selection_probability) == 4
    assert sum(value for _, value in summary.stop_frequency) == pytest.approx(1.0)
    html = report.to_html()
    assert "&lt;scenario &amp; one&gt;" in html
    assert "More than 60% of planned enrollment" in html
    assert "19" in html
    assert "trial-level simulation arrays" in html

    generator = np.random.default_rng(19)
    for scenario, summary in zip(scenarios, report.scenarios, strict=True):
        direct = simulate_boin(
            report.design,
            scenario.true_toxicity,
            cohorts=report.cohorts,
            cohort_size=report.cohort_size,
            trials=report.trials,
            start_dose=report.start_dose,
            titration=report.titration,
            titration_cap=report.titration_cap,
            moderate_toxicity=report.moderate_toxicity if report.titration else None,
            rng=generator,
        )
        assert summary.selection_probability == tuple(direct.selection_probability)
        assert summary.mean_patients == tuple(direct.mean_patients)
        assert summary.mean_toxicities == tuple(direct.mean_toxicities)
        assert summary.allocation_risk == dose_allocation_risks(
            direct.patients,
            scenario.true_toxicity,
            target=report.design.target,
            planned_patients=report.cohorts * report.cohort_size,
        )


def test_titration_summary_and_atomic_html_write(tmp_path: Path) -> None:
    report = boin_design_report(
        BOINDesign(target=0.3),
        (BOINReportScenario("accelerated", [0.1, 0.3, 0.5]),),
        cohorts=2,
        cohort_size=3,
        trials=20,
        titration=True,
        titration_cap=2,
        moderate_toxicity=[0.05, 0.1, 0.1],
        seed=7,
        language="zh",
    )
    scenario = report.scenarios[0]
    assert len(scenario.mean_titration_moderate_toxicities) == 3
    assert dict(scenario.titration_end_frequency)
    assert report.language == "zh"
    assert report.titration_cap == 2
    destination = tmp_path / "report.html"
    assert report.write_html(destination) == destination
    saved = destination.read_text(encoding="utf-8")
    assert 'lang="zh"' in saved
    assert "accelerated" in saved


def test_moderate_toxicity_must_be_exclusive_under_every_scenario() -> None:
    with pytest.raises(ValueError, match="every scenario"):
        boin_design_report(
            BOINDesign(target=0.3),
            (
                BOINReportScenario("valid", [0.1, 0.3]),
                BOINReportScenario("invalid", [0.95, 0.3]),
            ),
            cohorts=2,
            cohort_size=3,
            trials=10,
            titration=True,
            moderate_toxicity=[0.1, 0.1],
            seed=1,
        )


def test_aggregate_preflight_rejects_before_simulation(monkeypatch) -> None:
    report_module = importlib.import_module("mdanderson_stats.boin_design_report")

    def unexpected_simulation(*args, **kwargs):
        raise AssertionError("simulation must not start before aggregate preflight")

    monkeypatch.setattr(report_module, "simulate_boin", unexpected_simulation)
    scenarios = tuple(BOINReportScenario(f"scenario-{index}", [0.1, 0.3]) for index in range(20))
    with pytest.raises(ValueError, match="2000000"):
        report_module.boin_design_report(
            BOINDesign(target=0.3),
            scenarios,
            cohorts=10,
            cohort_size=10,
            trials=1010,
            seed=1,
        )
