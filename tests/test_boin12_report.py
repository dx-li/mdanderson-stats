from __future__ import annotations

import importlib

import numpy as np
import pytest

from mdanderson_stats.boin12 import BOIN12Design
from mdanderson_stats.boin12_report import BOIN12ReportScenario, boin12_report
from mdanderson_stats.boin12_simulation import simulate_boin12
from mdanderson_stats.boin12_two_stage import simulate_boin12_two_stage


def _truths() -> tuple[BOIN12ReportScenario, ...]:
    return (
        BOIN12ReportScenario(
            "lower < profile",
            [[0.72, 0.18, 0.06, 0.04], [0.55, 0.20, 0.15, 0.10], [0.35, 0.20, 0.25, 0.20]],
        ),
        BOIN12ReportScenario(
            "higher profile",
            [[0.50, 0.25, 0.15, 0.10], [0.38, 0.22, 0.24, 0.16], [0.25, 0.20, 0.33, 0.22]],
        ),
    )


def test_report_summaries_match_serial_single_and_two_stage_runs() -> None:
    design = BOIN12Design(0.35, 0.25, utilities=(100, 35, 65, 0))
    scenarios = _truths()
    settings = dict(cohorts=4, cohort_size=3, trials=18, start_dose=1, seed=2718)
    report = boin12_report(design, scenarios, **settings)
    generator = np.random.default_rng(settings["seed"])
    for scenario, summary in zip(scenarios, report.scenarios, strict=True):
        direct = simulate_boin12(
            design,
            scenario.joint_probability,
            cohorts=settings["cohorts"],
            cohort_size=settings["cohort_size"],
            trials=settings["trials"],
            start_dose=settings["start_dose"],
            rng=generator,
        )
        assert summary.obd_probability == tuple(direct.obd_probability)
        assert summary.mtd_probability == tuple(direct.mtd_probability)
        assert summary.mean_patients == tuple(direct.patients.mean(axis=0))
        assert summary.mean_toxicities == tuple(direct.toxicities.mean(axis=0))
        assert summary.mean_efficacies == tuple(direct.efficacies.mean(axis=0))
        assert np.isclose(sum(p for _, p in summary.stop_reason_frequency), 1)
        assert summary.transition_cohort_frequency is None

    threshold = 6
    two_stage = boin12_report(design, scenarios, stage1_threshold=threshold, **settings)
    generator = np.random.default_rng(settings["seed"])
    for scenario, summary in zip(scenarios, two_stage.scenarios, strict=True):
        direct = simulate_boin12_two_stage(
            design,
            scenario.joint_probability,
            stage1_threshold=threshold,
            cohorts=settings["cohorts"],
            cohort_size=settings["cohort_size"],
            trials=settings["trials"],
            start_dose=settings["start_dose"],
            rng=generator,
        )
        assert summary.obd_probability == tuple(direct.obd_probability)
        assert summary.transition_cohort_frequency is not None
        assert np.isclose(sum(p for _, p in summary.transition_cohort_frequency), 1)
        assert summary.mean_stage1_cohorts == float(direct.stage1_cohorts.mean())
        assert summary.mean_stage2_cohorts == float(direct.stage2_cohorts.mean())
        assert summary.joint_probability == scenario._grid


def test_report_preflight_rejects_duplicate_scenarios_before_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("mdanderson_stats.boin12_report")

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("simulation must not start before complete report preflight")

    monkeypatch.setattr(module, "simulate_boin12", unexpected)
    monkeypatch.setattr(module.np.random, "default_rng", unexpected)
    scenario = BOIN12ReportScenario("same", [[0.7, 0.2, 0.05, 0.05], [0.6, 0.2, 0.1, 0.1]])
    with pytest.raises(ValueError, match="unique"):
        boin12_report(BOIN12Design(0.35, 0.25), (scenario, scenario), seed=10)
    with pytest.raises(ValueError, match="work"):
        boin12_report(
            BOIN12Design(0.35, 0.25),
            (scenario,),
            cohorts=200,
            cohort_size=1,
            trials=6_000,
            seed=10,
        )


def test_report_writes_escaped_summary_and_preserves_no_selection(tmp_path) -> None:
    source_grid = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    scenario = BOIN12ReportScenario("all toxic <unsafe>", source_grid)
    source_grid[0] = [1.0, 0.0, 0.0, 0.0]
    assert scenario.joint_probability.flags.writeable is False
    assert scenario._grid[0] == (0.0, 0.0, 1.0, 0.0)
    seed = 2**63 + 17
    cutoff = np.array(0.95)
    design = BOIN12Design(0.35, 0.25, toxicity_cutoff=cutoff)
    report = boin12_report(
        design,
        (scenario,),
        cohorts=3,
        cohort_size=3,
        trials=8,
        seed=seed,
    )
    cutoff[...] = 0.5
    assert report.design.toxicity_cutoff == 0.95
    output = tmp_path / "report.html"
    assert report.write_html(output) == output
    rendered = output.read_text(encoding="utf-8")
    assert "all toxic &lt;unsafe&gt;" in rendered
    assert "No selection" in rendered
    assert "P(noT,E)" in rendered and "P(T,noE)" in rendered
    assert "<td>1</td><td>0</td><td>0</td><td>1</td><td>0</td>" in rendered
    assert str(seed) in rendered
    assert report.scenarios[0].obd_probability[0] == 1.0
    assert report.scenarios[0].mtd_probability[0] == 1.0
