from collections import Counter
from importlib import import_module

import numpy as np
import pytest

from mdanderson_stats.bf_boin import BFBOINDesign
from mdanderson_stats.bf_boin_report import BFBOINReportScenario, bf_boin_design_report
from mdanderson_stats.bf_boin_simulation import simulate_bf_boin


def test_report_replays_scenarios_serially_and_captures_guide_options():
    scenarios = (
        BFBOINReportScenario("<low & moderate>", [0.10, 0.25], [0.2, 0.4]),
        BFBOINReportScenario("high", [0.35, 0.5], [0.1, 0.2]),
    )
    design = BFBOINDesign(target=0.25, n_cap=3, stay_at_one_of_three=True, extra_safe=True)
    report = bf_boin_design_report(
        design,
        scenarios,
        cohorts=2,
        cohort_size=2,
        trials=8,
        seed=719,
    )
    generator = np.random.default_rng(719)
    for source, summary in zip(scenarios, report.scenarios, strict=True):
        direct = simulate_bf_boin(
            design,
            source.true_toxicity,
            source.true_response,
            cohorts=2,
            cohort_size=2,
            trials=8,
            rng=generator,
        )
        np.testing.assert_array_equal(summary.selection_probability, direct.selection_probability)
        np.testing.assert_array_equal(summary.mean_assigned_by_dose, direct.mean_assigned)
        np.testing.assert_array_equal(summary.mean_dlt_by_dose, direct.mean_toxicities)
        assert summary.stop_reason_frequency == tuple(
            (reason, count / 8) for reason, count in sorted(Counter(direct.stop_reason).items())
        )

    html = report.to_html()
    assert "Stay at 1 DLT among 3 at current dose" in html
    assert "n>3" in html
    assert "&lt;low &amp; moderate&gt;" in html
    assert "Effective lowest-dose stop" in html


def test_report_preflights_complete_work_before_simulation(monkeypatch):
    module = import_module("mdanderson_stats.bf_boin_report")
    called = False

    def unexpected_simulation(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("simulation must not start when the record bound fails")

    monkeypatch.setattr(module, "simulate_bf_boin", unexpected_simulation)
    with pytest.raises(ValueError, match="one scenario trial may retain"):
        bf_boin_design_report(
            BFBOINDesign(n_cap=6),
            (BFBOINReportScenario("scenario", [0.1, 0.2], [0.1, 0.2]),),
            cohorts=1_000,
            cohort_size=1,
            trials=1,
            seed=7,
        )
    assert not called
