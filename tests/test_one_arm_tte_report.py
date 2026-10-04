import importlib

import numpy as np
import pytest

from mdanderson_stats.one_arm_tte import one_arm_tte_design
from mdanderson_stats.one_arm_tte_report import (
    OneArmTTEScenario,
    simulate_one_arm_tte_scenarios,
)
from mdanderson_stats.one_arm_tte_simulation import simulate_one_arm_tte


def _design():
    return one_arm_tte_design(
        [2, 3],
        [1, 2],
        parameterization="mean",
        maximize=True,
        cutoff_inferiority=0,
        delta_inferiority=0,
        cutoff_superiority=None,
        max_patients=4,
        minimum_patients=1,
        monitor_at_accrual=True,
        followup_period=1,
    )


def test_scenario_report_matches_seeded_individual_simulations_and_is_compact():
    design = _design()
    scenarios = (
        OneArmTTEScenario("slow < control", 8, 1.5, 2**53 + 1),
        OneArmTTEScenario("fast & active", 2, 0.75, 202),
    )
    report = simulate_one_arm_tte_scenarios(
        design, scenarios, 3, credible_level=0.8, time_unit="months"
    )

    assert report.scenarios[0].name == scenarios[0].name
    assert report.scenarios[0].seed == 2**53 + 1
    assert report.scenarios[1].seed == 202
    assert report.max_total_monitoring_checks == 100_000
    assert report.scenarios[0].total_monitoring_checks == 9
    for scenario, summary in zip(scenarios, report.scenarios, strict=True):
        raw = simulate_one_arm_tte(
            design,
            scenario.true_tte,
            scenario.accrual_rate,
            3,
            seed=scenario.seed,
            credible_level=0.8,
        )
        assert summary.mean_sample_size == np.mean(raw.sample_sizes)
        assert summary.mean_events == np.mean(raw.events)
        assert summary.mean_exposure == pytest.approx(np.mean(raw.exposures))
        assert summary.mean_final_time == pytest.approx(np.mean(raw.final_times))
        assert summary.sample_size_quantiles == tuple(raw.sample_size_quantiles)
        assert summary.duration_quantiles == tuple(raw.duration_quantiles)
        assert summary.early_inferior_probability == raw.early_inferior_probability
        assert summary.early_inferior_mcse == raw.early_inferior_mcse
        assert summary.final_superior_probability == raw.final_superior_probability
        assert summary.final_superior_mcse == raw.final_superior_mcse
        assert not hasattr(summary, "sample_sizes")


def test_report_preflights_all_scenarios_and_enforces_actual_aggregate_check_budget(
    monkeypatch,
):
    module = importlib.import_module("mdanderson_stats.one_arm_tte_report")
    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("simulation should not start")

    monkeypatch.setattr(module, "simulate_one_arm_tte", forbidden)
    with pytest.raises(ValueError, match="unique ignoring case"):
        simulate_one_arm_tte_scenarios(
            _design(),
            (OneArmTTEScenario("Case", 2, 1, 1), OneArmTTEScenario("case", 3, 1, 2)),
            1,
        )
    with pytest.raises(ValueError, match="aggregate potential patients"):
        simulate_one_arm_tte_scenarios(_design(), (OneArmTTEScenario("valid", 2, 1, 1),), 25_001)
    assert calls == 0

    monkeypatch.undo()
    with pytest.raises(ValueError, match="max_monitoring_checks"):
        simulate_one_arm_tte_scenarios(
            _design(),
            (OneArmTTEScenario("a", 2, 1, 1), OneArmTTEScenario("b", 3, 1, 2)),
            3,
            max_total_monitoring_checks=17,
        )


def test_report_html_escapes_user_text_and_writes_atomically(tmp_path):
    report = simulate_one_arm_tte_scenarios(
        _design(),
        (OneArmTTEScenario("<script>alert('x')</script>", 2, 1, 7),),
        2,
        time_unit="month & <unit>",
    )
    html = report.to_html()
    assert "&lt;script&gt;" in html
    assert "month &amp; &lt;unit&gt;" in html
    assert "Aggregate accrual-phase check ceiling: 100000" in html
    assert "<script>alert('x')</script>" not in html
    destination = report.write_html(tmp_path / "nested" / "report.html")
    assert destination.read_text(encoding="utf-8") == html


def test_zero_monitor_budget_allows_final_only_design():
    design = one_arm_tte_design(
        [2, 3],
        [1, 2],
        cutoff_inferiority=0,
        max_patients=1,
        minimum_patients=1,
        monitor_at_accrual=True,
    )
    report = simulate_one_arm_tte_scenarios(
        design,
        (OneArmTTEScenario("single", 2, 1, 3),),
        2,
        max_total_monitoring_checks=0,
    )
    assert report.scenarios[0].total_monitoring_checks == 0
