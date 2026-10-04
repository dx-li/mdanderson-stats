"""Reproducible TITE-Keyboard protocol and scenario-report workflow."""

import importlib

import numpy as np
import pytest

from mdanderson_stats.keyboard import KeyboardDesign
from mdanderson_stats.tite_keyboard_simulation import simulate_tite_keyboard

report_module = importlib.import_module("mdanderson_stats.tite_keyboard_protocol_report")


def _request(**overrides):
    values = {
        "trial_name": "Study <A & B>",
        "design": KeyboardDesign(target=0.30),
        "scenarios": (report_module.TITEKeyboardScenario("low <truth>", [0.05, 0.20, 0.35], 2),),
        "window": 30,
        "accrual_rate": 1 / 10,
        "cohorts": 2,
        "cohort_size": 2,
        "trials": 30,
        "seed": 2718,
    }
    values.update(overrides)
    return report_module.TITEKeyboardProtocolRequest(**values)


def test_report_replays_captured_scenario_and_renders_boundaries(tmp_path):
    report = report_module.run_tite_keyboard_protocol(_request())
    assert report.seed == 2718
    assert len(report.scenarios) == 1
    scenario = report.scenarios[0]
    direct = simulate_tite_keyboard(
        report_module._capture_design(_request().design),
        scenario.true_toxicity,
        30,
        1 / 10,
        cohorts=2,
        cohort_size=2,
        trials=30,
        rng=scenario.scenario_seed,
        true_mtd=2,
    )
    np.testing.assert_array_equal(scenario.selection_probability, direct.selection_probability)
    np.testing.assert_array_equal(scenario.selection_mcse, direct.selection_mcse)
    np.testing.assert_allclose(scenario.mean_patients_by_dose, direct.patients.mean(axis=0))
    np.testing.assert_allclose(scenario.mean_toxicities_by_dose, direct.toxicities.mean(axis=0))
    assert scenario.mean_duration == pytest.approx(direct.duration.mean())
    assert scenario.mean_suspension_time == pytest.approx(direct.suspension_time.mean())
    assert sum(row[1] for row in scenario.stop_reason_probability) == pytest.approx(1)
    assert scenario.true_toxicity.flags.writeable is False
    assert scenario.selection_probability.flags.writeable is False
    assert scenario.allocation_risks == direct.allocation_risks
    assert scenario.allocation_risks is not None

    content = report.to_html(digits=17)
    assert "Study &lt;A &amp; B&gt;" in content
    assert "low &lt;truth&gt;" in content
    assert "effective non-DLT counts immediately below and at" in content
    assert "native report-layout parity is not asserted" in content
    assert "Risk of &lt;6 patients at true MTD (dose 2)" in content
    path = tmp_path / "report.html"
    report.write_html(path, digits=17)
    assert path.read_text(encoding="utf-8") == content


def test_timing_inputs_are_captured_without_silent_discard():
    report = report_module.run_tite_keyboard_protocol(
        _request(event_distribution="weibull", late_probability=0.7, trials=5)
    )
    np.testing.assert_array_equal(report.late_probability, [0.7, 0.7, 0.7])
    assert report.late_probability_defaulted is False
    assert report.event_trimester_probabilities_defaulted
    with pytest.raises(ValueError, match="only used with parametric"):
        report_module.run_tite_keyboard_protocol(_request(late_probability=0.7))


def test_budget_and_mutually_exclusive_adaptive_analysis_inputs_fail_preflight():
    adaptive = report_module.TITEKeyboardAdaptiveSettings(
        lambda_prior=(2.0, 1.0),
        gamma_prior=(2.0, 1.0),
        chains=2,
        draws=20,
        warmup=10,
        max_fit_work=1000,
        max_total_work=10_000,
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        report_module.run_tite_keyboard_protocol(
            _request(adaptive_timing=adaptive, trimester_probabilities=[1 / 3] * 3)
        )


def test_adaptive_report_records_child_seeds_diagnostics_and_work():
    adaptive = report_module.TITEKeyboardAdaptiveSettings(
        lambda_prior=(0.5, 0.5),
        gamma_prior=(0.5, 0.5),
        chains=2,
        draws=64,
        warmup=16,
        max_fit_work=5_000_000,
        max_total_work=20_000_000,
        max_split_rhat=100.0,
        max_weight_mcse=1.0,
    )
    report = report_module.run_tite_keyboard_protocol(
        _request(
            design=KeyboardDesign(target=0.2),
            scenarios=(report_module.TITEKeyboardScenario("adaptive", [0.1, 0.2], 1),),
            cohorts=2,
            cohort_size=3,
            trials=2,
            pending_fraction_limit=None,
            adaptive_timing=adaptive,
            max_aggregate_adaptive_work=20_000_000,
        )
    )
    result = report.scenarios[0]
    assert result.allocation_risks is not None
    assert result.allocation_risks.true_mtd == 1
    assert result.adaptive_fit_count > 0
    assert result.adaptive_diagnostics_passed == result.adaptive_fit_count
    assert result.outcome_seed is not None
    assert result.sampler_seed is not None
    assert result.outcome_seed != result.sampler_seed
    assert 0 < result.adaptive_work_units <= result.adaptive_work_budget
    assert result.adaptive_work_budget == 20_000_000
    assert report.trimester_probabilities is None
    assert report.trimester_probabilities_defaulted is False
    text = report.to_html()
    assert "Maximum adaptive split R-hat / pending-weight MCSE" in text
    assert "not used with adaptive timing" in text
    assert f"outcome stream seed: {result.outcome_seed}" in text
    assert f"timing-sampler seed: {result.sampler_seed}" in text
