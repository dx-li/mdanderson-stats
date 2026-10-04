from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest

from mdanderson_stats.bayes_factor_survival_calendar import simulate_bayes_factor_survival

report_module = import_module("mdanderson_stats.bayes_factor_survival_report")


def _report(**overrides):
    values = {
        "null_median": 4.0,
        "alternative_median_mode": 5.5,
        "accrual_rate": 2.0,
        "max_patients": 3,
        "repetitions": 3,
        "check_times": [1.0, 2.0],
        "final_followup": 1.0,
        "true_medians": [4.0, 5.5],
        "seed": 123,
        "time_unit": "months",
        "max_total_work": 36,
        "max_total_quadratures": 18,
    }
    values.update(overrides)
    return report_module.bayes_factor_survival_report(**values)


def test_report_replays_child_seeds_and_uses_terminal_not_final_monitor_rates():
    report = _report(boundary_events=[0, 1], max_boundary_rows=2)
    children = np.random.SeedSequence(123).spawn(2)
    for summary, truth, child in zip(report.scenarios, [4.0, 5.5], children, strict=True):
        seed = int(child.generate_state(1, dtype=np.uint64)[0])
        assert summary.seed == seed
        direct = simulate_bayes_factor_survival(
            null_median=4.0,
            alternative_median_mode=5.5,
            true_median=truth,
            accrual_rate=2.0,
            max_patients=3,
            repetitions=3,
            check_times=[1.0, 2.0],
            final_followup=1.0,
            seed=seed,
            max_total_work=36,
            max_total_quadratures=18,
        )
        early = direct.early_inferiority | direct.early_superiority
        expected_i = direct.early_inferiority | (~early & direct.final_inferiority)
        expected_s = direct.early_superiority | (~early & direct.final_superiority)
        np.testing.assert_allclose(
            [summary.terminal_inferiority_probability, summary.terminal_superiority_probability],
            [expected_i.mean(), expected_s.mean()],
        )
        np.testing.assert_allclose(
            [summary.early_inferiority_probability, summary.early_superiority_probability],
            [direct.early_inferiority.mean(), direct.early_superiority.mean()],
        )
        np.testing.assert_allclose(
            [summary.final_inferiority_probability, summary.final_superiority_probability],
            [direct.final_inferiority.mean(), direct.final_superiority.mean()],
        )
        assert summary.mean_patients_enrolled == direct.mean_patients_enrolled
        np.testing.assert_array_equal(
            summary.patient_count_quantiles, direct.patient_count_quantiles
        )
    assert report.boundaries is not None
    np.testing.assert_array_equal(report.boundaries.events, [0, 1])


def test_terminal_stop_keeps_early_classification_when_final_monitor_reverses():
    terminal_i, terminal_s = report_module._terminal_stop_flags(
        np.array([True, False, False]),
        np.array([False, False, False]),
        np.array([False, True, False]),
        np.array([True, False, True]),
    )
    np.testing.assert_array_equal(terminal_i, [True, True, False])
    np.testing.assert_array_equal(terminal_s, [False, False, True])


def test_aggregate_and_boundary_row_preflight_happen_before_kernel_work(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("kernel work must not start before report preflight")

    monkeypatch.setattr(report_module, "simulate_bayes_factor_survival", unexpected)
    monkeypatch.setattr(report_module, "bayes_factor_survival_boundaries", unexpected)
    with pytest.raises(ValueError, match="aggregate worst-case scenario work"):
        _report(max_total_work=35)
    with pytest.raises(ValueError, match="aggregate worst-case scenario evaluations"):
        _report(max_total_quadratures=17)
    with pytest.raises(ValueError, match="max_boundary_rows"):
        _report(boundary_events=[0, 1], max_boundary_rows=1)


def test_html_write_escapes_labels_and_preserves_continuous_boundaries(tmp_path):
    report = _report(
        boundary_events=[0],
        max_boundary_rows=1,
        time_unit="months & <months>",
        title="TTE <analysis>",
    )
    html = report.to_html()
    assert "TTE &lt;analysis&gt;" in html
    assert "months &amp; &lt;months&gt;" in html
    assert "&amp;amp;" not in html
    assert "continuous roots" in html
    assert "integer-day rounding" in html

    destination = tmp_path / "report.html"
    destination.write_text("previous", encoding="utf-8")
    invalid = replace(report, title="bad\nlabel")
    with pytest.raises(ValueError, match="control characters"):
        invalid.write_html(destination)
    assert destination.read_text(encoding="utf-8") == "previous"
    assert report.write_html(destination) == destination
    assert destination.read_text(encoding="utf-8") == html
