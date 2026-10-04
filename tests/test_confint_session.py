"""Contract checks for the bounded CONFINT calculation log."""

import numpy as np
import pytest

from mdanderson_stats.confint_session import ConfintSession


def test_named_calculations_record_defaults_submodes_and_diagnostics():
    session = ConfintSession()
    calls = [
        (
            "confint_normal_probability",
            {"sample_size": 20, "max_length": 1, "population_sd": 1, "target": "mean"},
        ),
        (
            "confint_normal_sd_limit",
            {"sample_size": 20, "max_length": 1, "assurance": 0.5, "target": "sd"},
        ),
        (
            "confint_normal_sample_size",
            {"max_length": 1, "population_sd": 1, "assurance": 0.5, "target": "mean"},
        ),
        (
            "confint_normal_probability",
            {
                "sample_size": 20,
                "sample_size2": 30,
                "max_length": 1,
                "population_sd": 1,
                "target": "mean_difference",
            },
        ),
        (
            "confint_binomial_probability",
            {"sample_size": 250, "max_length": 0.1, "event_probability": 0.25},
        ),
        (
            "confint_poisson_probability",
            {"exposure": 210, "max_length": 1, "event_rate": 20},
        ),
        (
            "confint_binomial_difference_probability",
            {
                "sample_size1": 2,
                "sample_size2": 3,
                "event_probability1": 0.5,
                "event_probability2": 0.4,
                "max_length": 0.6,
            },
        ),
        (
            "confint_survival_fixed_events",
            {"events": (0, 1, 2), "hazard": 0.2, "max_length": 1},
        ),
        (
            "confint_survival_probability",
            {
                "hazard": 0.2,
                "accrual_rate": 5,
                "accrual_time": 10,
                "followup_time": 0,
                "max_length": 0.5,
            },
        ),
        (
            "confint_survival_probability",
            {
                "hazard": 0.2,
                "accrual_rate": 5,
                "accrual_time": 10,
                "followup_time": 0,
                "max_length": 0.5,
                "target": "mean",
            },
        ),
        (
            "confint_survival_solve",
            {
                "hazard": 1,
                "accrual_rate": 5,
                "accrual_time": 10,
                "followup_time": 0,
                "max_length": None,
                "assurance": 0.5,
                "bounds": (0.1, 2),
            },
        ),
        (
            "confint_survival_hazard_range",
            {
                "accrual_rate": 5,
                "accrual_time": 10,
                "followup_time": 0,
                "max_length": 0.5,
                "assurance": 0.5,
                "grid_points": 17,
            },
        ),
    ]
    for procedure, settings in calls:
        session, result = session.calculate(procedure, **settings)
        assert result is not None

    report = session.report()
    assert session.calculation_count == 12
    assert "total CI width" in report
    assert "assurance" in report
    assert "confidence" in report
    assert "population_sd" in report
    assert "target': 'mean_difference'" in report
    assert "Pooled equal-variance two-sample mean-difference" in report
    assert "one-sample mean confidence interval" in report
    assert "population-SD confidence interval" in report
    assert "survival mean interval" in report
    assert "'confidence': 0.95" in report
    assert "'bounds': (0.1, 2)" in report
    assert "omitted_probability" in report
    assert "lower_clipped" in report


def test_snapshots_preserve_unattainable_limits_and_write_readable_report(tmp_path):
    session, probability = ConfintSession().calculate(
        "confint_binomial_probability",
        sample_size=np.int64(250),
        max_length=0.1,
        event_probability=0.25,
    )
    report_before_mutation = session.report()
    assert probability.flags.writeable is False
    probability.setflags(write=True)
    probability[...] = -1
    assert session.report() == report_before_mutation
    session, impossible = session.calculate(
        "confint_binomial_event_limit", sample_size=10, max_length=0.01
    )
    assert impossible is None
    report = session.report()
    assert "No event-probability range attains the requested assurance" in report
    assert "'confidence': 0.95" in report
    assert "'assurance': 0.9" in report
    assert session.report() == report

    path = session.write_report(tmp_path / "confint.txt")
    assert path.read_text(encoding="utf-8") == report
    assert "Clopper-Pearson" in report


def test_rejects_unbounded_array_inputs_before_kernel_and_preserves_file(tmp_path, monkeypatch):
    empty = ConfintSession()
    with pytest.raises(TypeError, match="must be scalar"):
        empty.calculate(
            "confint_normal_probability",
            sample_size=np.ones(5001, dtype=int),
            max_length=1,
            population_sd=1,
        )
    assert empty.calculation_count == 0
    with pytest.raises(TypeError, match="two-value tuple"):
        empty.calculate(
            "confint_survival_solve",
            hazard=1,
            accrual_rate=5,
            accrual_time=10,
            followup_time=0,
            max_length=None,
            assurance=0.5,
            bounds=[0.1, 2],
        )

    session, _ = empty.calculate(
        "confint_poisson_probability", exposure=10, max_length=1, event_rate=0.1
    )
    path = tmp_path / "existing.txt"
    path.write_text("original", encoding="utf-8")

    def fail_report(self):
        raise ValueError("report rendering failed")

    monkeypatch.setattr(ConfintSession, "report", fail_report)
    with pytest.raises(ValueError, match="report rendering failed"):
        session.write_report(path)
    assert path.read_text(encoding="utf-8") == "original"
