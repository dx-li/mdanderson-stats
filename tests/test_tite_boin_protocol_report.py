import numpy as np
import pytest

import mdanderson_stats.tite_boin_protocol_report as protocol_report
from mdanderson_stats.boin import BOINDesign
from mdanderson_stats.tite_boin_simulation import simulate_tite_boin


def _request(**overrides):
    values = {
        "trial_name": "Study <A & B>",
        "design": BOINDesign(target=0.3, early_stop_patients=4),
        "scenarios": (
            protocol_report.TITEBOINScenario("low", [0.05, 0.15, 0.25]),
            protocol_report.TITEBOINScenario("high", [0.1, 0.3, 0.6]),
        ),
        "window": 30,
        "time_unit": "days",
        "accrual_rate": 1 / 10,
        "cohorts": 2,
        "cohort_size": 2,
        "start_dose": 1,
        "trials": 30,
        "seed": 2718,
    }
    values.update(overrides)
    return protocol_report.TITEBOINProtocolRequest(**values)


def test_protocol_report_reruns_each_scenario_from_recorded_seed_and_summarizes():
    report = protocol_report.run_tite_boin_protocol(_request())
    assert [scenario.name for scenario in report.scenarios] == ["low", "high"]
    assert report.seed == 2718 and report.scenarios[0].seed != report.scenarios[1].seed

    direct = simulate_tite_boin(
        _request().design,
        [0.05, 0.15, 0.25],
        30,
        1 / 10,
        cohorts=2,
        cohort_size=2,
        trials=30,
        start_dose=1,
        rng=report.scenarios[0].seed,
    )
    first = report.scenarios[0]
    np.testing.assert_array_equal(first.selection_probability, direct.selection_probability)
    np.testing.assert_array_equal(first.selection_mcse, direct.selection_mcse)
    np.testing.assert_allclose(first.mean_patients_by_dose, direct.patients.mean(axis=0))
    np.testing.assert_allclose(first.mean_toxicities_by_dose, direct.toxicities.mean(axis=0))
    assert first.mean_duration == pytest.approx(direct.duration.mean())
    assert first.mean_suspension_time == pytest.approx(direct.suspension_time.mean())
    assert sum(probability for _, probability in first.stop_reason_probability) == pytest.approx(1)
    assert not first.true_toxicity.flags.writeable
    assert not first.selection_probability.flags.writeable


def test_html_escapes_labels_and_states_actual_decision_precedence():
    request = _request(
        design=BOINDesign(target=0.3, deescalate_at_two_of_six=True),
        scenarios=(protocol_report.TITEBOINScenario("<unsafe & named>", [0.1, 0.2, 0.3]),),
        trials=12,
        cohorts=2,
        cohort_size=3,
    )
    text = protocol_report.run_tite_boin_protocol(request).to_html(digits=17)
    assert "Study &lt;A &amp; B&gt;" in text
    assert "&lt;unsafe &amp; named&gt;" in text
    assert (
        "2/6 rule (target 0.28–0.33) forces de-escalation and overrides completion suspension"
        in text
    )
    assert "minimum pending follow-up" in text
    assert "wait until all enrolled DLT outcomes are ascertained" in text
    assert "not the application's HTML/Word/PDF template" in text
    assert "effective masses by third 0.33333333333333331" in text
    assert "uniform conditional time-to-DLT weights" in text


def test_exact_requested_boundaries_and_effective_parametric_timing_defaults_are_captured():
    design = BOINDesign.from_boundaries(0.3, 0.2, 0.4)
    report = protocol_report.run_tite_boin_protocol(
        _request(
            design=design,
            scenarios=(protocol_report.TITEBOINScenario("timing", [0.1, 0.3]),),
            event_distribution="weibull",
            late_probability=0.7,
            trials=8,
        )
    )
    assert report.escalation_boundary == design.escalation_boundary
    assert report.deescalation_boundary == design.deescalation_boundary
    assert report.safe_probability == design.safe_probability
    assert report.toxic_probability == design.toxic_probability
    assert report.late_probability_defaulted is False
    np.testing.assert_array_equal(report.late_probability, [0.7, 0.7])
    assert report.event_trimester_probabilities_defaulted
    assert report.trimester_probabilities_defaulted
    text = report.to_html()
    assert "0.2; 0.4" in text
    assert "0.7, 0.7" in text
    assert "not used with parametric event-time generation" in text


def test_output_and_aggregate_work_limits_are_checked_before_simulation(monkeypatch):
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("simulation must not start")

    monkeypatch.setattr(protocol_report, "simulate_tite_boin", forbidden)
    with pytest.raises(ValueError, match="trial-by-dose"):
        protocol_report.run_tite_boin_protocol(
            _request(
                scenarios=(protocol_report.TITEBOINScenario("wide", [0.01] * 100),),
                trials=10_001,
                cohorts=1,
                cohort_size=1,
            )
        )
    with pytest.raises(ValueError, match="aggregate simulated patient"):
        protocol_report.run_tite_boin_protocol(
            _request(
                scenarios=(
                    protocol_report.TITEBOINScenario("one", [0.1, 0.2]),
                    protocol_report.TITEBOINScenario("two", [0.1, 0.2]),
                ),
                trials=100_000,
                cohorts=20,
                cohort_size=2,
            )
        )
    assert not called


def test_invalid_truth_mass_and_unit_label_fail_before_simulation():
    with pytest.raises(ValueError, match="true_toxicity probabilities"):
        protocol_report.run_tite_boin_protocol(
            _request(scenarios=(protocol_report.TITEBOINScenario("bad", [0.1, 1.1]),))
        )
    with pytest.raises(ValueError, match="summing to one"):
        protocol_report.run_tite_boin_protocol(_request(trimester_probabilities=[0.2, 0.2, 0.2]))
    with pytest.raises(ValueError, match="time_unit"):
        protocol_report.run_tite_boin_protocol(_request(time_unit="x" * 257))


def test_write_html_is_utf8_and_replaces_destination(tmp_path):
    report = protocol_report.run_tite_boin_protocol(
        _request(scenarios=(protocol_report.TITEBOINScenario("é", [0.1, 0.2, 0.3]),), trials=5)
    )
    destination = tmp_path / "nested" / "protocol.html"
    destination.parent.mkdir()
    destination.write_text("old", encoding="utf-8")
    assert report.write_html(destination) == destination
    assert destination.read_text(encoding="utf-8") == report.to_html()
    assert "é" in destination.read_text(encoding="utf-8")
