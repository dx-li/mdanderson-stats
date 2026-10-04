from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.keyboard import KeyboardDesign
from mdanderson_stats.keyboard_report import keyboard_protocol_report
from mdanderson_stats.keyboard_simulation import simulate_keyboard


def test_protocol_report_matches_serial_simulations_and_writes_static_html(tmp_path: Path):
    design = KeyboardDesign(target=0.3, extra_safe=True, early_stop_patients=9)
    scenarios = [[0.08, 0.22, 0.38], [0.12, 0.30, 0.55], [0.10, 0.22, 0.50]]
    report = keyboard_protocol_report(
        design,
        scenarios,
        cohort_size=3,
        cohorts=4,
        start_dose=1,
        trials=80,
        seed=2026,
        title="Phase I <Keyboard & safety>",
    )

    generator = np.random.default_rng(2026)
    for scenario, summary in zip(scenarios, report.scenarios, strict=True):
        expected = simulate_keyboard(
            report.design,
            scenario,
            cohort_size=3,
            cohorts=4,
            start_dose=1,
            trials=80,
            rng=generator,
        )
        np.testing.assert_allclose(summary.selection_probability, expected.selection_probability)
        np.testing.assert_allclose(summary.selection_mcse, expected.selection_mcse)
        np.testing.assert_allclose(summary.mean_patients, expected.mean_patients)
        np.testing.assert_allclose(summary.mean_toxicities, expected.mean_toxicities)
        assert sum(value for _, value in summary.stop_frequency) == pytest.approx(1)
        if any(value == report.design.target for value in scenario):
            above_target_patients = expected.patients[
                :, np.asarray(scenario) > report.design.target
            ].sum(axis=1)
            for fraction, risk_probability, risk_mcse in zip(
                (0.6, 0.8),
                summary.overdose_allocation_probability or (),
                summary.overdose_allocation_mcse or (),
                strict=True,
            ):
                indicator = above_target_patients > fraction * (3 * 4)
                assert risk_probability == pytest.approx(indicator.mean())
                assert risk_mcse == pytest.approx(
                    np.sqrt(indicator.mean() * (1 - indicator.mean()) / 80)
                )
        else:
            assert summary.overdose_allocation_probability is None
            assert summary.overdose_allocation_mcse is None
            assert summary.overdose_allocation_unavailable_reason
    assert report.boundaries.patients[-1] == 12

    destination = tmp_path / "protocol.html"
    assert report.write_html(destination) == destination
    html = destination.read_text(encoding="utf-8")
    assert "Phase I &lt;Keyboard &amp; safety&gt;" in html
    assert "NumPy integer seed" in html and "2026" in html
    assert "uniform Beta(1,1)" in html and "weak Beta(.05,.05)" in html
    assert "Extra-safe lowest-dose stop if enabled" in html
    assert "More than 60% of planned enrollment above target" in html
    assert (
        "source summary is defined only when at least one true dose probability equals target"
        in html
    )
    assert "not the MD Anderson app's HTML/Word template" in html
    assert "https://" not in html


def test_report_preflights_aggregate_work_before_running_simulations(monkeypatch):
    def should_not_run(*args, **kwargs):
        raise AssertionError("simulation started before aggregate work preflight")

    monkeypatch.setattr("mdanderson_stats.keyboard_report.simulate_keyboard", should_not_run)
    scenarios = np.tile([0.1, 0.2], (20, 1))
    with pytest.raises(ValueError, match="aggregate simulated patient replications"):
        keyboard_protocol_report(
            KeyboardDesign(),
            scenarios,
            cohort_size=10,
            cohorts=20,
            trials=10000,
            seed=1,
        )


def test_report_rejects_invalid_scenario_shape_and_seed_before_simulation():
    with pytest.raises(ValueError, match="same dose count"):
        keyboard_protocol_report(
            KeyboardDesign(),
            [[0.1, 0.2], [0.1, 0.2, 0.3]],
            cohort_size=3,
            cohorts=4,
            seed=1,
        )
    with pytest.raises(ValueError, match="seed must be an integer"):
        keyboard_protocol_report(
            KeyboardDesign(),
            [[0.1, 0.2]],
            cohort_size=3,
            cohorts=4,
            seed=True,
        )
