import numpy as np
import pytest

from mdanderson_stats.top_binary import TOPBinaryDesign
from mdanderson_stats.top_endpoints import TOPMultiEndpointDesign
from mdanderson_stats.top_multi_simulation import simulate_top_multiendpoint
from mdanderson_stats.top_report import (
    TOPBinaryScenario,
    TOPMultiEndpointScenario,
    top_binary_report,
    top_multiendpoint_report,
)
from mdanderson_stats.top_simulation import simulate_top_binary


def test_binary_report_matches_engine_and_writes_escaped_atomic_html(tmp_path):
    design = TOPBinaryDesign(4, 0.2, 0.8, 0.5, looks=[2, 4], suspension="strict")
    scenario = TOPBinaryScenario("fixed <arrival>", 0.4, 2.0, 1.5, 8, 123, arrival="fixed")
    report = top_binary_report(design, [scenario])
    direct = simulate_top_binary(design, 0.4, 2.0, 1.5, trials=8, arrival="fixed", rng=123)
    case = report.cases[0]
    assert case.seed == 123 and case.trials == 8
    assert case.inputs[3] == ("arrival gaps", "fixed")
    for action in case.actions:
        count = int(np.count_nonzero(direct.decision == action.name))
        assert action.count == count
        assert action.denominator == 8
        assert action.probability == count / 8
        assert action.mcse == pytest.approx(
            np.sqrt(action.probability * (1 - action.probability) / 8)
        )
    html = report.to_html()
    assert "fixed &lt;arrival&gt;" in html
    assert "Exact design boundaries" in html and "MCSE" in html
    destination = report.write_html(tmp_path / "report.html")
    assert destination.read_text(encoding="utf-8") == html


@pytest.mark.parametrize("mode", ["coprimary", "efficacy_toxicity"])
def test_multiendpoint_report_preserves_mode_truth_and_simulated_actions(mode):
    design = TOPMultiEndpointDesign(
        4,
        [0.12, 0.28, 0.18, 0.42],
        0.8,
        0.5,
        mode=mode,
        windows=[2, 3],
        looks=[2, 4],
        suspension="strict",
    )
    scenario = TOPMultiEndpointScenario(
        "joint truth",
        (0.2, 0.3, 0.1, 0.4),
        1.2,
        7,
        77,
        arrival="fixed",
        truth_timing_probabilities=(0.5, 0.3, 0.2),
    )
    report = top_multiendpoint_report(design, [scenario])
    direct = simulate_top_multiendpoint(
        design,
        scenario.joint_probabilities,
        scenario.accrual_rate,
        trials=scenario.trials,
        arrival=scenario.arrival,
        truth_timing_probabilities=scenario.truth_timing_probabilities,
        rng=scenario.seed,
    )
    case = report.cases[0]
    assert case.seed == scenario.seed
    assert "11,10,01,00" in case.inputs[0][0]
    assert case.inputs[-1][0] == "truth timing by endpoint and window third"
    for action in case.actions:
        count = int(np.count_nonzero(direct.decision == action.name))
        assert action.count == count and action.denominator == scenario.trials
    assert report.settings[1] == ("mode", mode)
    assert any("endpoint 2" in row[1] for row in report.boundaries.rows) == (mode == "coprimary")


def test_report_preflights_later_invalid_scenario_and_bounds_integer_seed(monkeypatch):
    import mdanderson_stats.top_report as report_module

    design = TOPBinaryDesign(4, 0.2, 0.8, 0.5, looks=[2, 4])
    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("invalid batch must be rejected before simulation")

    monkeypatch.setattr(report_module, "simulate_top_binary", forbidden)
    scenarios = [
        TOPBinaryScenario("valid", 0.3, 2, 1, 4, 1),
        TOPBinaryScenario("invalid", 0.3, -2, 1, 4, 2),
    ]
    with pytest.raises(ValueError, match="window/rate"):
        report_module.top_binary_report(design, scenarios)
    assert calls == 0
    with pytest.raises(ValueError, match=r"2\*\*64"):
        report_module.top_binary_report(design, [TOPBinaryScenario("seed", 0.3, 2, 1, 4, 2**64)])
