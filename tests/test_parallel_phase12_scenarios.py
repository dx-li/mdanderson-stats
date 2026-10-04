from importlib import import_module

import numpy as np
import pytest

from mdanderson_stats import simulate_parallel_phase12_oc
from mdanderson_stats.parallel_phase12_scenarios import (
    ParallelPhase12Scenario,
    parallel_phase12_scenario_from_native_input,
    simulate_parallel_phase12_scenarios,
)


def test_named_scenario_batch_replays_and_reports_captured_inputs(tmp_path):
    scenarios = (
        ParallelPhase12Scenario("low response", [0.1] * 4, [0.05] * 4, 2, 8501),
        ParallelPhase12Scenario("high response", [0.1] * 4, [0.7] * 4, 2, 8502, (3,)),
    )
    batch = simulate_parallel_phase12_scenarios(scenarios)
    assert [item.label for item in batch.scenarios] == ["low response", "high response"]
    for scenario, actual in zip(scenarios, batch.scenarios, strict=True):
        expected = simulate_parallel_phase12_oc(
            scenario.toxicity_probability,
            scenario.efficacy_probability,
            n_trials=scenario.n_trials,
            seed=scenario.seed,
            optimal_arms=scenario.optimal_arms,
        )
        np.testing.assert_array_equal(
            actual.operating_characteristics.per_trial_seeds, expected.per_trial_seeds
        )
        np.testing.assert_array_equal(
            actual.operating_characteristics.selection_probability,
            expected.selection_probability,
        )
        assert actual.seed == scenario.seed
        assert actual.optimal_arms == scenario.optimal_arms

    rendered = batch.report()
    assert "Fixed design: N<=100" in rendered
    assert "selection_mcse=" in rendered
    assert "mean_treated_mcse=" in rendered
    assert "toxicity_rate_mcse=" in rendered
    assert "optimal_arms=(3,)" in rendered
    assert "optimal_selection_mcse=" in rendered
    destination = tmp_path / "scenario-report.txt"
    assert batch.write_report(destination) == destination
    assert destination.read_text(encoding="utf-8") == rendered


def test_all_scenario_work_is_preflighted_before_any_simulation(monkeypatch):
    module = import_module("mdanderson_stats.parallel_phase12_scenarios")
    called = False

    def unexpected_call(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("scenario simulation must not begin")

    monkeypatch.setattr(module, "simulate_parallel_phase12_oc", unexpected_call)
    scenarios = tuple(
        ParallelPhase12Scenario(f"scenario-{i}", [0.1] * 4, [0.2] * 4, 1_000, i) for i in range(11)
    )
    with pytest.raises(ValueError, match="aggregate scenario workload"):
        module.simulate_parallel_phase12_scenarios(scenarios)
    assert not called


def test_native_parameter_order_is_efficacy_then_toxicity():
    scenario = parallel_phase12_scenario_from_native_input(
        "0.1 0.2 0.3 0.4\n0.05 0.10 0.15 0.20",
        label="from C parameter file",
        n_trials=12,
        seed=8507,
    )
    assert scenario.efficacy_probability == (0.1, 0.2, 0.3, 0.4)
    assert scenario.toxicity_probability == (0.05, 0.10, 0.15, 0.20)
    for text in ("0.1 0.2 0.3 0.4 0.1 0.2 0.3", "0.1 0.2 0.3 0.4 0.1 0.2 0.3 0.4 0.5"):
        with pytest.raises(ValueError, match="exactly eight"):
            parallel_phase12_scenario_from_native_input(text, label="bad", n_trials=1, seed=1)
    with pytest.raises(ValueError, match="8192-character"):
        parallel_phase12_scenario_from_native_input(
            " " * 8193, label="oversized", n_trials=1, seed=1
        )


def test_input_validation_rejects_ambiguous_scenarios():
    with pytest.raises(ValueError, match="four-vector"):
        ParallelPhase12Scenario("bad", [[0.1] * 4], [0.2] * 4, 1, 1)
    with pytest.raises(ValueError, match="probabilities"):
        ParallelPhase12Scenario("bad", [0.1, 0.2, 0.3, 1.1], [0.2] * 4, 1, 1)
    with pytest.raises(ValueError, match="optimal_arms"):
        ParallelPhase12Scenario("bad", [0.1] * 4, [0.2] * 4, 1, 1, ([0] * 100_000,))
    valid = ParallelPhase12Scenario("duplicate", [0.1] * 4, [0.2] * 4, 1, 1)
    with pytest.raises(ValueError, match="unique"):
        simulate_parallel_phase12_scenarios((valid, valid))
