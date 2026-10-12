import csv
import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mdanderson_stats import PRTStudy, read_prt_study
from mdanderson_stats.prt_conduct import _analyze_prt_history
from mdanderson_stats.prt_legacy import PRTLegacyScreening
from mdanderson_stats.prt_simulation import (
    PRTSimulationBatch,
    _history,
    run_prt_simulation_calendar,
    simulate_prt,
    simulate_prt_batch,
)

FIXTURES = Path(__file__).parent / "fixtures"
PATHS = json.loads((FIXTURES / "prt-desktop-enrollment.json").read_text())["cases"]
PRECISION = json.loads((FIXTURES / "prt-desktop-calendar-precision.json").read_text())["cases"]


def study(**kwargs):
    return replace(
        PRTStudy(
            3,
            3,
            interval_endpoints=(3, 7, 10),
            max_patients=7,
            true_toxicity=(0.1, 0.2, 0.3),
            mean_interarrival=5,
            hazard_alpha=5,
            hazard_beta=1,
            repetitions=2,
            prior_mean=-1.5,
            prior_variance=0.05,
            draws=128,
            warmup=100,
            chains=4,
        ),
        **kwargs,
    )


@pytest.mark.parametrize("case", PRECISION)
def test_native_precision_observation(case):
    result = _history(
        study(interval_endpoints=(25, 50, 100)),
        [0],
        [case["arrival"]],
        [case["delay"]],
        case["clock"],
    )
    assert result.patients[0].censored == bool(case["censored"])


@pytest.mark.parametrize("case", PATHS)
def test_composed_native_enrollment_paths(monkeypatch, case):
    # Supplied native decisions isolate the calendar; screening/controller each have
    # separate executable fixtures. This does not claim full posterior/RNG parity.
    monkeypatch.setattr(
        "mdanderson_stats.prt_simulation.prt_legacy_screening",
        lambda *a, **kw: PRTLegacyScreening(True, False, "injected"),
    )
    decisions = iter(case["decisions"][1:])
    observed = []

    def analysis(history, **kwargs):
        observed.append(history.patients)
        if kwargs["controller"] == "simulation_final":
            return SimpleNamespace(status="computed", action="select", dose=0, rule="7")
        suspend, dose = next(decisions)
        return SimpleNamespace(
            status="computed",
            action="suspend" if suspend else "stay",
            dose=None if suspend else dose - 1,
            rule="fixture",
        )

    monkeypatch.setattr("mdanderson_stats.prt_simulation._analyze_prt_history", analysis)
    potential = np.full((len(case["arrivals"]), 3), 11.0)
    index = patient = 0
    for suspend, dose in case["decisions"]:
        if suspend:
            index += 1
        else:
            cohort = min(3, case["maximum"] - patient)
            for j in range(cohort):
                potential[index + j, dose - 1] = case["delays"][patient + j]
            index += cohort
            patient += cohort
    result = run_prt_simulation_calendar(
        study(max_patients=case["maximum"]),
        case["arrivals"],
        potential,
        rng=np.random.default_rng(1),
        profile="legacy_1_0",
    )
    assert len(result.steps) == len(case["result"])
    for step, patients, expected in zip(result.steps, observed, case["result"], strict=True):
        assert (step.time, step.candidate_index, step.enrolled) == (
            expected["clock"],
            expected["candidate_index"],
            expected["enrolled"],
        )
        assert [p.dose + 1 for p in patients] == expected["doses"]
        assert [int(p.censored) for p in patients] == expected["censored"]
        assert [p.observed_interval for p in patients] == expected["interval"]
    assert result.patient_arrivals == tuple(case["result"][-1]["arrivals"])


def test_screening_suspends_declines_and_final_skipped_is_unresolved():
    result = run_prt_simulation_calendar(
        study(max_patients=3),
        [0, 0, 0],
        np.full((3, 3), 11.0),
        rng=np.random.default_rng(1),
        profile="legacy_1_0",
    )
    assert result.status == "final_inference_skipped"
    assert result.selected_dose is None
    assert result.assigned == (3, 0, 0) and result.events == (0, 0, 0)
    assert all(p.observed_interval == 0 for p in result.patients_as_of_end)
    assert json.loads(result.to_json())["steps"][0]["analysis"] is None
    assert len(list(csv.DictReader(io.StringIO(result.to_csv())))) == 3
    result = run_prt_simulation_calendar(
        study(),
        np.arange(6) * 0.1,
        np.full((6, 3), 11.0),
        rng=np.random.default_rng(1),
        profile="legacy_1_0",
    )
    assert result.status == "arrival_exhausted"
    assert result.declined == 3 and result.assigned == (3, 0, 0)
    assert all(s.action == "suspend" for s in result.steps)


def test_budget_is_not_reported_as_safety_stop():
    result = run_prt_simulation_calendar(
        study(),
        np.arange(20) * 0.1,
        np.full((20, 3), 11.0),
        rng=np.random.default_rng(1),
        profile="legacy_1_0",
        max_analyses=1,
    )
    assert result.status == "analysis_budget"
    assert result.selected_dose is None and len(result.steps) == 1
    batch = PRTSimulationBatch(study(), (result, replace(result, status="safety_stop")), "{}")
    summary = batch.summary()
    assert summary["selection_probability"] == [0, 0, 0]
    assert summary["safety_stop_probability"] == summary["unresolved_probability"] == 0.5
    assert not summary["complete"]
    assert json.loads(batch.to_json())["summary"] == summary
    assert list(csv.DictReader(io.StringIO(batch.to_csv())))[0]["unresolved"] == "1"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"profile": "safe"},
        {"max_mcse": 0.1},
        {"max_rhat": 1},
        {"max_fits": False},
        {"max_analyses": 801},
        {"max_likelihood_evaluations": 0},
    ],
)
def test_invalid_settings_preserve_rng(kwargs):
    rng = np.random.default_rng(1)
    before = repr(rng.bit_generator.state)
    with pytest.raises(ValueError):
        simulate_prt(study(), rng=rng, **({"profile": "legacy_1_0"} | kwargs))
    assert repr(rng.bit_generator.state) == before


def test_generated_saved_replay_and_real_sampler():
    original = study(repetitions=1)
    result = simulate_prt(
        read_prt_study(original.to_json()),
        rng=np.random.default_rng(123),
        profile="legacy_1_0",
        max_analyses=10,
    )
    assert len(result.arrival_times) == 800
    assert result.generation_state_json is not None
    restored = np.random.default_rng()
    restored.bit_generator.state = json.loads(result.rng_state_json)
    replay = run_prt_simulation_calendar(
        original,
        result.arrival_times,
        result.potential_delays,
        rng=restored,
        profile="legacy_1_0",
        max_analyses=10,
    )
    assert replace(result, generation_state_json=None).to_json() == replay.to_json()
    assert any(s.analysis is not None for s in result.steps)
    assert all(
        s.analysis.controller in ("simulation", "simulation_final")
        for s in result.steps
        if s.analysis is not None
    )
    assert result.status in ("selected", "safety_stop", "numerical_hold", "final_inference_skipped")


def test_generated_batch_reproducible_and_never_discards_holds():
    a = simulate_prt_batch(
        study(), rng=np.random.default_rng(2), profile="legacy_1_0", max_analyses=1
    )
    b = simulate_prt_batch(
        study(), rng=np.random.default_rng(2), profile="legacy_1_0", max_analyses=1
    )
    assert a.to_json() == b.to_json()
    assert len(a.trials) == 2
    s = a.summary()
    assert (
        sum(s["selection_probability"]) + s["safety_stop_probability"] + s["unresolved_probability"]
        == 1
    )


def test_simulation_controller_is_distinct_and_final_is_selected(monkeypatch):
    from test_prt_conduct import synthetic_fit

    monkeypatch.setattr(
        "mdanderson_stats.prt_conduct.fit_prt_model", lambda *a, **kw: synthetic_fit()
    )
    inputs = study(draws=2048)
    history = _history(inputs, [0, 0, 0], [0, 1, 2], [11, 11, 11], 20)
    interim = _analyze_prt_history(
        history, rng=np.random.default_rng(1), profile="legacy_1_0", controller="simulation"
    )
    assert interim.status == "computed" and interim.action == "escalate" and interim.dose == 1
    final = _analyze_prt_history(
        history, rng=np.random.default_rng(1), profile="legacy_1_0", controller="simulation_final"
    )
    assert final.status == "computed" and final.action == "select" and final.dose == 0
    assert json.loads(final.to_json())["schema"] == "mdanderson_stats.prt.simulation_final"


def test_real_multistage_completion_with_quality_gates():
    inputs = study(
        doses=2,
        true_toxicity=(0.1, 0.2),
        max_patients=6,
        draws=2048,
        warmup=1000,
        prior_variance=0.005,
    )
    result = run_prt_simulation_calendar(
        inputs,
        np.arange(20) * 20,
        np.full((20, 2), 11.0),
        rng=np.random.default_rng(123),
        profile="legacy_1_0",
    )
    assert result.status == "selected" and result.selected_dose == 0
    assert [s.action for s in result.steps] == ["escalate", "select"]
    assert result.assigned == (3, 3) and result.events == (0, 0)
    assert result.duration == 110
    for step in result.steps:
        assert step.analysis.chain_decisions_agree
        assert step.analysis.max_split_rhat < 1.01
        assert step.analysis.max_risk_mcse < 0.001


def test_early_stop_has_no_future_event_leakage(monkeypatch):
    def stop(*a, **kw):
        return SimpleNamespace(status="computed", action="stop", dose=None, rule="3")

    monkeypatch.setattr("mdanderson_stats.prt_simulation._analyze_prt_history", stop)
    result = run_prt_simulation_calendar(
        study(),
        [0, 1, 2, 4, 5, 6, 7, 8],
        np.full((8, 3), 5.0),
        rng=np.random.default_rng(1),
        profile="legacy_1_0",
    )
    assert result.status == "safety_stop" and result.duration == 4
    assert result.events == (0, 0, 0) and result.assigned == (3, 0, 0)


def test_preflight_rejects_batch_work_before_rng():
    rng = np.random.default_rng(1)
    before = repr(rng.bit_generator.state)
    with pytest.raises(ValueError, match="budget"):
        simulate_prt_batch(study(repetitions=101), rng=rng, profile="legacy_1_0")
    assert repr(rng.bit_generator.state) == before
