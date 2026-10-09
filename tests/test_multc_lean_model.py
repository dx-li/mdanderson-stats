import csv
import inspect
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    MultcLeanEndpointInput,
    MultcLeanModel,
    MultcLeanScenario,
    parse_multc_lean_model,
    read_multc_lean_model,
    replay_multc_lean_study,
)

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE = json.loads((FIXTURES / "multc-lean-managed-model.json").read_text())


def assert_restored(model, restored):
    for endpoint, key in (
        (model.response, "m_responseParams"),
        (model.toxicity, "m_toxicityParams"),
    ):
        values = restored[key]
        for field in (
            "standard_a",
            "standard_b",
            "experimental_a",
            "experimental_b",
            "use_standard_constant",
            "standard_constant",
        ):
            assert getattr(endpoint, field) == values[field]
    assert model.max_subjects == restored["m_maxPatients"]
    assert model.min_subjects == restored["m_minPatients"]
    assert model.cohort_size == restored["m_cohortSize"]
    assert model.response_cutoff == restored["m_responseParams"]["pi"]
    assert model.response_margin == restored["m_responseParams"]["delta"]
    assert model.toxicity_cutoff == restored["m_toxicityParams"]["pi"]
    assert model.toxicity_margin == restored["m_toxicityParams"]["delta"]
    values = restored["m_arrayScenarioParams"] or []
    assert len(model.scenarios) == len(values)
    for s, v in zip(model.scenarios, values, strict=True):
        assert s.joint_probabilities == (
            v["P_Res_Tox"],
            v["P_Res_NoTox"],
            v["P_NoRes_Tox"],
            v["P_NoRes_NoTox"],
        )
        assert s.mean_interarrival == v["meanInterArrivalTime"]
        assert s.response_window == v["window"]


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["case"])
def test_original_managed_file_order_and_values(case, tmp_path):
    model = parse_multc_lean_model(case["text"])
    assert_restored(model, case["restored"])
    assert parse_multc_lean_model(model.to_model_text()) == model
    path = tmp_path / "MultcLeanDesktop.model"
    model.write_model(path)
    assert read_multc_lean_model(path) == model


def simple_model(*, scenarios=(), **changes):
    endpoint = MultcLeanEndpointInput(1, 1, 1, 1, True, 0.5)
    return MultcLeanModel(
        endpoint,
        endpoint,
        max_subjects=3,
        response_cutoff=1,
        toxicity_cutoff=1,
        scenarios=scenarios,
        **changes,
    )


def test_original_reset_defaults_and_simulation_repetition_default():
    defaults = REFERENCE["defaults"]
    assert_restored(MultcLeanModel.example(), {**defaults, "m_arrayScenarioParams": None})
    assert (
        inspect.signature(MultcLeanModel.run).parameters["trials"].default
        == (REFERENCE["default_repetitions"])
    )
    original = MultcLeanModel.example().to_design()
    assert original.pretrial_check
    assert original.max_subjects == 30
    assert original.historical_response.constant is None


def test_file_preserves_inactive_history_fields_and_full_float_precision():
    endpoint = MultcLeanEndpointInput(0, -17, 0.12345678901234568, 1.8765432109876543, True, 0.5)
    model = MultcLeanModel(endpoint, endpoint)
    exported = model.to_model_text()
    replay = parse_multc_lean_model(exported)
    assert replay == model
    assert replay.response.standard_b == -17
    assert replay.to_design().historical_response.constant == 0.5
    assert format(endpoint.experimental_a, ".17g") in exported


def test_bom_and_native_line_endings():
    text = simple_model().to_model_text()
    for ending in ("\n", "\r", "\r\n"):
        assert parse_multc_lean_model("\ufeff" + text.replace("\n", ending)) == simple_model()


def test_inactive_history_rejects_finite_values_outside_float64_range():
    if np.finfo(np.longdouble).max <= np.finfo(float).max:
        pytest.skip("platform longdouble has no wider finite range")
    with pytest.raises(ValueError, match="representable"):
        MultcLeanEndpointInput(np.finfo(np.longdouble).max, 1, 1, 1, True, 0.5)


@pytest.mark.parametrize("response,toxicity", [(0, 0), (1, 1), (0.3, 0.25), (0.1, 0.9)])
def test_independent_category_conversion(response, toxicity):
    truth = MultcLeanScenario.independent(response, toxicity)
    p = truth.joint_probabilities
    assert p[0] + p[1] == pytest.approx(response)
    assert p[0] + p[2] == pytest.approx(toxicity)
    assert p[0] == pytest.approx(response * toxicity)


@pytest.mark.parametrize("cohort", [1, 2])
def test_imported_model_exact_ocs_match_independent_r_path_enumeration(cohort):
    text = (
        "1 1 1 1 True 0.45\n1 1 1 1 True 0.3\n"
        f"6 2 {cohort} 0.8 0 0.85 0\n2\n"
        "0.1 0.45 0.15 0.3 0 0\n0.25 0.3 0 0.45 0 0\n"
    )
    study = parse_multc_lean_model(text).run(seed=40)
    with (FIXTURES / "multc-path-enumeration.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for i, result in enumerate(study.results, 1):
        expected = np.zeros(7)
        for row in rows:
            if int(row["cohort"]) == cohort and int(row["scenario"]) == i:
                expected[int(row["n"])] += float(row["probability"])
        np.testing.assert_allclose(
            result.exact.sample_size_probability, expected, atol=2e-14, rtol=2e-13
        )
        assert result.simulation is None


def test_mixed_timing_preserves_exact_estimates_and_replays_legacy_seeds(tmp_path):
    model = simple_model(
        scenarios=(
            MultcLeanScenario((0, 0, 0, 1), 0.25, 2),
            MultcLeanScenario((0.1, 0.2, 0.3, 0.4), 0, 3),
            MultcLeanScenario((0.25, 0.25, 0.25, 0.25)),
            MultcLeanScenario((0.25, 0.25, 0.25, 0.25), 3, 0),
        )
    )
    study = model.run(
        seed=90,
        trials=32,
        max_unit_exponentials_per_trial=10,
        scenario_names=("timed", "zero interarrival", "disabled", "zero window"),
        show_potential_boundaries=True,
    )
    simulation = study.results[0].simulation
    assert simulation is not None
    assert simulation.summary.mean_sample_size == 3
    assert simulation.summary.mean_responses == simulation.summary.mean_toxicities == 0
    assert simulation.summary.mean_balks == 0
    assert all(r.simulation is None for r in study.results[1:])
    assert all(float(r.exact.expected_sample_size) == pytest.approx(3) for r in study.results)
    assert study.results[1].exact.expected_responses == pytest.approx(0.9)
    assert study.results[1].exact.expected_toxicities == pytest.approx(1.2)
    replay = replay_multc_lean_study(study.to_json())
    assert replay.to_json() == study.to_json()
    second = replay.results[0].simulation
    assert second is not None
    np.testing.assert_array_equal(simulation.trial_seeds, second.trial_seeds)
    np.testing.assert_array_equal(simulation.summary.duration, second.summary.duration)
    assert simulation.replay_trial(0).duration == simulation.summary.duration[0]
    study.write_json(tmp_path / "inputs.json")
    study.write_protocol(tmp_path / "protocol.md")
    study.write_report(tmp_path / "report.html")
    report = (tmp_path / "report.html").read_text()
    assert "Exact count estimates" in report
    assert "Legacy timing Monte Carlo estimates" in report
    assert "Reachable stopping intervals" in report
    assert "cap completion" in report
    assert "Duration simulation disabled" in report
    assert "prior screen precedes enrollment" in report
    assert "0.25" in report


def test_default_10000_replicates_for_a_prior_rejected_study():
    model = replace(
        simple_model(scenarios=(MultcLeanScenario((0, 0, 0, 1), 1, 2),)), response_cutoff=0
    )
    study = model.run(seed=43, max_unit_exponentials_per_trial=1)
    simulation = study.results[0].simulation
    assert simulation is not None
    assert simulation.summary.trials == 10000
    assert simulation.summary.mean_duration == simulation.summary.mean_sample_size == 0
    assert study.results[0].exact.sample_size_probability[0] == 1
    assert simulation.summary.sample_size_probability[0] == 1


def test_names_are_escaped_and_plot_values_match_exact_cdf_and_pmf():
    import matplotlib.pyplot as plt

    study = replace(
        simple_model(scenarios=(MultcLeanScenario((0, 0, 0, 1)),)),
        max_subjects=6,
        response_cutoff=0.95,
    ).run(seed=4, scenario_names=("<script>alert('x')</script>",))
    html = study.to_html()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    for cumulative in (True, False):
        figure = study.plot(cumulative=cumulative)
        try:
            line = figure.axes[0].lines[0]
            np.testing.assert_array_equal(line.get_xdata(), np.arange(7))
            expected = [0, 0, 0, 0, 1, 1, 1] if cumulative else [0, 0, 0, 0, 1, 0, 0]
            np.testing.assert_array_equal(line.get_ydata(), expected)
        finally:
            plt.close(figure)


def test_empty_model_runs_bounds_and_produces_an_editable_protocol():
    study = simple_model().run(seed=1)
    assert study.results == ()
    assert "# Multc Lean study protocol" in study.protocol()
    assert replay_multc_lean_study(study.to_json()).model == study.model
    with pytest.raises(ValueError, match="at least one"):
        study.plot()


@pytest.mark.parametrize(
    "change,match",
    [
        ({"seed": -1}, "seed"),
        ({"trials": True}, "trials"),
        ({"max_total_exact_work": 1}, "max_total_exact_work"),
        ({"max_total_draws": 1}, "max_total_draws"),
        ({"max_storage_bytes": 1}, "max_storage_bytes"),
        ({"scenario_names": ["a", "a"]}, "scenario_names"),
        ({"scenario_names": ["bad\n", "b"]}, "scenario_names"),
        ({"show_potential_boundaries": 1}, "show_potential_boundaries"),
    ],
)
def test_aggregate_input_and_work_checks_precede_rng(monkeypatch, change, match):
    model = simple_model(scenarios=(MultcLeanScenario((0, 0, 0, 1), 0.2, 2),) * 2)

    def forbidden(*args, **kwargs):
        pytest.fail("RNG created before preflight")

    monkeypatch.setattr(np.random, "Generator", forbidden)
    with pytest.raises(ValueError, match=match):
        model.run(**{**dict(seed=3, trials=2), **change})


@pytest.mark.parametrize(
    "text",
    [
        "",
        "% headers only\n",
        "\n" + REFERENCE["cases"][0]["text"],
        REFERENCE["cases"][0]["text"].replace("False", "yes"),
        REFERENCE["cases"][0]["text"].replace("30 70", "30  70"),
        REFERENCE["cases"][0]["text"].replace("30 70", "30\t70"),
        REFERENCE["cases"][0]["text"].replace("30 70", "NaN 70"),
        REFERENCE["cases"][0]["text"].replace("30 70", "1e999 70"),
        REFERENCE["cases"][0]["text"].replace("30 70", "0x1e 70"),
        REFERENCE["cases"][0]["text"].replace("30 70", "3_0 70"),
        REFERENCE["cases"][0]["text"].replace("30 1 1", "30.0 1 1"),
        REFERENCE["cases"][0]["text"] + "extra data\n",
        REFERENCE["cases"][0]["text"] + "\n",
        REFERENCE["cases"][0]["text"].replace("\n0\r\n", "\n-1\n"),
        REFERENCE["cases"][0]["text"].replace("\n0\r\n", "\n101\n"),
        REFERENCE["cases"][1]["text"].replace("0.1 0.2 0.3 0.4", "0.1 0.2 0.3 0.5"),
        REFERENCE["cases"][1]["text"].replace("0.1 0.2", "-0.1 0.2"),
        REFERENCE["cases"][0]["text"].replace("30 70", "0 70"),
        REFERENCE["cases"][0]["text"].replace("0.6 1.4", "101 1.4"),
        REFERENCE["cases"][0]["text"].replace("30 1 1", "30 5 3"),
        REFERENCE["cases"][0]["text"].replace("0.95 0 0.95 0", "0.95 -0.1 0.95 0.1"),
        REFERENCE["cases"][0]["text"].replace("0.95 0 0.95 0", "1.1 0 0.95 0"),
        "x" * 1_048_577,
    ],
)
def test_invalid_native_text_raises_instead_of_resetting_defaults(text):
    with pytest.raises(ValueError):
        parse_multc_lean_model(text)


def test_file_size_precedes_decode_and_model_data_is_immutable(tmp_path):
    path = tmp_path / "large.model"
    path.write_bytes(b"\xff" * 1_048_577)
    with pytest.raises(ValueError, match="1 MiB"):
        read_multc_lean_model(path)
    truth = [0.1, 0.2, 0.3, 0.4]
    scenarios = [MultcLeanScenario(truth)]
    model = simple_model(scenarios=scenarios)
    truth[0] = 0.9
    scenarios.clear()
    assert model.scenarios[0].joint_probabilities == (0.1, 0.2, 0.3, 0.4)
    with pytest.raises(FrozenInstanceError):
        model.max_subjects = 6


@pytest.mark.parametrize(
    "change",
    [
        {"format_version": True},
        {"format_version": 2},
        {"extra": 3},
        {"options": {}},
        {"model_text": "bad"},
    ],
)
def test_invalid_replay_json_fields_are_rejected(change):
    raw = json.loads(simple_model().run(seed=2).to_json())
    raw.update(change)
    with pytest.raises(ValueError):
        replay_multc_lean_study(json.dumps(raw))


def test_duplicate_nonfinite_and_oversized_json_rejected():
    for text in ('{"x":1,"x":2}', '{"x":NaN}', "[]", "x" * 1_048_577):
        with pytest.raises(ValueError):
            replay_multc_lean_study(text)


@pytest.mark.parametrize(
    "args",
    [
        {"joint_probabilities": [0, 0, 0, 1], "mean_interarrival": -1},
        {"joint_probabilities": [0, 0, 0, 1], "response_window": 10001},
        {"joint_probabilities": [0, 0, 0, True]},
        {"joint_probabilities": [0, 0, 0]},
    ],
)
def test_invalid_scenario_domain(args):
    with pytest.raises(ValueError):
        MultcLeanScenario(**args)


def test_oversized_or_bad_scenario_lists_rejected():
    with pytest.raises(ValueError, match="100"):
        MultcLeanModel.example(scenarios=[MultcLeanScenario((0, 0, 0, 1))] * 101)
    with pytest.raises(ValueError, match="MultcLeanScenario"):
        simple_model(scenarios=[None])
    with pytest.raises(ValueError, match="boolean"):
        MultcLeanEndpointInput(1, 1, 1, 1, 1, 0.5)
