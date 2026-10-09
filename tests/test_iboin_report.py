import json

import numpy as np
import pytest

from mdanderson_stats import (
    IBOINDesign,
    replay_iboin_report,
    simulate_iboin_report,
    simulate_iboin_trial,
)


def report(**overrides):
    options = dict(
        cohort_size=3,
        max_patients=12,
        repetitions=4,
        prior_mode="effective",
        isotonic_weights=[1, 2, 3],
        seed=20261008,
        eligible_doses=[True, True, False],
        tie_policy="highest",
        enforce_deescalation_boundary=True,
        titration_cap=2,
    )
    options.update(overrides)
    return simulate_iboin_report(
        IBOINDesign([0.1, 0.25, 0.4], [2, 3, 4], robust_prior=True, extra_safe=True),
        [0.1, 0.1, 0.1],
        [0.01, 0.2, 0.5],
        **options,
    )


def test_captured_inputs_replay_exact_trials_and_summary():
    original = report()
    inputs = json.loads(original.inputs_json)
    assert inputs["design"]["prior_ess"] == [2, 3, 4]
    assert inputs["effective_prior_ess"] == [2, 3, 0]
    assert inputs["simulation"]["isotonic_weights"] == [1, 2, 3]
    assert inputs["simulation"]["eligible_doses"] == [True, True, False]
    repeated = replay_iboin_report(original.inputs_json)
    assert repeated.inputs_json == original.inputs_json
    for name in vars(original.results):
        actual, expected = getattr(repeated.results, name), getattr(original.results, name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected
    settings = inputs["simulation"].copy()
    seeds = settings.pop("trial_seeds")
    settings.pop("repetitions")
    settings.pop("max_total_work")
    settings.pop("max_total_storage_bytes")
    trials = [
        simulate_iboin_trial(IBOINDesign(**inputs["design"]), seed=s, **settings) for s in seeds
    ]
    np.testing.assert_array_equal(
        original.results.selected_dose,
        [trial.selection.selected_dose or 0 for trial in trials],
    )
    np.testing.assert_allclose(
        original.results.mean_patients_by_dose,
        np.mean([trial.replay.patients for trial in trials], axis=0),
    )
    np.testing.assert_allclose(
        original.results.mean_dlt_by_dose,
        np.mean([trial.replay.toxicities for trial in trials], axis=0),
    )


def test_uint64_seeds_survive_standard_json_and_replay():
    original = report(seed=None, trial_seeds=np.array([2**64 - 1, 2**63, 0, 1], dtype=np.uint64))
    assert json.loads(original.inputs_json)["simulation"]["trial_seeds"][0] == 2**64 - 1
    np.testing.assert_array_equal(
        replay_iboin_report(original.inputs_json).results.total_patients,
        original.results.total_patients,
    )


def test_html_escapes_titles_and_saves_complete_inputs_atomically(tmp_path):
    original = report(title='<script>alert("x")</script>')
    output = original.write_html(tmp_path / "report.html")
    text = output.read_text()
    assert "<script>" not in text
    assert "&lt;script&gt;" in text
    assert "Original ESS" in text and "Effective ESS" in text
    assert "No MTD selected" in text
    assert "MCSE" in text and "Stopping probabilities" in text
    saved = original.write_inputs(tmp_path / "inputs.json")
    repeated = replay_iboin_report(saved.read_text())
    np.testing.assert_array_equal(original.results.trial_seeds, repeated.results.trial_seeds)
    assert {p.name for p in tmp_path.iterdir()} == {"report.html", "inputs.json"}


def test_single_repetition_reports_undefined_mcse_without_nonstandard_json():
    single = report(repetitions=1)
    assert "unavailable (one repetition)" in single.to_html()
    assert "NaN" not in single.inputs_json
    assert np.isnan(single.results.selection_mcse).all()


@pytest.mark.parametrize("alteration", ["version", "unknown", "missing", "effective", "policy"])
def test_replay_rejects_changed_or_incomplete_schema(alteration):
    inputs = json.loads(report().inputs_json)
    if alteration == "version":
        inputs["format_version"] = True
    elif alteration == "unknown":
        inputs["simulation"]["invented"] = 1
    elif alteration == "missing":
        del inputs["simulation"]["tie_policy"]
    elif alteration == "effective":
        inputs["effective_prior_ess"] = [2, 3, 4]
    else:
        inputs["simulation"]["prior_mode"] = "native"
    with pytest.raises(ValueError):
        replay_iboin_report(json.dumps(inputs))


@pytest.mark.parametrize("text", ['{"title": "a", "title": "b"}', '{"x": NaN}', "[]", "{"])
def test_replay_rejects_invalid_json(text):
    with pytest.raises(ValueError):
        replay_iboin_report(text)


def test_simulation_resource_limits_are_enforced_before_report_generation():
    with pytest.raises(ValueError, match="work"):
        report(max_total_work=1)
    with pytest.raises(ValueError, match="storage"):
        report(max_total_storage_bytes=1)
    with pytest.raises(ValueError, match="title"):
        report(title="bad\nname")
    with pytest.raises(ValueError, match="4 MiB"):
        replay_iboin_report(" " * (4 * 1024 * 1024 + 1))


def test_report_inputs_are_independent_of_caller_arrays():
    truth = np.array([0.01, 0.2, 0.5])
    weights = np.array([1.0, 2.0, 3.0])
    original = simulate_iboin_report(
        IBOINDesign([0.1, 0.25, 0.4], [0, 0, 0]),
        [0, 0, 0],
        truth,
        cohort_size=3,
        max_patients=9,
        repetitions=2,
        prior_mode="none",
        isotonic_weights=weights,
        seed=4,
    )
    truth[:] = 0
    weights[:] = 100
    inputs = json.loads(original.inputs_json)
    assert inputs["simulation"]["dlt_probability"] == [0.01, 0.2, 0.5]
    assert inputs["simulation"]["isotonic_weights"] == [1, 2, 3]
    assert not original.results.trial_seeds.flags.writeable


@pytest.mark.parametrize("seed", [-1, 2**64, True, 1.5, "1"])
def test_replay_rejects_seed_coercion(seed):
    inputs = json.loads(report().inputs_json)
    inputs["simulation"]["trial_seeds"][0] = seed
    with pytest.raises(ValueError, match="trial_seeds"):
        replay_iboin_report(json.dumps(inputs))


def test_failed_atomic_replace_preserves_existing_report_and_removes_temporary(
    tmp_path, monkeypatch
):
    import mdanderson_stats.iboin_report as module

    destination = tmp_path / "report.html"
    destination.write_text("existing report")

    def fail_replace(*args):
        raise OSError("replacement failed")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        report().write_html(destination)
    assert destination.read_text() == "existing report"
    assert list(tmp_path.iterdir()) == [destination]
