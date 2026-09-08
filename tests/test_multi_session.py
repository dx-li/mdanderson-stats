"""Integration checks for data replacement, procedure dispatch and report provenance."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import BetaMixture, MultiSession, schweder_fit, sharpened_testing

TEXT = ".001 .01 .02 .1 .2 .3 .4 .5 .6 .7 .8 .9"


def test_report_tracks_defaults_original_order_and_immutable_results(tmp_path):
    session = MultiSession(".8,.2 .2 .1", source="entered")
    result = session.run("multiple_testing", method="holm", alpha=0.1)
    report = session.report()
    (dataset,) = report["datasets"]
    assert dataset["entered_values"] == [0.8, 0.2, 0.2, 0.1]
    assert dataset["order"] == [3, 1, 2, 0]
    assert dataset["warnings"][0]["token"] == ","
    event = report["events"][-1]
    assert event["settings"] == {"method": "holm", "alpha": 0.1}
    assert event["result"]["method"] == "holm"
    assert_allclose(event["result"]["adjusted_pvalues"], result.adjusted_pvalues)
    expected = event["result"]["adjusted_pvalues"].copy()
    result.adjusted_pvalues[:] = 0
    event["result"]["adjusted_pvalues"][:] = [1] * 4
    assert session.report()["events"][-1]["result"]["adjusted_pvalues"] == expected
    path = session.write_report(tmp_path / "report.json")
    assert json.loads(path.read_text()) == session.report()
    session.write_report(path)  # Explicit replacement, never duplicate append events.
    assert len(json.loads(path.read_text())["events"]) == 2


def test_data_change_cannot_reuse_old_schweder_estimate():
    session = MultiSession(TEXT)
    session.run("schweder_fit")
    first = session.run("sharpened_testing")
    replacement = ".001 .002 .003 .004 .005 .1 .2 .3 .4 .5 .6 .7"
    session.change_data(replacement, source="second")
    result = session.run("sharpened_testing")
    current = session.data.entered_values
    estimate = schweder_fit(current).null_estimate
    assert_array_equal(result, sharpened_testing(current, estimate))
    report = session.report()
    assert report["events"][-1]["dataset"] == 2
    assert report["events"][-1]["settings"]["null_estimate"] == estimate
    assert report["events"][2]["dataset"] == 1
    assert len(first) == len(result)


def test_invalid_replacement_preserves_session_and_report(tmp_path):
    session = MultiSession(TEXT)
    before = session.report()
    with pytest.raises(ValueError):
        session.change_data(".1 .2")
    with pytest.raises(FileNotFoundError):
        session.change_file(tmp_path / "missing")
    assert session.report() == before
    data = session.data
    data.pvalues[:] = 0
    assert session.data.pvalues[-1] == 0.9
    path = tmp_path / "next.txt"
    path.write_text(".1 .2 .3 .4")
    session.change_file(path)
    assert session.report()["datasets"][-1]["source"] == str(path)
    assert len(session.data.pvalues) == 4


def test_rng_can_be_reset_and_state_replayed():
    session = MultiSession(TEXT, seed=42)
    first = session.run("schweder_bootstrap", samples=5)
    event = session.report()["events"][-1]
    generator = np.random.default_rng()
    generator.bit_generator.state = event["settings"]["rng"]["generator_state"]
    second = session.run("schweder_bootstrap", samples=5, rng=generator)
    assert_array_equal(first.estimates, second.estimates)
    session.set_seed(42)
    third = session.run("schweder_bootstrap", samples=5)
    assert_array_equal(first.estimates, third.estimates)
    assert "rng_after" in event


def test_all_statistical_groups_produce_serializable_report_results():
    session = MultiSession(TEXT, seed=123)
    session.run("order_statistic_diagnostics")
    session.run("nonparametric_testing", null_estimate=5)
    fitted = session.run("fit_beta_mixture_k", k=0)
    session.run("beta_mixture_testing", model=fitted.model, sequence="input")
    session.run("beta_mixture_bootstrap", model=fitted.model, replicates=2)
    session.run("select_beta_mixture", max_components=1, algorithm="em")
    report = session.report()
    assert report["events"][-1]["settings"]["workflow"] == "desktop"
    assert report["events"][-1]["result"]["status"]
    json.dumps(report, allow_nan=False)
    assert set(session.procedures) >= {
        "multiple_testing",
        "schweder_bootstrap",
        "select_beta_mixture",
    }


def test_failed_numerical_run_is_not_reported_as_success():
    session = MultiSession(".1 .1 .1 .1")
    with pytest.raises(ValueError):
        session.run("schweder_fit")
    event = session.report()["events"][-1]
    assert event["status"] == "failed" and "result" not in event
    assert event["error"]["type"] == "SchwederFitError"
    with pytest.raises(ValueError, match="Unknown"):
        session.run("not_a_procedure")
    with pytest.raises(TypeError):
        session.run("multiple_testing", misspelled_alpha=0.1)


def test_terminal_input_and_nonfinite_diagnostics_are_preserved():
    session = MultiSession("0 .2 .3 1 q .4", terminal=True)
    session.run("beta_mixture_testing", model=BetaMixture(0, [1], [0.5], [2]))
    report = session.report()
    assert_allclose(report["datasets"][0]["entered_values"], [0, 0.2, 0.3, 1])
    text = json.dumps(report, allow_nan=False)
    assert "nonfinite" in text  # Infinite endpoint log densities remain explicit.


def test_automatic_null_fit_failure_is_recorded_with_its_provenance():
    session = MultiSession(".1 .1 .1 .1")
    with pytest.raises(ValueError):
        session.run("sharpened_testing")
    event = session.report()["events"][-1]
    assert event["status"] == "failed"
    assert event["null_estimate_source"].startswith("schweder_fit")
    assert "result" not in event
