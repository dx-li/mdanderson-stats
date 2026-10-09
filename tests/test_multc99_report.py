import json

import numpy as np
import pytest

from mdanderson_stats.multc99 import Multc99Event, multc99_design
from mdanderson_stats.multc99_randomized import simulate_multc99_randomized
from mdanderson_stats.multc99_report import multc99_study_report, replay_multc99_study
from mdanderson_stats.multc99_trial import simulate_multc99


def _design():
    return multc99_design(
        [1, 1],
        [1, 1],
        [Multc99Event("<event>", (1, 0), lower_cutoff=0.3, event_type="efficacy")],
        max_subjects=5,
    )


@pytest.mark.parametrize("mode", ["cohort", "period", "randomized"])
def test_saved_inputs_repeat_captured_results_and_atomic_html_escapes_names(tmp_path, mode):
    design = _design()
    if mode == "randomized":
        result = simulate_multc99_randomized(
            design,
            [[0.2, 0.8], [0.8, 0.2]],
            target_event="<event>",
            trials=12,
            reassign=True,
            seed=13,
        )
    else:
        timing = (
            dict(monitoring_period=1, accrual_rate=0.3, response_window=2.3)
            if mode == "period"
            else {}
        )
        result = simulate_multc99(design, [0.3, 0.7], trials=12, seed=13, **timing)
    report = multc99_study_report(result, title="<Study>")
    path = report.write_inputs(tmp_path / "inputs.json")
    replay = replay_multc99_study(path)
    assert replay.inputs_json == report.inputs_json
    np.testing.assert_array_equal(replay.results.elementary_counts, result.elementary_counts)
    if mode == "randomized":
        np.testing.assert_array_equal(replay.results.selected_arms, result.selected_arms)
    else:
        np.testing.assert_array_equal(replay.results.sample_sizes, result.sample_sizes)
    html_path = report.write_html(tmp_path / "report.html")
    html = html_path.read_text()
    assert "&lt;Study&gt;" in html and "&lt;event&gt;" in html
    assert "<Study>" not in html and "<event>" not in html
    assert "Complete captured inputs" in html
    assert "Events and boundaries" in html


@pytest.mark.parametrize("change", ["schema", "numpy", "fields", "nan", "duplicate", "title"])
def test_invalid_replay_inputs_raise_without_overwriting_source(tmp_path, change):
    report = multc99_study_report(simulate_multc99(_design(), [0.3, 0.7], trials=2))
    inputs = json.loads(report.inputs_json)
    if change == "schema":
        inputs["schema_version"] = True
    elif change == "numpy":
        inputs["numpy_version"] = "unavailable"
    elif change == "fields":
        inputs["simulation"]["unknown"] = 1
    elif change == "nan":
        inputs["simulation"]["elementary_probabilities"][0] = float("nan")
    elif change == "title":
        inputs["title"] = ""
    text = json.dumps(inputs)
    if change == "duplicate":
        text = text.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
    path = tmp_path / "invalid.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        replay_multc99_study(path)
    assert path.read_text() == text


def test_large_inputs_are_bounded_and_missing_parent_writes_leave_no_files(tmp_path):
    path = tmp_path / "too-large.json"
    path.write_text(" " * (4 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="4 MiB"):
        replay_multc99_study(path)
    report = multc99_study_report(simulate_multc99(_design(), [0, 1], trials=1))
    with pytest.raises(FileNotFoundError):
        report.write_html(tmp_path / "missing" / "report.html")
    assert not (tmp_path / "missing").exists()
