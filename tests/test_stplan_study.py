"""Saved, replayable community studies built from existing STPLAN methods."""

import json

import numpy as np
import pytest

from mdanderson_stats.stplan_study import (
    STPLANForwardCase,
    STPLANInverseCase,
    STPLANStudySpecification,
)


def _specification():
    normal_parameters = {
        "difference": 0.5,
        "sd": 1.0,
        "sample_size": 30,
        "alpha": 0.05,
    }
    cases = (
        STPLANForwardCase("normal power", "stplan_normal_one_sample_power", normal_parameters),
        STPLANInverseCase(
            "normal sample size",
            "stplan_normal_one_sample_power",
            "sample_size",
            0.8,
            (2, 1000),
            {"difference": 0.5, "sd": 1.0},
        ),
        STPLANInverseCase(
            "normal alpha",
            "stplan_normal_one_sample_power",
            "alpha",
            0.8,
            (0.0001, 0.49),
            {"difference": 0.5, "sd": 1.0, "sample_size": 20},
        ),
        STPLANInverseCase(
            "exact binomial size",
            "stplan_exact_binomial_power",
            "sample_size",
            0.8,
            (1, 100),
            {"null_probability": 0.2, "alternative_probability": 0.4},
            integer=True,
        ),
        STPLANInverseCase(
            "fractional allocation <review>",
            "stplan_binomial_k_sample_power",
            "sample_sizes",
            0.8,
            (1, 2000),
            {"probabilities": [0.1, 0.2, 0.3]},
            allocation_weights=[1, 2, 1],
        ),
    )
    return STPLANStudySpecification("saved study", cases)


def test_json_replay_and_html_capture_inputs_bounds_and_fractional_allocation(tmp_path):
    spec = _specification()
    original = spec.run()
    encoded = spec.to_json()
    replayed_spec = STPLANStudySpecification.from_json(encoded)
    replayed = replayed_spec.run()

    assert [row.achieved_power for row in original.results] == pytest.approx(
        [row.achieved_power for row in replayed.results]
    )
    assert original.results[1].value == pytest.approx(26.1375038059685, abs=2e-7)
    assert original.results[2].value == pytest.approx(0.0900291512320887, abs=2e-8)
    assert replayed.results[2].value == pytest.approx(original.results[2].value)
    assert original.results[2].inputs["alpha"] == pytest.approx(original.results[2].value)
    assert original.results[3].value == 35
    assert original.results[3].achieved_power == pytest.approx(0.804825496569362)
    assert original.results[3].evaluations == 35
    assert original.results[3].previous_value == 34
    assert original.results[3].previous_power == pytest.approx(0.766919046716108)
    ksample = original.results[4]
    assert ksample.value == pytest.approx(308.310043775048, abs=2e-7)
    assert ksample.inputs["sample_sizes"] == pytest.approx((77.07751094, 154.15502189, 77.07751094))
    assert "Explicit search bounds" in original.to_html()
    assert "fractional allocation &lt;review&gt;" in original.to_html()
    assert "native automatic-bound" in original.to_html()

    result_path = tmp_path / "study-results.json"
    html_path = tmp_path / "study.html"
    original.write_json(result_path)
    original.write_html(html_path)
    saved_results = json.loads(result_path.read_text())
    assert saved_results["specification"]
    assert "native automatic-bound defaults" in saved_results["limitations"]
    exact = saved_results["results"][3]
    assert exact["evaluations"] == 35 and exact["previous_value"] == 34
    assert html_path.read_text() == original.to_html()


def test_case_inputs_are_snapshotted_and_bad_later_schema_fails_before_any_solve(monkeypatch):
    import mdanderson_stats.stplan_study as study_module

    parameters = {"difference": 0.5, "sd": 1.0}
    first = STPLANInverseCase(
        "first",
        "stplan_normal_one_sample_power",
        "sample_size",
        0.8,
        (2, 100),
        parameters,
        max_evaluations=200,
    )
    parameters["difference"] = 999
    assert first.parameters["difference"] == 0.5
    probabilities = [0.1, 0.3]
    groups = STPLANForwardCase(
        "group vector",
        "stplan_binomial_k_sample_power",
        {"probabilities": probabilities, "sample_sizes": [40, 50]},
    )
    probabilities[0] = 0.9
    assert groups.parameters["probabilities"] == (0.1, 0.3)
    target = np.array(0.8)
    lower, upper = np.array(2.0), np.array(100.0)
    bounded = STPLANInverseCase(
        "bounded",
        "stplan_normal_one_sample_power",
        "sample_size",
        target,
        (lower, upper),
        {"difference": 0.5, "sd": 1.0},
        max_evaluations=100,
    )
    target[()] = 0.6
    lower[()] = 10.0
    assert bounded.target_power == 0.8
    assert bounded.bounds == (2.0, 100.0)

    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("schema-invalid study should fail before a solve")

    monkeypatch.setattr(study_module, "stplan_solve", forbidden)
    invalid = STPLANForwardCase("bad", "stplan_exact_binomial_power", {"null_probability": 0.2})
    spec = STPLANStudySpecification("invalid later case", (first, invalid))
    with pytest.raises(TypeError):
        spec.run()
    assert calls == 0


def test_saved_study_rejects_oversized_or_unknown_json_schema():
    with pytest.raises(ValueError, match="unknown or missing"):
        STPLANStudySpecification.from_json('{"format_version":1,"name":"x","cases":[],"extra":1}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        STPLANStudySpecification.from_json('{"name":"x","name":"y","format_version":1,"cases":[]}')
