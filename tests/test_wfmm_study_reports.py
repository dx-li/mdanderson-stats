"""Saved contrasts and random predictions checked against direct draw algebra."""

import json
from dataclasses import replace

import numpy as np
import pytest

from mdanderson_stats import WFMMStudy, read_wfmm_study, run_wfmm_study

Y = np.array(
    [
        [1.0, 4, 2, 3],
        [2, 2, 3, 1],
        [4, 3, 1, 5],
        [5, 6, 4, 2],
        [6, 3, 7, 4],
        [3, 1, 2, 6],
        [2, 5, 5, 3],
        [4, 2, 6, 2],
    ]
)
Z = np.repeat(np.eye(4), 2, axis=0)
LT = np.array([[0.5, 0], [0.5, 0], [0, 0.25], [0, 0.75]])


def study(**kwargs):
    return WFMMStudy(
        Y, np.ones((8, 1)), transform="identity", draws=8, warmup=4, seed=193, **kwargs
    )


def test_fixed_contrasts_probabilities_and_report_are_direct_draw_summaries():
    saved = study(
        effect_contrast=[[2.0], [-1.0]],
        time_contrast=LT,
        confidence=0.8,
        include_covariance=True,
        quantiles=[0.1, 0.5, 0.9],
        effect_sizes=[0.0, 1.0],
    )
    result = run_wfmm_study(read_wfmm_study(saved.to_json()))
    raw = np.einsum("ep,cdpt->cdet", saved.effect_contrast, result.fit.coefficients) @ LT
    values = raw.reshape(-1, 2, 2)
    np.testing.assert_allclose(result.summary.mean, values.mean(0), atol=1e-14)
    np.testing.assert_allclose(result.summary.standard_deviation, values.std(0, ddof=1), atol=1e-14)
    np.testing.assert_allclose(
        result.summary.quantiles, np.quantile(values, [0.1, 0.5, 0.9], axis=0), atol=1e-14
    )
    expected = np.mean(abs(values)[None] > np.array([0.0, 1.0])[:, None, None, None], axis=1)
    np.testing.assert_array_equal(result.summary.effect_size_probability, expected)
    report = json.loads(result.report_json())
    assert report["curves"]["effect_contrast"] == [[2.0], [-1.0]]
    assert report["curves"]["time_contrast"] == LT.tolist()
    assert report["curves"]["confidence"] == 0.8
    assert "region_index" in result.curve_csv().splitlines()[0]
    assert result.covariance.variance_function_mean.shape[-1] == 4
    assert result.covariance.covariance_mean.shape == (1, 4, 4)
    assert report["variance_functions"]["coordinate_system"] == "original_time_grid"


def test_existing_random_predictions_and_saved_random_summary_match_retained_draws():
    saved = study(
        random_design=Z,
        summarize_random_effects=True,
        time_contrast=LT,
        prediction_design=np.zeros((4, 1)),
        prediction_existing_random_design=np.eye(4),
    )
    result = run_wfmm_study(read_wfmm_study(saved.to_json()))
    expected = (result.fit.random_effects @ LT).reshape(-1, 4, 2)
    np.testing.assert_allclose(result.random_effects.mean, expected.mean(0), atol=1e-14)
    np.testing.assert_allclose(result.prediction.mean, result.random_effects.mean, atol=1e-14)
    np.testing.assert_allclose(
        result.prediction.standard_deviation, expected.std(0, ddof=1), atol=1e-14
    )
    assert "random_effects" in json.loads(result.report_json())


def test_shared_new_levels_replay_and_do_not_change_fitting_stream():
    base = study(random_design=Z)
    saved = replace(
        base,
        prediction_design=np.zeros((3, 1)),
        prediction_new_random_design=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        prediction_new_random_strata=[0, 0],
    )
    original, first, second = (
        run_wfmm_study(s) for s in (base, saved, read_wfmm_study(saved.to_json()))
    )
    np.testing.assert_array_equal(original.fit.coefficients, first.fit.coefficients)
    np.testing.assert_array_equal(first.prediction.mean, second.prediction.mean)
    np.testing.assert_array_equal(first.prediction.mean[0], first.prediction.mean[1])
    np.testing.assert_array_equal(
        first.prediction.standard_deviation[0], first.prediction.standard_deviation[1]
    )
    assert not np.array_equal(first.prediction.mean[0], first.prediction.mean[2])


def test_older_complete_version_one_defaults_reopen():
    original = study()
    data = json.loads(original.to_json())
    for name in [
        "boundary",
        "summarize_random_effects",
        "include_covariance",
        "effect_contrast",
        "time_contrast",
        "confidence",
        "quantiles",
        "effect_sizes",
        "prediction_existing_random_design",
        "prediction_new_random_design",
        "prediction_new_random_strata",
    ]:
        del data[name]
    assert read_wfmm_study(json.dumps(data)).to_json() == original.to_json()


@pytest.mark.parametrize(
    "changes",
    [
        dict(effect_contrast=[[1.0, 2.0]]),
        dict(time_contrast=np.ones((3, 2))),
        dict(time_contrast=np.ones((4, 5))),
        dict(confidence=True),
        dict(include_covariance=1),
        dict(confidence=1.0),
        dict(quantiles=[0.5, 0.1]),
        dict(quantiles=[0.0, 0.5]),
        dict(effect_sizes=[-1]),
        dict(effect_sizes=[np.inf]),
        dict(summarize_random_effects=True),
        dict(summarize_random_effects=1),
        dict(prediction_existing_random_design=[[1.0]]),
        dict(prediction_design=[[1.0]], prediction_existing_random_design=[[1.0]]),
        dict(prediction_design=[[1.0]], random_design=Z, prediction_existing_random_design=[[1.0]]),
        dict(
            prediction_design=[[1.0]],
            random_design=Z,
            prediction_new_random_design=[[1.0]],
            prediction_new_random_strata=[1],
        ),
        dict(prediction_design=[[1.0]], random_design=Z, prediction_new_random_strata=[0]),
    ],
)
def test_saved_invalid_reporting_and_random_prediction_inputs_fail_early(changes):
    with pytest.raises(ValueError):
        study(**changes)
