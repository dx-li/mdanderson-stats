"""Saved workflow replay, prediction separation, diagnostics and schema failures."""

import csv
import io
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


def study(**kwargs):
    return WFMMStudy(Y, np.ones((8, 1)), draws=8, warmup=4, seed=193, **kwargs)


@pytest.mark.parametrize("transform", ["identity", "wavelet", "PC", "PCw", "wPC"])
def test_saved_replay_preserves_fit_prediction_and_audited_reports(transform):
    original = study(transform=transform, alpha_pc=0.9, prediction_design=[[1.0]])
    restored = read_wfmm_study(original.to_json())
    assert restored.to_json() == original.to_json()
    result = run_wfmm_study(restored)
    assert result.fit is not None
    repeated = run_wfmm_study(restored)
    np.testing.assert_array_equal(result.fit.coefficients, repeated.fit.coefficients)
    assert result.report_json() == repeated.report_json()
    # A population prediction at the fitted intercept has the same posterior
    # curve distribution, including after rectangular analysis and restoration.
    np.testing.assert_allclose(result.prediction.mean, result.summary.mean, atol=1e-14)
    np.testing.assert_allclose(
        result.prediction.standard_deviation, result.summary.standard_deviation, atol=1e-14
    )
    report = json.loads(result.report_json())
    assert report["status"] == "sampled"
    assert report["posterior_precision"] == "not_certified"
    assert report["clinical_validity"] == "not_established"
    rows = list(csv.DictReader(io.StringIO(result.curve_csv())))
    assert len(rows) == 4
    np.testing.assert_array_equal([float(r["mean"]) for r in rows], result.summary.mean[0])


def test_prediction_rng_does_not_change_posterior_and_residual_draws_replay():
    original = study(transform="identity")
    predictive = replace(
        original,
        prediction_design=[[1.0], [2.0]],
        prediction_residual=True,
        prediction_strata=[0, 0],
    )
    base, pred, again = (run_wfmm_study(s) for s in (original, predictive, predictive))
    np.testing.assert_array_equal(base.fit.coefficients, pred.fit.coefficients)
    np.testing.assert_array_equal(pred.prediction.mean, again.prediction.mean)
    assert pred.prediction.mean.shape == (2, 4)
    assert not np.array_equal(pred.prediction.mean[:1], base.summary.mean)


def test_random_effect_strata_and_filtered_studies_compose():
    random = study(transform="identity", random_design=np.repeat(np.eye(4), 2, axis=0))
    result = run_wfmm_study(read_wfmm_study(random.to_json()))
    assert result.fit is not None and result.fit.random_effects is not None
    assert result.covariance.variance_function_mean.shape == (2, 4)
    filtered = study(transform="wavelet", lowpass=1)
    result = run_wfmm_study(filtered)
    assert result.fit is not None
    np.testing.assert_array_equal(result.compression.selection.retained_indices, [0, 1])
    # With only approximation coefficients, each reconstructed adjacent pair
    # has identical marginal posterior mean and SD.
    np.testing.assert_allclose(
        result.summary.mean[:, ::2], result.summary.mean[:, 1::2], atol=1e-14
    )
    np.testing.assert_allclose(
        result.summary.standard_deviation[:, ::2],
        result.summary.standard_deviation[:, 1::2],
        atol=1e-14,
    )


def test_initializer_nonconvergence_is_saved_as_hold_without_sampling():
    result = run_wfmm_study(study(transform="identity", empirical_bayes_iterations=1))
    assert result.fit is None and result.summary is None and result.prediction is None
    report = json.loads(result.report_json())
    assert report["status"] == "held" and report["initializer_holds"]
    with pytest.raises(ValueError, match="held"):
        result.curve_csv()


def test_inputs_are_owned_and_saved_arrays_are_finite():
    y = Y.copy()
    s = WFMMStudy(y, np.ones((8, 1)))
    y[:] = 0
    np.testing.assert_array_equal(s.curves, Y)
    assert not s.curves.flags.writeable
    with pytest.raises(ValueError):
        WFMMStudy(np.full((8, 4), np.inf), np.ones((8, 1)))


def test_saved_custom_matrix_pair_reconstructs_and_invalid_inverse_fails():
    a = np.diag([1.0, 2.0, 3.0, 4.0])
    s = study(
        transform="custom",
        analysis_matrix=a,
        synthesis_matrix=np.linalg.inv(a),
        prediction_design=[[1.0]],
    )
    result = run_wfmm_study(read_wfmm_study(s.to_json()))
    assert result.fit is not None
    np.testing.assert_allclose(result.prediction.mean, result.summary.mean, atol=1e-14)
    with pytest.raises(ValueError, match="inverses"):
        run_wfmm_study(replace(s, synthesis_matrix=np.eye(4)))


def test_empty_pass_selection_and_unused_custom_matrices_fail():
    with pytest.raises(ValueError, match="no coefficients"):
        run_wfmm_study(study(highpass=1, lowpass=1))
    with pytest.raises(ValueError, match="custom"):
        study(analysis_matrix=np.eye(4), synthesis_matrix=np.eye(4))


def test_report_nulls_are_explicit_for_deterministic_coefficients():
    result = run_wfmm_study(
        WFMMStudy(np.zeros((8, 4)), np.ones((8, 1)), transform="identity", draws=8, warmup=0)
    )
    report = json.loads(result.report_json())
    if result.fit is not None:
        assert all(x is None for x in report["coefficient_rhat"][0])
    else:
        assert report["initializer_holds"]
    assert report["null_diagnostic_meaning"]


@pytest.mark.parametrize(
    "changes",
    [
        {"seed": True},
        {"seed": -1},
        {"chains": 1},
        {"draws": 7},
        {"warmup": -1},
        {"levels": 3},
        {"filter_length": 3},
        {"alpha_pc": 1.1},
        {"alpha_energy": 0},
        {"compression_ties": "native"},
        {"transform": "unknown"},
        {"min_variance": 0},
        {"random_strata": [0]},
        {"prediction_residual": True},
        {"prediction_residual": 1},
        {"prediction_design": [[1]], "prediction_residual": True},
        {"prediction_design": [[1]], "prediction_strata": [0]},
        {"prediction_design": [[1, 2]]},
        {"optimizer_iterations": 0},
    ],
)
def test_saved_controls_are_checked(changes):
    with pytest.raises(ValueError):
        replace(study(), **changes)


@pytest.mark.parametrize(
    "mode", ["unknown", "missing", "version", "bool_version", "duplicate", "nan", "nested"]
)
def test_malformed_or_unsupported_json_is_rejected(mode):
    data = json.loads(study().to_json())
    if mode == "unknown":
        data["extra"] = 1
    elif mode == "missing":
        del data["seed"]
    elif mode == "version":
        data["version"] = 2
    elif mode == "bool_version":
        data["version"] = True
    elif mode == "nan":
        data["alpha_energy"] = float("nan")
    text = json.dumps(data)
    if mode == "duplicate":
        text = text[:-1] + ', "seed": 7}'
    if mode == "nested":
        text = "[" * 2000 + "0" + "]" * 2000
    with pytest.raises(ValueError):
        read_wfmm_study(text)
