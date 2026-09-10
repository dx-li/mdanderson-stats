import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import backward_elimination, berds


def fixture():
    return json.loads((Path(__file__).parent / "fixtures/berds-native.json").read_text())


def test_original_c_complete_native_score_curve():
    data = fixture()
    result = berds(
        data["response"],
        data["predictors"],
        repetitions=6,
        validation_indices=data["validation_indices"],
        scoring="native",
    )
    assert_allclose(result.alphas, data["alphas"], rtol=1e-10)
    assert_allclose(result.scores, data["scores"], rtol=1e-12)
    assert_allclose(result.alpha_domain, data["domain"], rtol=1e-10)
    assert_allclose(result.alpha, data["alpha"], rtol=1e-10)


def test_refitted_validation_paths_and_final_model_match_r_lm():
    data = fixture()
    result = berds(
        data["response"],
        data["predictors"],
        repetitions=6,
        validation_indices=data["validation_indices"],
    )
    assert_allclose(result.split_scores, data["refit_split_scores"], rtol=1e-12)
    assert result.model.selected == (0, 1)
    assert_allclose(
        result.model.coefficients, [0.47470841412602971, -0.68952987611932370, 0, 0], rtol=1e-12
    )
    assert_allclose(result.model.intercept, 0.87687970497259049, rtol=1e-12)
    assert_allclose(result.model.mean_squared_error, 0.4574988991944432, rtol=1e-12)
    assert result.model.native_mean_squared_error < result.model.mean_squared_error
    for alpha, score in zip(result.alphas, result.scores, strict=True):
        deleted = np.sum(np.minimum.accumulate(result.split_alphas, axis=1) > alpha, axis=1)
        values = np.sort(result.split_scores[np.arange(6), deleted])
        assert_allclose(score, np.mean(values[:5]), rtol=1e-14)


def test_units_replay_and_invalid_regressions():
    data = fixture()
    y, x = np.array(data["response"]), np.array(data["predictors"])
    first = berds(y, x, repetitions=6, seed=123)
    replay = berds(y, x, repetitions=6, validation_indices=first.validation_indices)
    assert_allclose(first.scores, replay.scores, rtol=0, atol=0)
    for scale in (1e-100, 1e100):
        model = backward_elimination(y * scale, x * [1e-100, 1e100, 0.1, 10])
        ordinary = backward_elimination(y, x)
        assert model.selected == ordinary.selected
        assert_allclose(
            model.coefficients / scale * [1e-100, 1e100, 0.1, 10], ordinary.coefficients, rtol=1e-12
        )
        assert_allclose(
            model.mean_squared_error / scale / scale, ordinary.mean_squared_error, rtol=1e-12
        )
    with pytest.raises(ValueError, match="rank-deficient"):
        backward_elimination(y, np.column_stack((x[:, 0], x[:, 0])))
    with pytest.raises(ValueError, match="variance"):
        backward_elimination(x[:, 0], x[:, [0]])
