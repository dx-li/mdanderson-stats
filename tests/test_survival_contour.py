"""Focused Cox tie and contour behavior checks."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.survan_cox import survan_cox
from mdanderson_stats.survan_cox_likelihood import _CoxLikelihood
from mdanderson_stats.survival_contour import survival_cox_contour


def test_efron_score_and_information_match_finite_differences():
    time = np.array([3.0, 2.0, 2.0, 1.0])
    event = np.array([1.0, 1.0, 1.0, 1.0])
    design = np.array([[-0.7], [0.1], [0.8], [1.2]])
    beta = np.array([0.35])
    model = _CoxLikelihood(time, event, design, ties="efron")
    nll, gradient, information, _ = model.evaluate(beta)
    step = 1e-5
    plus = model.evaluate(beta + step)[0]
    minus = model.evaluate(beta - step)[0]
    assert_allclose(gradient[0], (plus - minus) / (2 * step), rtol=2e-8, atol=2e-9)
    gplus = model.evaluate(beta + step)[1][0]
    gminus = model.evaluate(beta - step)[1][0]
    assert_allclose(information[0, 0], (gplus - gminus) / (2 * step), rtol=2e-7, atol=2e-8)
    assert np.isfinite(nll)


def test_efron_handles_extreme_tied_event_and_censor_predictors():
    model = _CoxLikelihood(
        np.array([1.0, 1.0]), np.array([1.0, 0.0]), np.array([[-1000.0], [1000.0]]), ties="efron"
    )
    nll, gradient, information, _ = model.evaluate(np.array([1.0]))
    assert_allclose(nll, 2000.0)
    assert_allclose(gradient, [2000.0])
    assert_allclose(information, [[0.0]], atol=1e-12)


def test_breslow_default_is_preserved_and_contour_has_expected_shapes():
    data = json.loads((Path(__file__).parent / "fixtures/survan-cox-native.json").read_text())
    time, event, x = np.asarray(data["time"]), np.asarray(data["event"]), np.asarray(data["x"])
    old = survan_cox(time, event, x)
    explicit = survan_cox(time, event, x, ties="breslow")
    assert old.ties == "breslow"
    assert_allclose(old.coefficients, explicit.coefficients)
    result = survival_cox_contour(time, event, x, 0, n_grid=5)
    assert result.fit.ties == "efron"
    assert result.survival.shape == (5, np.unique(time).size)
    assert result.quantile_survival.shape == (5, np.unique(time).size)
    assert np.all(np.diff(result.survival, axis=1) <= 1e-14)
    assert not result.survival.flags.writeable
    with pytest.raises(ValueError, match="ties"):
        survan_cox(time, event, x, ties="bad")
