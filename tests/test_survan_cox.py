import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.survan_cox import survan_cox
from mdanderson_stats.survan_cox_likelihood import _CoxLikelihood


def test_manual_example_and_native_multivariable_information():
    root = Path(__file__).parent / "fixtures"
    data = np.asarray(json.loads((root / "survan-native.json").read_text())[0]["records"])
    fit = survan_cox(data[:, 0], data[:, 1], data[:, 2])
    assert_allclose(fit.negative_log_likelihood, 100.7191290008, atol=1e-9)
    assert_allclose(fit.null_negative_log_likelihood, 102.1583312986, atol=1e-9)
    assert_allclose(fit.coefficients, [-0.595902], atol=1e-5)
    assert_allclose(fit.standard_errors, [0.348404], atol=1e-6)
    native = json.loads((root / "survan-cox-native.json").read_text())
    fit = survan_cox(native["time"], native["event"], native["x"])
    assert_allclose(fit.negative_log_likelihood, native["nll"], atol=1e-10)
    assert_allclose(native["gradient"], 0, atol=1e-7)
    info = np.zeros((3, 3))
    info[np.tril_indices(3)] = native["information"]
    info += np.tril(info, -1).T
    assert_allclose(info @ fit.covariance, np.eye(3), atol=1e-9)


def test_extreme_risk_weights_and_rescaled_moments():
    x = np.array([[0.0], [1.0], [2.0]])
    model = _CoxLikelihood(np.array([1.0, 2.0, 3.0]), np.ones(3), x)
    nll, gradient, info, _ = model.evaluate(np.array([1000.0]))
    assert_allclose(nll, 3000)
    assert_allclose(gradient, [3])
    assert_allclose(info, [[0]], atol=1e-12)
    fast = model.evaluate(np.array([0.7]))
    slow = model._rescaled(model.x @ np.array([0.7]))
    for a, b in zip(fast, slow, strict=True):
        assert_allclose(a, b, atol=1e-13)


def test_separation_rank_and_covariate_units():
    with pytest.raises(ValueError, match="monotone"):
        survan_cox([1, 2, 3, 4], [1, 1, 1, 1], [4, 3, 2, 1])
    with pytest.raises(ValueError, match="rank"):
        survan_cox([1, 2, 3, 4], [1, 1, 1, 1], [1, 1, 1, 1])
    native = json.loads((Path(__file__).parent / "fixtures/survan-cox-native.json").read_text())
    x = np.asarray(native["x"])
    base = survan_cox(native["time"], native["event"], x)
    units = np.array([1e-100, 1e100, 1.0])
    scaled = survan_cox(native["time"], native["event"], x * units)
    assert_allclose(scaled.coefficients * units, base.coefficients, atol=1e-9)
    assert_allclose(scaled.log_relative_hazard(x * units), base.log_relative_hazard(x), atol=1e-10)
