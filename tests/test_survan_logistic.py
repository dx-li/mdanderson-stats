import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.survan_logistic import survan_logistic


def test_manual_example_and_exact_two_group_fit():
    x = np.r_[np.zeros(19), np.ones(21)]
    y = np.r_[np.ones(17), np.zeros(2), np.ones(19), np.zeros(2)]
    fit = survan_logistic(x, y)
    assert_allclose(fit.coefficients, [np.log(19 / 17), np.log(17 / 2)], atol=1e-7)
    assert_allclose(
        fit.standard_errors, [np.sqrt(1 / 17 + 1 / 19 + 1), np.sqrt(1 / 17 + 0.5)], atol=1e-7
    )
    assert_allclose(fit.negative_log_likelihood, 12.99775562100, atol=1e-10)
    assert_allclose(fit.pvalue, 0.9159926865, atol=1e-9)
    assert_allclose(fit.predict([0, 1]), [17 / 19, 19 / 21], atol=1e-8)


def test_native_likelihood_information_and_stationary_solution():
    data = json.loads((Path(__file__).parent / "fixtures/survan-logistic-native.json").read_text())
    for case in data["cases"]:
        fit = survan_logistic(data["x"], data["event"], intercept=case["intercept"])
        assert_allclose(fit.negative_log_likelihood, case["nll"], atol=1e-10)
        assert_allclose(case["gradient"], 0, atol=2e-6)
        p = fit.coefficients.size
        info = np.zeros((p, p))
        ix = np.tril_indices(p)
        info[ix] = case["information"]
        info = info + np.tril(info, -1).T
        assert_allclose(info @ fit.covariance, np.eye(p), atol=1e-9)
        assert_allclose(fit.predict(data["x"]), fit.fitted_probabilities, atol=1e-13)
        if not case["intercept"]:
            assert_allclose(fit.null_negative_log_likelihood, 100 * np.log(2))


def test_units_and_invalid_regressions():
    x = np.r_[np.zeros(19), np.ones(21)]
    y = np.r_[np.ones(17), np.zeros(2), np.ones(19), np.zeros(2)]
    base = survan_logistic(x, y)
    for unit in [1e-100, 1e100]:
        fit = survan_logistic(x * unit, y)
        assert_allclose(fit.coefficients * [unit, 1], base.coefficients, rtol=1e-8)
        assert_allclose(fit.predict(x * unit), base.fitted_probabilities, atol=1e-12)
    with pytest.raises(ValueError, match="separation"):
        survan_logistic([-2, -1, 1, 2], [0, 0, 1, 1])
    with pytest.raises(ValueError, match="rank"):
        survan_logistic(np.ones(5), [0, 1, 0, 1, 0])
