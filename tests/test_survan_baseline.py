import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats.expsurv import exploratory_survival
from mdanderson_stats.survan_baseline import survan_baseline


def test_original_failure_time_roots_and_defining_equation():
    root = Path(__file__).parent / "fixtures"
    x = np.asarray(json.loads((root / "survan-native.json").read_text())[0]["records"])
    native = json.loads((root / "survan-baseline-native.json").read_text())
    eta = x[:, 2] * native["beta"]
    r = survan_baseline(x[:, 0], x[:, 1], eta)
    assert_allclose(np.column_stack((r.time[1:], r.alpha[1:])), native["alpha"], atol=2e-6, rtol=0)
    weights = np.exp(eta)
    for t, alpha in zip(r.time[1:], r.alpha[1:], strict=True):
        dead = weights[(x[:, 0] == t) & (x[:, 1] == 1)]
        lhs = np.sum(dead / (-np.expm1(np.log(alpha) * dead)))
        assert_allclose(lhs, weights[x[:, 0] >= t].sum(), rtol=1e-11)
    assert_allclose(r.survival, np.cumprod(r.alpha), atol=1e-14)


def test_equal_risks_reduce_to_km_under_extreme_reference_shifts():
    t = [0, 1, 1, 2, 3, 3]
    e = [0, 1, 0, 1, 1, 1]
    km = exploratory_survival(t, e)
    for shift in [-1000.0, 0.0, 1000.0]:
        result = survan_baseline(t, e, np.full(6, shift))
        assert_allclose(result.predict(shift), km.at(result.time), atol=1e-13)
        assert result.predict([shift - 1, shift, shift + 1]).shape == (3, result.time.size)
    assert_allclose(survan_baseline([1, 2], [0, 0], [0, 1]).survival, [1])


def test_analytic_tied_root_and_dominant_subject_remains_positive():
    # Tied failure risks 1,2 and total risk 10 give 10*a²+a-7=0.
    result = survan_baseline([1, 1, 2], [1, 1, 0], np.log([1, 2, 7]))
    assert_allclose(result.alpha[-1], (np.sqrt(281) - 1) / 20, atol=1e-13)
    dominant = survan_baseline([1, 2], [1, 0], [0, -20])
    assert_allclose(dominant.survival[-1], 1 / (1 + np.exp(20)), rtol=1e-13)
    rare = survan_baseline([1, 2], [1, 0], [0, 1000])
    assert_allclose(rare.predict(1000)[-1], np.exp(-1), rtol=1e-13)
