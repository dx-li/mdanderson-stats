import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.survan_km import survan_km


def test_original_life_table_and_quantile_calculations():
    root = Path(__file__).parent / "fixtures"
    data = np.asarray(json.loads((root / "survan-native.json").read_text())[0]["records"])
    native = json.loads((root / "survan-km-native.json").read_text())
    r = survan_km(data[:, 0], data[:, 1])
    table = np.column_stack(
        (r.time, r.survival, r.standard_error**2, r.at_risk, r.events, r.censored, r.lower, r.upper)
    )
    assert_allclose(table, native["table"], rtol=2e-5, atol=1e-6)
    q = r.quantiles(np.arange(1, 10) / 10)
    ref = np.asarray(native["quantiles"])
    assert_allclose(q.time, ref[:, 1], rtol=2e-6)
    assert_allclose(q.lower, ref[:, 2])
    assert_allclose(q.upper[1:], ref[1:, 3])
    assert np.isnan(q.upper[0])
    assert_array_equal(q.flag, ref[:, 4])


def test_censor_plateaus_endpoints_and_zero_time_events():
    r = survan_km([1, 2, 3], [1, 0, 1])
    assert r.survival[1] == r.survival[2]
    assert r.lower[2] < r.lower[1]  # Risk-based intervals change at censor times.
    assert r.upper[-1] == r.lower[-1] == r.standard_error[-1] == 0
    empty = survan_km([1, 2], [0, 0])
    assert_allclose(empty.lower, 1)
    assert empty.quantiles(0.5).flag == 4
    assert np.isnan(empty.quantiles(0.5).time)
    zero = survan_km([0, 1], [1, 1])
    assert zero.quantiles(0.75).time == 0
    rare = r.quantiles([1e-300, 1 - 1e-15])
    assert np.isfinite(rare.time).all()


def test_reentry_marks_candidate_interval_unreliable():
    event = [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    result = survan_km(np.arange(1, 21), event).quantiles([0.4, 0.5])
    assert_array_equal(result.flag, [3, 3])
    assert_allclose(result.lower, [10, 7])
    assert_allclose(result.upper, [18, 17])
