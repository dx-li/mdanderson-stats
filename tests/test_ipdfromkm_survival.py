import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import ipd_survival_summary


def test_r_landmarks_greenwood_and_nelson_aalen():
    fit = ipd_survival_summary([1, 2, 2, 4, 5, 7], [1, 1, 0, 1, 0, 1])
    x = fit.at([0, 1, 2, 3, 4])
    assert_allclose(x.survival, [1, 5 / 6, 2 / 3, 2 / 3, 4 / 9], atol=1e-15)
    assert_allclose(
        x.standard_error,
        [0, 0.15214515486254615, 0.19245008972987529, 0.19245008972987529, 0.22222222222222224],
        atol=1e-15,
    )
    assert_allclose(
        x.lower,
        [1, 0.58265479547713961, 0.37860646088709510, 0.37860646088709510, 0.16680793662807852],
        atol=1e-15,
    )
    assert_allclose(x.upper, 1)
    assert_allclose(fit.estimates.cumulative_hazard[-1], 1.7, atol=1e-15)
    assert_allclose(fit.estimates.hazard_standard_error[-1], 1.08576649832682204, atol=1e-15)
    q = fit.quantiles()
    assert_array_equal(q.time, [2, 4, 7])
    assert_array_equal(q.lower, [1, 2, 4])
    assert np.all(np.isnan(q.upper))


def test_plateau_quantiles_and_unreached_levels():
    fit = ipd_survival_summary([1, 2, 3, 4], [1] * 4)
    q = fit.quantiles()
    assert_array_equal(q.time, [1.5, 2.5, 3.5])
    assert_array_equal(q.lower, [1, 1, 2])
    assert np.isnan(fit.estimates.standard_error[-1])
    for scale in (1e-200, 1e200):
        x = ipd_survival_summary(np.array([1, 2, 10, 10]) * scale, [1, 1, 0, 0])
        assert_allclose(x.quantiles(0.5).time / scale, 6)
        assert np.isnan(x.quantiles(0.25).time)
        assert np.isnan(x.quantiles(0.25).lower)
    no_events = ipd_survival_summary([1, 2, 3], [0, 0, 0])
    assert np.isnan(no_events.quantiles(0.5).time)
    assert_array_equal(no_events.estimates.lower, [1, 1, 1])
    assert_array_equal(no_events.estimates.cumulative_hazard, [0, 0, 0])
    with pytest.raises(ValueError, match="observed follow-up"):
        no_events.at(4)
    assert not fit.estimates.lower.flags.writeable
