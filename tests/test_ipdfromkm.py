import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import reconstruct_ipd


def test_original_r_risk_table_reconstruction():
    # CRAN IPDfromKM 0.1.10 getIPD numerical body, before dplyr reporting.
    for total_events in (None, 50):
        result = reconstruct_ipd(
            np.arange(11),
            np.linspace(1, 0.5, 11),
            risk_time=[0, 4, 8],
            at_risk=[100, 72, 40],
            total_events=total_events,
        )
        assert_array_equal(result.risk, [100, 96, 86, 77, 72, 62, 52, 43, 40, 33, 26])
        assert_array_equal(result.events, [0, 5, 4, 5, 4, 4, 3, 3, 3, 3, 2])
        assert_array_equal(result.censored, [4, 5, 5, 0, 6, 6, 6, 0, 4, 4, 0])
        assert_allclose(result.survival[-1], 0.508055626652268, atol=1e-15)
        assert result.remaining == 24
        assert result.time.size == 100
        assert result.event.sum() == 36  # The supplied 50 is not an enforced total.
        # Independently calculate product-limit estimates from emitted records.
        probability = 1.0
        for time, expected in zip(result.curve_time, result.survival, strict=True):
            deaths = result.event[result.time == time].sum()
            risk = np.count_nonzero(result.time >= time)
            probability *= 1 - deaths / risk
            assert_allclose(probability, expected, atol=1e-15)


def test_no_risk_table_and_terminal_events():
    # Original R case with fractional final censor estimate and exhausted risk.
    times = np.arange(21) / 2
    curve = np.exp(times * 2 * np.log(0.906372143992728))
    fractional = reconstruct_ipd(times, curve, risk_time=[0, 4, 8], at_risk=[100, 30, 9])
    assert_array_equal(fractional.censored[-5:], [2, 2, 3, 1, 0])
    assert fractional.risk[-1] == 0
    assert np.isfinite(fractional.survival[-1])
    x = reconstruct_ipd(np.arange(11), np.linspace(1, 0.5, 11), patients=100)
    assert_array_equal(x.events, [0] + [5] * 10)
    assert_array_equal(x.censored, np.zeros(11))
    assert x.remaining == 50
    for curve, deaths in (([1, 1, 1], 0), ([1, 0.5, 0], 10)):
        x = reconstruct_ipd([0, 1, 2], curve, patients=10)
        assert x.event.sum() == deaths
        assert len(x.time) == 10
        assert np.all(np.isfinite(x.survival))
    x = reconstruct_ipd([0, 1, 1, 2], [1, 1, 0.5, 0.5], patients=10)
    assert_array_equal(x.survival, [1, 0.5, 0.5, 0.5])
    assert x.max_absolute_error == 0.5  # Right-continuity at vertical segments.


def test_time_scaling_and_invalid_inputs():
    kwargs = dict(at_risk=[100, 72, 40])
    base = reconstruct_ipd(np.arange(11), np.linspace(1, 0.5, 11), risk_time=[0, 4, 8], **kwargs)
    for scale in (1e-200, 1e200):
        x = reconstruct_ipd(
            np.arange(11) * scale,
            np.linspace(1, 0.5, 11),
            risk_time=np.array([0, 4, 8]) * scale,
            **kwargs,
        )
        assert_allclose(x.time / scale, base.time, rtol=1e-15)
        assert_array_equal(x.event, base.event)
        assert_array_equal(x.survival, base.survival)
    with pytest.raises(ValueError, match="nonincreasing"):
        reconstruct_ipd([0, 1, 2], [1, 0.8, 0.9], patients=10)
    with pytest.raises(ValueError, match="each risk interval"):
        reconstruct_ipd([0, 2, 3], [1, 0.8, 0.5], risk_time=[0, 1, 1.5], at_risk=[10, 8, 6])
    assert not base.time.flags.writeable
