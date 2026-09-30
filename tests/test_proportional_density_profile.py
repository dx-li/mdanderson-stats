import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.proportional_density_profile import proportional_density_profile


def _moderate_effect(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    event_time = scale * np.arange(1.0, 15.0)
    event_arm = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1])
    censor_time = scale * np.array([3.5, 8.5, 15.0])
    time = np.r_[event_time, censor_time, censor_time]
    event = np.r_[np.ones(event_time.size), np.zeros(2 * censor_time.size)]
    treatment = np.r_[event_arm, np.zeros(censor_time.size), np.ones(censor_time.size)]
    return time, event, treatment


def test_profile_test_and_inverted_interval_match_independent_r_reference():
    result = proportional_density_profile(
        *_moderate_effect(), beta_null=0.2, equal_censoring=True, confidence=0.95
    )
    assert result.alpha_at_null == pytest.approx(-1.45499297442886, abs=2e-11)
    assert result.beta_estimate == pytest.approx(0.195628967030482, abs=2e-12)
    assert result.likelihood_ratio == pytest.approx(0.000818156393958, abs=3e-11)
    assert result.pvalue == pytest.approx(0.977180873864839, abs=2e-10)
    assert result.confidence == 0.95
    assert result.beta_interval is not None
    assert_allclose(
        result.beta_interval,
        (-0.0805456810788242, 0.546292431918802),
        atol=2e-9,
        rtol=0,
    )
    assert result.profile_evaluations > 2
    assert result.work_bound > 0


def test_profile_inference_scales_with_time_units_and_can_skip_interval():
    base = proportional_density_profile(*_moderate_effect(), beta_null=0.2, equal_censoring=True)
    scaled = proportional_density_profile(
        *_moderate_effect(10.0), beta_null=0.02, equal_censoring=True
    )
    assert scaled.likelihood_ratio == pytest.approx(base.likelihood_ratio, abs=2e-12)
    assert scaled.pvalue == pytest.approx(base.pvalue, abs=2e-12)
    assert scaled.alpha_at_null == pytest.approx(base.alpha_at_null, abs=2e-12)
    assert scaled.beta_interval is not None and base.beta_interval is not None
    assert_allclose(np.array(scaled.beta_interval) * 10, base.beta_interval, atol=2e-10, rtol=0)

    test_only = proportional_density_profile(
        *_moderate_effect(), beta_null=0.2, equal_censoring=True, interval=False
    )
    assert test_only.beta_interval is None
    assert test_only.confidence is None
    near_one = proportional_density_profile(
        *_moderate_effect(),
        beta_null=0.2,
        equal_censoring=True,
        confidence=np.nextafter(1.0, 0.0),
    )
    assert near_one.beta_interval is not None
    assert np.all(np.isfinite(near_one.beta_interval))
    with pytest.raises(ArithmeticError, match="floating-point resolution"):
        proportional_density_profile(
            *_moderate_effect(),
            beta_null=0.2,
            equal_censoring=True,
            confidence=1e-20,
        )


def test_common_censoring_assertion_and_work_limit_are_explicit():
    inputs = _moderate_effect()
    with pytest.raises(ValueError, match="requires equal_censoring=True"):
        proportional_density_profile(*inputs, beta_null=0.2, equal_censoring=False)
    with pytest.raises(ValueError, match="max_work"):
        proportional_density_profile(*inputs, beta_null=0.2, equal_censoring=True, max_work=1)
    with pytest.raises(TypeError):
        proportional_density_profile(*inputs, beta_null=0.2)  # type: ignore[call-arg]
