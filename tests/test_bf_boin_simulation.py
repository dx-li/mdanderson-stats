import numpy as np
import pytest

from mdanderson_stats.bf_boin import BFBOINDesign
from mdanderson_stats.bf_boin_simulation import _weibull_endpoint, simulate_bf_boin


def test_weibull_calibration_matches_both_endpoint_probabilities():
    shape, scale = _weibull_endpoint(0.25, 2.0)

    def cdf(t: float) -> float:
        return float(1 - np.exp(-((t / scale) ** shape)))

    assert cdf(2.0) == pytest.approx(0.25)
    assert cdf(1.0) == pytest.approx(0.125)


def test_calendar_simulation_reproducible_and_conserves_assigned_patients():
    kwargs = dict(
        cohorts=2,
        cohort_size=2,
        trials=3,
        n_cap=3,
    )
    first = simulate_bf_boin(
        BFBOINDesign(n_cap=3),
        [0.0, 0.0],
        rng=17,
        **{k: v for k, v in kwargs.items() if k != "n_cap"},
    )
    second = simulate_bf_boin(
        BFBOINDesign(n_cap=3),
        [0.0, 0.0],
        rng=17,
        **{k: v for k, v in kwargs.items() if k != "n_cap"},
    )
    np.testing.assert_array_equal(first.assigned, second.assigned)
    np.testing.assert_array_equal(first.patients, second.patients)
    np.testing.assert_array_equal(first.assigned, first.patients)
    assert len(first.assigned_history) == 3
    assert all(a.size > 0 for a in first.final_assessments)
    assert np.all(first.trial_duration >= first.escalation_end)


def test_fractional_current_settings_and_invalid_arrival_distribution_rejected():
    with pytest.raises(ValueError):
        simulate_bf_boin(BFBOINDesign(), [0.1, 0.2], arrival_distribution="bad")
