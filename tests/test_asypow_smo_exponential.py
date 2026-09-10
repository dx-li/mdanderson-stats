import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import asypow_smo_exponential


def test_original_expected_likelihood_and_optimal_null():
    p = np.array([0.1, 0.2])
    length = np.array([5, 8])
    allocation = np.array([1, 2]) / 3
    design = asypow_smo_exponential(p, length, group_size=[1, 2])
    assert_allclose(design.null_parameters, 0.170169783580770589, rtol=1e-14)
    assert_allclose(design.divergence_per_observation, 0.032423046936867372, rtol=1e-12)
    assert_allclose(design.power(300), 0.83993065760027485, atol=1e-12)
    assert_allclose(design.sample_size(), 272.91884462790563, rtol=1e-12)
    # Expected score vanishes at the fitted common rate; its derivative is negative.
    events = 1 + np.expm1(-p * length) / (p * length)
    score = np.sum(allocation * events * (1 / design.null_parameters - 1 / p))
    assert_allclose(score, 0, atol=1e-14)
    fixed = asypow_smo_exponential(p, length, null_rates=0.15, group_size=[1, 2])
    assert fixed.degrees_of_freedom == 2
    assert_allclose(fixed.divergence_per_observation, 0.038608754773719056, rtol=1e-12)
    assert_allclose(fixed.power(fixed.sample_size()), 0.8, atol=1e-13)
    assert_allclose(fixed.power(300, fixed.significance(300)), 0.8, atol=1e-13)
    assert not fixed.null_parameters.flags.writeable


def test_survival_time_units_and_extreme_censoring():
    original = asypow_smo_exponential([0.1, 0.2], [5, 8], group_size=[1, 2])
    for scale in [1e-200, 1e200]:
        scaled = asypow_smo_exponential(
            np.array([0.1, 0.2]) / scale, np.array([5, 8]) * scale, group_size=[1e300, 2e300]
        )
        assert_allclose(
            scaled.divergence_per_observation, original.divergence_per_observation, rtol=1e-12
        )
        assert_allclose(scaled.null_parameters * scale, original.null_parameters, rtol=1e-12)
    # P(event) underflows, but its product with the hazard ratio is finite.
    tiny = asypow_smo_exponential([1e-300], 1e-100, null_rates=1e100)
    assert_allclose(tiny.divergence_per_observation, 1, rtol=1e-12)
    complete = asypow_smo_exponential([1e300], 1e100, null_rates=1e-300)
    assert_allclose(complete.divergence_per_observation, 2 * (600 * np.log(10) - 1), rtol=1e-12)
    q = 0.1 * (1 + 1e-10)
    relative = (q - 0.1) / 0.1
    close = asypow_smo_exponential([0.1], 5, null_rates=q)
    event_probability = 1 + np.expm1(-0.5) / 0.5
    assert_allclose(
        close.divergence_per_observation,
        event_probability * relative**2 * (1 - 2 * relative / 3),
        rtol=1e-12,
    )
    assert asypow_smo_exponential([0.1, 0.1], [2, 10]).divergence_per_observation == 0
    with pytest.raises(ValueError, match="positive"):
        asypow_smo_exponential([0.1, 0.2], [0, 5])
