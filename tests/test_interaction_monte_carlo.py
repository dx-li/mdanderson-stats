import numpy as np
from scipy.stats import t

from mdanderson_stats import (
    MedianEffectFit,
    interaction_index_monte_carlo,
    interaction_index_ray,
)


def curves(scale):
    return [
        MedianEffectFit(a, b, np.diag([scale, scale / 2]), 15, 0.1)
        for a, b in [(0, 1), (0.2, 1.3), (0.1, 1.1)]
    ]


def test_paper_rms_formula_from_retained_coefficient_draws():
    fits = curves(0.005)
    effect = np.array([0.2, 0.5, 0.8])
    result = interaction_index_monte_carlo(fits[:2], fits[2], [2, 3], effect, samples=4000, rng=65)
    samples = result.coefficient_draws
    z = np.log(effect / (1 - effect))
    inverse = np.exp((z[None, None, :] - samples[..., 0, None]) / samples[..., 1, None])
    indices = inverse[:, -1] * np.sum(np.array([0.4, 0.6])[None, :, None] / inverse[:, :2], axis=1)
    rms = np.sqrt(np.mean((indices - result.index) ** 2, axis=0))
    np.testing.assert_allclose(result.standard_error, rms, rtol=2e-14)
    width = t.isf(0.025, 39) * rms
    np.testing.assert_allclose(
        result.interval, np.stack((result.index - width, result.index + width), axis=-1)
    )
    repeat = interaction_index_monte_carlo(fits[:2], fits[2], [2, 3], effect, samples=4000, rng=65)
    np.testing.assert_array_equal(result.coefficient_draws, repeat.coefficient_draws)
    assert not result.coefficient_draws.flags.writeable


def test_small_uncertainty_agrees_with_delta_and_zero_covariance_is_exact():
    fits = curves(1e-6)
    effect = [0.25, 0.5, 0.75]
    delta = interaction_index_ray(fits[:2], fits[2], [1, 1], effect)
    result = interaction_index_monte_carlo(fits[:2], fits[2], [1, 1], effect, samples=80000, rng=18)
    np.testing.assert_allclose(
        result.standard_error, delta.index * delta.log_standard_error, rtol=0.015
    )
    fixed = curves(0)
    exact = interaction_index_monte_carlo(fixed[:2], fixed[2], [1, 1], effect, samples=100, rng=1)
    np.testing.assert_allclose(exact.standard_error, 0, atol=3e-16)


def test_reversed_slopes_are_retained_and_untransformed_limits_are_not_clipped():
    # At effect .5 with zero intercept, every inverse dose equals 1 even when slopes reverse.
    fits = [MedianEffectFit(0, 0.1, np.diag([0.0, 1.0]), 10, 0.1)] * 3
    result = interaction_index_monte_carlo(fits[:2], fits[2], [1, 1], 0.5, samples=2000, rng=7)
    assert np.all(result.slope_reversal_fraction > 0.35)
    np.testing.assert_allclose(result.index, 1)
    np.testing.assert_allclose(result.standard_error, 0)
    # Large intercept uncertainty yields a wide raw-scale interval with a negative lower bound.
    wide = [MedianEffectFit(0, 1, np.diag([1.0, 0.0]), 10, 0.1)] * 3
    raw = interaction_index_monte_carlo(wide[:2], wide[2], [1, 1], 0.5, samples=4000, rng=7)
    assert raw.interval[0] < 0
