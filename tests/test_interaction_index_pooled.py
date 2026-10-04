import numpy as np
from scipy.stats import t

from mdanderson_stats.interaction_index import interaction_index
from mdanderson_stats.interaction_index_pooled import interaction_index_pooled_error
from mdanderson_stats.median_effect import MedianEffectFit


def _fits():
    return [
        MedianEffectFit(0.2, 1.1, np.array([[0.018, -0.002], [-0.002, 0.009]]), 5, 0.04),
        MedianEffectFit(-0.3, 0.8, np.array([[0.011, 0.001], [0.001, 0.016]]), 10, 0.09),
    ]


def _oracle(fits, doses, effects):
    df = np.asarray([fit.observations - 2 for fit in fits])
    residual_mse = float(np.dot(df, [fit.residual_variance for fit in fits]) / df.sum())
    logits = np.log(effects) - np.log1p(-effects)
    inverse = np.stack([(logits - fit.intercept) / fit.slope for fit in fits], axis=-1)
    terms = np.log(doses) - inverse
    maximum = np.max(terms, axis=-1, keepdims=True)
    shares = np.exp(terms - maximum) / np.sum(np.exp(terms - maximum), axis=-1, keepdims=True)
    log_index = maximum[..., 0] + np.log(np.sum(np.exp(terms - maximum), axis=-1))
    variance = np.zeros(effects.shape)
    response_gradient = np.zeros(effects.shape)
    for j, fit in enumerate(fits):
        g = shares[..., j] / fit.slope
        gradient = np.stack((g, g * inverse[..., j]), axis=-1)
        variance += np.einsum("...i,ij,...j->...", gradient, fit.covariance, gradient)
        response_gradient -= g
    variance += response_gradient**2 * residual_mse
    se = np.sqrt(variance)
    crit = t.isf(0.025, int(df.sum()))
    return log_index, se, np.stack((log_index - crit * se, log_index + crit * se), axis=-1)


def test_pooled_error_uses_residual_df_and_matches_independent_delta_oracle():
    fits = _fits()
    effects = np.array([0.25, 0.5, 0.75])
    doses = np.array([[0.3, 0.7], [0.4, 0.6], [0.8, 0.2]])
    result = interaction_index_pooled_error(fits, doses, effects)
    expected = _oracle(fits, doses, effects)
    np.testing.assert_allclose(result.log_index, expected[0], rtol=0, atol=2e-14)
    np.testing.assert_allclose(result.log_standard_error, expected[1], rtol=2e-14, atol=0)
    np.testing.assert_allclose(result.log_interval, expected[2], rtol=2e-14, atol=2e-14)
    assert result.degrees_of_freedom == 11


def test_pooled_error_is_same_as_explicit_raw_mean_variance_at_ordinary_effects():
    fits = _fits()
    effects = np.array([0.3, 0.6])
    dfs = np.array([fit.observations - 2 for fit in fits])
    pooled = float(np.dot(dfs, [fit.residual_variance for fit in fits]) / dfs.sum())
    raw_variance = effects**2 * (1 - effects) ** 2 * pooled
    doses = np.array([[0.3, 0.7], [0.6, 0.4]])
    pooled_result = interaction_index_pooled_error(fits, doses, effects)
    explicit_result = interaction_index(fits, doses, effects, effect_variance=raw_variance)
    np.testing.assert_allclose(
        pooled_result.log_standard_error, explicit_result.log_standard_error, rtol=2e-14
    )


def test_transformed_pooling_remains_finite_when_raw_effect_variance_underflows():
    tiny = 1e-200
    fits = [
        MedianEffectFit(0.0, 1.0, np.zeros((2, 2)), 3, tiny),
        MedianEffectFit(0.0, 1.0, np.zeros((2, 2)), 4, tiny),
    ]
    effect = 1e-160
    assert effect**2 * (1 - effect) ** 2 * tiny == 0.0
    result = interaction_index_pooled_error(fits, [0.5, 0.5], effect)
    assert np.isfinite(result.log_standard_error)
    assert result.log_standard_error > 0


def test_pooled_error_handles_zero_residual_component_and_rejects_direction_mismatch():
    fits = _fits()
    mixed = [fits[0], MedianEffectFit(-0.3, -0.8, fits[1].covariance, 10, 0.09)]
    with np.testing.assert_raises(ValueError):
        interaction_index_pooled_error(mixed, [0.5, 0.5], 0.5)
    zero = [fits[0], MedianEffectFit(-0.3, 0.8, np.zeros((2, 2)), 10, 0.0)]
    result = interaction_index_pooled_error(zero, [0.5, 0.5], 0.5)
    assert np.isfinite(result.log_standard_error)
