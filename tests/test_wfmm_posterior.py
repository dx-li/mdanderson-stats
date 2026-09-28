import numpy as np
import pytest

from mdanderson_stats.wfmm_basis import wfmm_basis, wfmm_transform
from mdanderson_stats.wfmm_posterior import wfmm_summarize


def test_identity_effect_and_time_contrasts_match_hand_calculated_summaries():
    curves = np.asarray(
        [
            [
                [[1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0]],
                [[2.0, 3.0, 4.0, 5.0], [1.0, 1.5, 2.0, 2.5]],
                [[0.0, 1.0, 2.0, 3.0], [0.0, 0.5, 1.0, 1.5]],
            ]
        ]
    )
    effects = np.asarray([[1.0, -1.0]])
    time_regions = np.asarray([[0.5, 0.5], [0.5, 0.5], [0.0, 0.5], [0.0, 0.5]])
    basis = wfmm_basis(4, transform="identity")
    summary = wfmm_summarize(
        curves,
        basis,
        effect_contrast=effects,
        time_contrast=time_regions,
        confidence=0.8,
        quantiles=[0.25, 0.5, 0.75],
        effect_sizes=[0.5, 1.0],
        retain_curves=True,
    )
    transformed = np.einsum("le,cdet,tr->cdlr", effects, curves, time_regions)
    flat = transformed.reshape(3, 1, 2)

    np.testing.assert_allclose(summary.curves, transformed)
    np.testing.assert_allclose(summary.mean, flat.mean(axis=0))
    np.testing.assert_allclose(summary.standard_deviation, flat.std(axis=0, ddof=1))
    np.testing.assert_allclose(
        summary.quantiles, np.quantile(flat, [0.25, 0.5, 0.75], axis=0, method="linear")
    )
    expected_exceedance = np.mean(
        np.abs(flat)[None, ...] > np.asarray([0.5, 1.0])[:, None, None, None], axis=1
    )
    np.testing.assert_array_equal(summary.effect_size_probability, expected_exceedance)
    expected_sign_score = 2 * np.minimum(np.mean(flat > 0, axis=0), np.mean(flat < 0, axis=0))
    np.testing.assert_array_equal(summary.sign_tail_score, expected_sign_score)
    np.testing.assert_allclose(summary.simultaneous_critical_value, [1.0])
    np.testing.assert_allclose(summary.simultaneous_lower, [[0.25, 1.5]])
    np.testing.assert_allclose(summary.simultaneous_upper, [[1.25, 3.5]])
    np.testing.assert_allclose(summary.simbas_probability, [[0.0, 0.0]])
    assert summary.effect_size_probability.shape == (2, 1, 2)
    assert not summary.mean.flags.writeable


def test_wavelet_reconstruction_summaries_match_the_original_curve_draws():
    original = np.random.default_rng(81).normal(size=(2, 5, 2, 8))
    basis = wfmm_basis(8, transform="wavelet", filter_length=2, levels=2)
    transformed = wfmm_transform(original.reshape(-1, 8), basis).coefficients.reshape(2, 5, 2, 8)

    wavelet_summary = wfmm_summarize(transformed, basis, retain_curves=True)
    identity_summary = wfmm_summarize(original, wfmm_basis(8, transform="identity"))

    np.testing.assert_allclose(wavelet_summary.curves, original, atol=2e-15)
    np.testing.assert_allclose(wavelet_summary.mean, identity_summary.mean, atol=2e-15)
    np.testing.assert_allclose(
        wavelet_summary.standard_deviation, identity_summary.standard_deviation, atol=2e-15
    )
    np.testing.assert_allclose(
        wavelet_summary.simbas_probability, identity_summary.simbas_probability
    )


def test_simbas_constant_limits_and_preflight_rejects_contrast_expansion():
    draws = np.zeros((1, 3, 2, 4))
    draws[:, :, 1, :] = 2.0
    summary = wfmm_summarize(draws, wfmm_basis(4, transform="identity"))
    np.testing.assert_array_equal(summary.simbas_probability[0], np.ones(4))
    np.testing.assert_array_equal(summary.simbas_probability[1], np.zeros(4))
    np.testing.assert_array_equal(summary.sign_tail_score, np.zeros((2, 4)))

    large_basis = wfmm_basis(4096, transform="identity")
    coefficients = np.zeros((2, 10, 10, 4096))
    contrasts = np.ones((128, 10))
    with pytest.raises(ValueError, match="pre-time-contrast posterior curves"):
        wfmm_summarize(
            coefficients,
            large_basis,
            effect_contrast=contrasts,
            time_contrast=np.ones((4096, 1)),
        )
