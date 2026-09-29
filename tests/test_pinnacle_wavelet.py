import numpy as np
import pytest

from mdanderson_stats.pinnacle_pipeline import run_pinnacle
from mdanderson_stats.pinnacle_wavelet import (
    PinnacleDenoiseSettings,
    pinnacle_daubechies_filter,
    pinnacle_denoise,
    pinnacle_irdwt,
    pinnacle_rdwt,
)


def test_filter_and_periodic_redundant_transform_reconstruct():
    scaling, wavelet = pinnacle_daubechies_filter(2)
    np.testing.assert_allclose(scaling, [2**-0.5, 2**-0.5], atol=1e-14)
    np.testing.assert_allclose(wavelet, [2**-0.5, -(2**-0.5)], atol=1e-14)
    image = np.random.default_rng(7).normal(size=(8, 16))
    transform = pinnacle_rdwt(image, filter_length=4, levels=2)
    np.testing.assert_allclose(pinnacle_irdwt(transform), image, atol=2e-14, rtol=2e-14)
    assert not transform.details[0][2].flags.writeable


def test_denoising_conventions_and_divisibility_contract():
    image = np.zeros((8, 8))
    image[3, 4] = 1
    paper = pinnacle_denoise(image, filter_length=2, levels=1, threshold_multiplier=0)
    toolbox = pinnacle_denoise(
        image, filter_length=2, levels=1, threshold_multiplier=0, convention="rwt"
    )
    np.testing.assert_allclose(paper.image, image, atol=1e-14)
    np.testing.assert_allclose(toolbox.image, image, atol=1e-14)
    with pytest.raises(ValueError, match="divide both dimensions"):
        pinnacle_denoise(np.ones((6, 8)), filter_length=2)


def test_two_pass_pipeline_detects_on_average_and_preserves_crop_coordinates():
    first = np.ones((8, 10))
    first[3, 4] = 10
    second = np.ones((8, 10))
    second[3, 4] = 12

    def factory():
        yield first
        yield second

    result = run_pinnacle(
        factory,
        region=(1, 7, 2, 8),
        filter_length=2,
        levels=1,
        threshold_multiplier=0,
        threshold_quantile=0.8,
        suppression_radius=1,
        peak_radius=0,
        background="none",
        normalization="none",
        quantification_denoising=PinnacleDenoiseSettings(
            filter_length=2,
            threshold_multiplier=0,
            convention="paper",
            levels=1,
        ),
    )
    assert result.image_count == 2
    assert result.denoising.origin == (1, 2)
    assert result.denoising.region == (1, 7, 2, 8)
    assert [tuple(pair) for pair in result.peaks.coordinates] == [(3, 4)]
    np.testing.assert_allclose(result.quantification.raw[:, 0], [10, 12])
    assert result.quantification.denoising_noise_estimates.shape == (2,)
    assert result.quantification.coordinates[0, 0] == 3


def test_two_pass_pipeline_rejects_a_nonreplayable_factory():
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        for value in (1.0, 4.0):
            image = np.full((8, 8), value)
            image[3, 3] += 10 if calls == 1 else 5
            yield image

    with pytest.raises(ValueError, match="same count, shapes and ordered pixel data"):
        run_pinnacle(
            factory,
            filter_length=2,
            levels=1,
            threshold_multiplier=0,
            background="none",
            normalization="none",
        )
