import numpy as np
import pytest

from mdanderson_stats.pinnacle import (
    pinnacle_detect_peaks,
    pinnacle_mean_image,
    pinnacle_quantify,
)
from mdanderson_stats.pinnacle_wavelet import PinnacleDenoiseSettings, pinnacle_denoise


def test_mean_image_streams_aligned_images_and_applies_crop():
    yielded: list[int] = []

    def images():
        for index in range(3):
            yielded.append(index)
            yield np.full((4, 5), index, dtype=float)

    mean = pinnacle_mean_image(images(), region=(1, 4, 2, 5))
    np.testing.assert_array_equal(mean, np.ones((3, 3)))
    assert yielded == [0, 1, 2]
    assert not mean.flags.writeable


def test_cross_maxima_suppression_ties_and_crop_coordinates_are_deterministic():
    image = np.zeros((8, 9), dtype=float)
    image[2, 2] = image[2, 3] = 10
    image[5, 6] = 9
    peaks = pinnacle_detect_peaks(
        image, threshold_quantile=0.7, suppression_radius=1, region=(1, 7, 1, 8)
    )
    np.testing.assert_array_equal(peaks.coordinates, [[2, 2], [5, 6]])
    np.testing.assert_array_equal(peaks.intensities, [10, 9])
    assert peaks.region == (1, 7, 1, 8)
    assert not peaks.coordinates.flags.writeable


def test_quantification_background_and_normalization_variants():
    first = np.full((7, 8), 2.0)
    second = np.full((7, 8), 4.0)
    first[3, 3] = 10
    first[3, 6] = 8
    second[3, 3] = 14
    second[3, 6] = 10
    coords = np.array([[3, 3], [3, 6]])

    result = pinnacle_quantify(
        [first, second],
        coords,
        peak_radius=0,
        background="local_minimum",
        background_radius=1,
        normalization="mean_pinnacle",
        region=(1, 6, 1, 7),
    )
    np.testing.assert_array_equal(result.raw, [[10, 8], [14, 10]])
    np.testing.assert_array_equal(result.background, [[2, 2], [4, 4]])
    np.testing.assert_allclose(result.corrected, [[8, 6], [10, 6]])
    np.testing.assert_allclose(result.normalized, [[8 / 7, 6 / 7], [1.25, 0.75]])

    volume = pinnacle_quantify(
        [first], coords, peak_radius=0, background="none", normalization="image_volume"
    )
    assert volume.normalization_factors[0] == np.sum(first)
    assert volume.raw.shape == (1, 2)

    signed = pinnacle_quantify(
        [first],
        coords[1:],
        peak_radius=0,
        background="global_quantile",
        background_quantile=1.0,
        normalization="none",
    )
    assert signed.corrected[0, 0] < 0
    assert signed.normalized[0, 0] == signed.corrected[0, 0]


def test_bounded_iterators_and_incompatible_inputs_fail_loudly():
    with pytest.raises(ValueError, match="at least two"):
        pinnacle_mean_image([np.ones((2, 2))])
    with pytest.raises(ValueError, match="identical dimensions"):
        pinnacle_mean_image([np.ones((2, 2)), np.ones((3, 2))])
    with pytest.raises(ValueError, match="max_images"):
        pinnacle_mean_image((np.ones((1, 1)) for _ in range(3)), max_images=2)
    with pytest.raises(ValueError, match="positive finite factor"):
        pinnacle_quantify(
            [np.ones((3, 3))], [[1, 1]], background="local_minimum", normalization="mean_pinnacle"
        )


def test_rectangular_background_window_and_scalar_compatibility():
    image = np.full((7, 7), 10.0)
    image[2, 1] = 1.0
    coords = [[3, 3]]
    rectangular = pinnacle_quantify(
        [image],
        coords,
        peak_radius=0,
        background="local_quantile",
        background_radius=(1, 2),
        background_quantile=0,
        normalization="none",
    )
    square = pinnacle_quantify(
        [image],
        coords,
        peak_radius=0,
        background="local_quantile",
        background_radius=1,
        background_quantile=0,
        normalization="none",
    )
    assert rectangular.background[0, 0] == 1
    assert square.background[0, 0] == 10
    assert rectangular.background_radius == (1, 2)
    assert square.background_radius == 1


def test_individual_gel_denoising_preserves_raw_volume_and_rejects_negative_raw():
    image = np.ones((8, 8))
    image[3, 4] = 8
    settings = PinnacleDenoiseSettings(
        filter_length=2,
        threshold_multiplier=0,
        convention="paper",
        levels=1,
    )
    result = pinnacle_quantify(
        [image],
        [[3, 4]],
        peak_radius=0,
        background="none",
        normalization="image_volume",
        denoising=settings,
    )
    np.testing.assert_allclose(result.raw, [[8]], atol=1e-13)
    assert result.normalization_factors[0] == np.sum(image)
    assert result.denoising == settings
    assert result.denoising_noise_estimates.shape == (1,)
    assert result.denoising_thresholds.shape == (1,)
    with pytest.raises(ValueError, match="nonnegative"):
        pinnacle_quantify([-np.ones((8, 8))], [[3, 4]], denoising=settings, normalization="none")


def test_cropped_individual_denoising_matches_direct_peak_and_rectangular_background():
    rows, cols = np.indices((16, 18))
    image = (
        4
        + ((13 * rows + 7 * cols) % 17) / 20
        + 6 * np.exp(-((rows - 6) ** 2 + (cols - 9) ** 2) / 3)
    )
    region = (2, 14, 2, 16)
    settings = PinnacleDenoiseSettings(
        filter_length=2,
        threshold_multiplier=1.3,
        convention="rwt",
        levels=1,
    )
    denoised = pinnacle_denoise(
        image,
        filter_length=settings.filter_length,
        threshold_multiplier=settings.threshold_multiplier,
        convention=settings.convention,
        levels=settings.levels,
        region=region,
    )
    local_row, local_col = 6 - region[0], 9 - region[2]
    peak = np.max(
        denoised.image[local_row - 1 : local_row + 2, local_col - 1 : local_col + 2]
    )
    background = np.quantile(
        denoised.image[local_row - 2 : local_row + 3, local_col - 3 : local_col + 4],
        0.25,
    )
    measured = pinnacle_quantify(
        [image],
        [[6, 9]],
        region=region,
        peak_radius=1,
        background="local_quantile",
        background_radius=(2, 3),
        background_quantile=0.25,
        normalization="none",
        denoising=settings,
    )
    np.testing.assert_allclose(measured.raw[0, 0], peak)
    np.testing.assert_allclose(measured.background[0, 0], background)
    assert measured.denoising_noise_estimates[0] > 0
    assert measured.denoising_thresholds[0] > 0


def test_pipeline_caps_per_gel_budget_at_shared_work_limit():
    from mdanderson_stats.pinnacle_pipeline import run_pinnacle

    image = np.ones((8, 8))
    image[3, 3] = 10
    settings = PinnacleDenoiseSettings(
        filter_length=2,
        threshold_multiplier=0,
        convention="paper",
        levels=1,
    )
    with pytest.raises(ValueError, match="combined max_work_bytes"):
        run_pinnacle(
            lambda: iter((image, image)),
            filter_length=2,
            levels=1,
            threshold_multiplier=0,
            max_work_bytes=12_000,
            quantification_denoising=settings,
        )
