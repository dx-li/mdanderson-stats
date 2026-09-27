"""Independent R measurements and original RWT 2.4 transform references."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mdanderson_stats.pinnacle import (
    pinnacle_detect_peaks,
    pinnacle_mean_image,
    pinnacle_quantify,
)
from mdanderson_stats.pinnacle_wavelet import (
    pinnacle_daubechies_filter,
    pinnacle_denoise,
    pinnacle_irdwt,
    pinnacle_rdwt,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _rows(name):
    with (FIXTURES / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_daubechies_filters_against_independent_r():
    for row in _rows("pinnacle-filters.csv"):
        expected = np.array(row["coefficients"].split("|"), dtype=float)
        scaling, wavelet = pinnacle_daubechies_filter(int(row["length"]))
        np.testing.assert_allclose(scaling, expected, rtol=2e-10, atol=2e-12)
        expected_wavelet = expected[::-1].copy()
        expected_wavelet[1::2] *= -1
        np.testing.assert_allclose(wavelet, expected_wavelet, rtol=2e-10, atol=2e-12)


def test_transforms_and_denoising_against_original_rwt_c():
    reference = json.loads((FIXTURES / "pinnacle-wavelet.json").read_text())
    for case in reference["cases"]:
        source = np.array(case["input"])
        options = {"filter_length": case["filter_length"], "levels": case["levels"]}
        transform = pinnacle_rdwt(source, **options)
        np.testing.assert_allclose(transform.low_low, case["low"], rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(transform.details, case["detail"], rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(pinnacle_irdwt(transform), source, rtol=3e-12, atol=3e-12)
        for convention, expected in case["denoised"].items():
            result = pinnacle_denoise(
                source,
                **options,
                convention=convention,
                threshold_multiplier=expected["multiplier"],
            )
            np.testing.assert_allclose(
                [result.noise_estimate, result.threshold],
                [expected["sigma"], expected["threshold"]],
                rtol=3e-12,
                atol=3e-12,
            )
            np.testing.assert_allclose(result.image, expected["image"], rtol=3e-12, atol=3e-12)


def test_streamed_measurements_against_independent_r():
    images = np.empty((3, 6, 8))
    expected_mean = np.empty((6, 8))
    for row in _rows("pinnacle-images.csv"):
        i, r, c = (int(row[key]) for key in ("image", "row", "col"))
        images[i, r, c] = float(row["value"])
        expected_mean[r, c] = float(row["average"])
    mean = pinnacle_mean_image(iter(images))
    np.testing.assert_allclose(mean, expected_mean, rtol=2e-14, atol=2e-14)
    peaks = pinnacle_detect_peaks(mean, region=(1, 6, 1, 8))
    reference_peaks = _rows("pinnacle-peaks.csv")
    np.testing.assert_array_equal(
        peaks.coordinates,
        [[int(row[key]) for key in ("row", "col")] for row in reference_peaks],
    )
    np.testing.assert_allclose(
        peaks.intensities, [float(row["intensity"]) for row in reference_peaks], rtol=2e-14
    )
    np.testing.assert_allclose(peaks.threshold, float(reference_peaks[0]["threshold"]), rtol=2e-14)
    cases = defaultdict(list)
    for row in _rows("pinnacle-quantification.csv"):
        cases[row["background"], row["normalization"]].append(row)
    for (background, normalization), records in cases.items():
        result = pinnacle_quantify(
            iter(images),
            peaks,
            peak_radius=1,
            background=background,
            background_radius=2,
            background_quantile=0.25,
            normalization=normalization,
        )
        for field, key in (
            ("raw", "raw"),
            ("background", "baseline"),
            ("corrected", "corrected"),
            ("normalized", "normalized"),
        ):
            expected = np.array([float(row[key]) for row in records]).reshape(3, -1)
            np.testing.assert_allclose(getattr(result, field), expected, rtol=2e-13, atol=2e-13)
        expected_factors = [float(row["factor"]) for row in records if row["peak"] == "0"]
        np.testing.assert_allclose(result.normalization_factors, expected_factors, rtol=2e-13)
