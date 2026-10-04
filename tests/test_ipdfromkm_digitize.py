"""Focused checks for explicit KM-image digitization and previews."""

from __future__ import annotations

import numpy as np
import pytest

from mdanderson_stats.ipdfromkm import reconstruct_ipd
from mdanderson_stats.ipdfromkm_digitize import (
    KMImage,
    digitize_km_points,
    km_axis_calibration,
    load_km_image,
    plot_ipdfromkm_diagnostics,
    plot_km_digitization,
)
from mdanderson_stats.ipdfromkm_preprocess import prepare_km_coordinates


def test_axis_transform_preserves_clicks_and_is_stable_across_time_units() -> None:
    image = KMImage(np.zeros((100, 200, 4), dtype=np.uint8), "curve.png", "PNG")
    calibration = km_axis_calibration(
        image,
        x_pixel_anchors=[10, 190],
        x_values=[0, 1e-300],
        y_pixel_anchors=[80, 20],
        y_values=[0, 1],
        time_unit="years",
    )
    clicks = np.array([[10, 20], [100, 50], [190, 80]], dtype=float)
    curve = digitize_km_points(clicks, calibration)
    np.testing.assert_array_equal(curve.pixel_points, clicks)
    np.testing.assert_allclose(curve.time, [0, 0.5e-300, 1e-300], rtol=1e-14, atol=0)
    np.testing.assert_allclose(curve.survival, [1, 0.5, 0], rtol=0, atol=1e-15)
    assert not curve.time.flags.writeable

    huge = km_axis_calibration(
        image,
        x_pixel_anchors=[10, 190],
        x_values=[1e300, 1.7e308],
        y_pixel_anchors=[80, 20],
        y_values=[0, 1],
        time_unit="days",
    )
    mapped = digitize_km_points([[100, 50]], huge)
    assert np.isfinite(mapped.time[0])
    assert 1e300 < mapped.time[0] < 1.7e308


def test_image_load_preview_and_reconstruction_diagnostic(tmp_path) -> None:
    pillow = pytest.importorskip("PIL.Image")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pixels = np.full((40, 80, 3), 255, dtype=np.uint8)
    path = tmp_path / "digitized.png"
    pillow.fromarray(pixels).save(path)
    image = load_km_image(path)
    assert image.pixels.shape == (40, 80, 4)
    assert not image.pixels.flags.writeable
    calibration = km_axis_calibration(
        image,
        x_pixel_anchors=[5, 75],
        x_values=[0, 4],
        y_pixel_anchors=[35, 5],
        y_values=[0, 1],
        time_unit="months",
    )
    points = digitize_km_points([[5, 5], [20, 5], [20, 20], [50, 20], [75, 35]], calibration)
    preview = plot_km_digitization(image, calibration, points)
    assert preview.get_title().startswith("digitized.png")
    prepared = prepare_km_coordinates(points.time, points.survival, scale=1)
    fit = reconstruct_ipd(prepared.time, prepared.survival, patients=20)
    axes = plot_ipdfromkm_diagnostics(prepared, fit, time_unit="months")
    line = axes[0].lines[0]
    assert line.get_xdata().size == 1 + 2 * np.unique(fit.time).size
    assert line.get_xdata()[0] == 0  # the full KM step grid includes the origin
    assert axes[0].get_ylabel() == "Survival probability"
    plt.close("all")


def test_input_limits_and_provenance_pairing_are_explicit(tmp_path) -> None:
    image = KMImage(np.zeros((10, 10, 4), dtype=np.uint8), "a.png", "PNG")
    calibration = km_axis_calibration(
        image,
        x_pixel_anchors=[0, 9],
        x_values=[0, 9],
        y_pixel_anchors=[9, 0],
        y_values=[0, 1],
        time_unit="days",
    )
    with pytest.raises(ValueError, match="within the calibrated image"):
        digitize_km_points([[10, 2]], calibration)
    with pytest.raises(ValueError, match="1..512"):
        digitize_km_points(np.zeros((513, 2)), calibration)
    with pytest.raises(ValueError, match="dimensions do not match"):
        plot_km_digitization(
            KMImage(np.zeros((9, 9, 4), dtype=np.uint8), "a.png", "PNG"),
            calibration,
        )

    too_large = tmp_path / "large.png"
    with too_large.open("wb") as stream:
        stream.truncate(100_000_001)
    with pytest.raises(ValueError, match="100 MB"):
        load_km_image(too_large)
