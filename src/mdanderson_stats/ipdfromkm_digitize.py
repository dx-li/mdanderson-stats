"""Explicit bitmap calibration and plotting for IPDfromKM curves."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite
from .boin import _owned
from .ipdfromkm import ReconstructedIPD
from .ipdfromkm_preprocess import PreparedKMCurve, prepare_km_coordinates
from .ipdfromkm_survival import ipd_survival_summary

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_MAX_IMAGE_PIXELS = 10_000_000
_MAX_IMAGE_FILE_BYTES = 100_000_000
_MAX_CLICKED_POINTS = 512


@dataclass(frozen=True)
class KMImage:
    """A bounded, decoded RGB/RGBA bitmap with its source name and format."""

    pixels: NDArray[np.uint8]
    source: str
    image_format: str

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])


@dataclass(frozen=True)
class KMAxisCalibration:
    """Four manually selected anchors for linear image-to-KM coordinate maps.

    Pixel coordinates use image convention: ``(0, 0)`` is the top-left pixel
    center, x increases rightward, and y increases downward. X anchors are
    ordered left-to-right; y anchors are ordered by their actual lower/upper
    survival values, regardless of pixel order.
    """

    image_size: tuple[int, int]
    x_pixel_anchors: tuple[float, float]
    x_values: tuple[float, float]
    y_pixel_anchors: tuple[float, float]
    y_values: tuple[float, float]
    time_unit: str
    image_source: str


@dataclass(frozen=True)
class DigitizedKMCurve:
    """Raw clicked pixels and their uncleaned calibrated KM coordinates."""

    pixel_points: FloatArray
    time: FloatArray
    survival: FloatArray
    calibration: KMAxisCalibration

    def prepare(self) -> PreparedKMCurve:
        """Apply the package's explicit native-coordinate cleaning rules."""
        return prepare_km_coordinates(self.time, self.survival, scale=1)


def load_km_image(path: str | Path) -> KMImage:
    """Decode one JPEG, PNG, BMP, or TIFF after file and pixel-count preflight.

    Requires the optional ``image`` extra. Animated and multipage files are
    rejected; only the single plotted bitmap used for manual digitization is
    supported. The decoded image is bounded to 10 million pixels.
    """
    source_path = Path(path)
    try:
        size_bytes = source_path.stat().st_size
    except OSError as exc:
        raise ValueError(f"cannot inspect KM image {source_path}") from exc
    if size_bytes > _MAX_IMAGE_FILE_BYTES:
        raise ValueError("KM image file exceeds the 100 MB encoded-size limit")
    if size_bytes <= 0:
        raise ValueError("KM image file must not be empty")
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("load_km_image requires the optional 'image' extra") from exc

    try:
        with Image.open(source_path) as image:
            image_format = str(image.format or "").upper()
            if image_format not in {"JPEG", "PNG", "BMP", "TIFF"}:
                raise ValueError("KM images must be JPEG, PNG, BMP, or TIFF bitmaps")
            width, height = image.size
            if width < 1 or height < 1 or width * height > _MAX_IMAGE_PIXELS:
                raise ValueError("KM image dimensions exceed the 10-million-pixel limit")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("KM digitization accepts single-frame images only")
            image.load()
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"cannot decode KM image {source_path}") from exc
    if rgba.shape != (height, width, 4):
        raise ValueError("decoded KM image dimensions changed during loading")
    rgba.flags.writeable = False
    return KMImage(rgba, source_path.name, image_format)


def km_axis_calibration(
    image: KMImage,
    *,
    x_pixel_anchors: ArrayLike,
    x_values: ArrayLike,
    y_pixel_anchors: ArrayLike,
    y_values: ArrayLike,
    time_unit: str,
) -> KMAxisCalibration:
    """Record the four user-selected pixel/data anchors for an image.

    ``x_pixel_anchors`` and ``x_values`` are left then right. ``y_pixel_anchors``
    and ``y_values`` are lower-survival then upper-survival anchors. The latter
    normally have descending pixel rows because image y increases downward.
    """
    if not isinstance(image, KMImage):
        raise TypeError("image must be a KMImage returned by load_km_image")
    _validate_image_pixels(image)
    if not isinstance(time_unit, str) or not time_unit.strip() or len(time_unit) > 40:
        raise ValueError("time_unit must be a nonempty label of at most 40 characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in time_unit):
        raise ValueError("time_unit must not contain control characters")

    xp = _pair(x_pixel_anchors, "x_pixel_anchors")
    xv = _pair(x_values, "x_values")
    yp = _pair(y_pixel_anchors, "y_pixel_anchors")
    yv = _pair(y_values, "y_values")
    if xp[0] >= xp[1] or xv[0] < 0 or xv[0] >= xv[1] or yp[0] == yp[1] or yv[0] >= yv[1]:
        raise ValueError("axis anchors must define left-to-right x and lower-to-upper survival")
    if not 0 <= yv[0] < yv[1] <= 1:
        raise ValueError("survival anchor values must be ordered in [0,1]")
    for value in xp:
        if not 0 <= value <= image.width - 1:
            raise ValueError("x anchor pixels must lie within the image")
    for value in yp:
        if not 0 <= value <= image.height - 1:
            raise ValueError("y anchor pixels must lie within the image")
    return KMAxisCalibration(
        (image.width, image.height), xp, xv, yp, yv, time_unit.strip(), image.source
    )


def digitize_km_points(pixel_points: ArrayLike, calibration: KMAxisCalibration) -> DigitizedKMCurve:
    """Transform manually selected image pixels to time and survival values.

    The transform is the separate two-point linear calibration used by the
    cached IPDfromKM ``getpoints`` source. Click order is preserved; use
    :meth:`DigitizedKMCurve.prepare` to sort and clean as a separate operation.
    """
    if not isinstance(calibration, KMAxisCalibration):
        raise TypeError("calibration must be a KMAxisCalibration")
    raw = _point_matrix(pixel_points)
    width, height = calibration.image_size
    if np.any((raw[:, 0] < 0) | (raw[:, 0] > width - 1)):
        raise ValueError("clicked x pixels must lie within the calibrated image")
    if np.any((raw[:, 1] < 0) | (raw[:, 1] > height - 1)):
        raise ValueError("clicked y pixels must lie within the calibrated image")
    x0, x1 = calibration.x_pixel_anchors
    t0, t1 = calibration.x_values
    y0, y1 = calibration.y_pixel_anchors
    s0, s1 = calibration.y_values
    if np.any((raw[:, 0] < x0) | (raw[:, 0] > x1)):
        raise ValueError("clicked points must lie between the calibrated time-axis anchors")
    if np.any((raw[:, 1] < min(y0, y1)) | (raw[:, 1] > max(y0, y1))):
        raise ValueError("clicked points must lie between the calibrated survival-axis anchors")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x_fraction = (raw[:, 0] - x0) / (x1 - x0)
        y_fraction = (raw[:, 1] - y0) / (y1 - y0)
        time = (1 - x_fraction) * t0 + x_fraction * t1
        survival = (1 - y_fraction) * s0 + y_fraction * s1
    if np.any(~np.isfinite(time)) or np.any(~np.isfinite(survival)):
        raise ArithmeticError("calibrated KM coordinates are not representable")
    return DigitizedKMCurve(_owned(raw), _owned(time), _owned(survival), calibration)


def pick_km_curve(
    image: KMImage,
    *,
    x_values: ArrayLike,
    y_values: ArrayLike,
    time_unit: str,
    timeout: float = 300,
) -> DigitizedKMCurve:
    """Interactively select four axis anchors and up to 512 curve points.

    On the displayed image, click left then right time-axis anchors, followed
    by lower then upper survival-axis anchors. Then click curve points in time
    order; press the middle mouse button or Enter to finish early. Right-click
    (or Backspace) removes the last point during either selection stage.
    Requires the optional ``plot`` extra and an interactive Matplotlib backend.
    Each selection stage has a finite timeout in seconds (default five minutes).
    """
    if not isinstance(image, KMImage):
        raise TypeError("image must be a KMImage returned by load_km_image")
    _validate_image_pixels(image)
    x_data = _pair(x_values, "x_values")
    y_data = _pair(y_values, "y_values")
    timeout_value = float(finite(timeout, "timeout"))
    if not np.isfinite(timeout_value) or not 0 < timeout_value <= 3600:
        raise ValueError("timeout must be in (0, 3600] seconds")
    if x_data[0] >= x_data[1] or not 0 <= y_data[0] < y_data[1] <= 1:
        raise ValueError("axis values must increase and survival anchors must lie in [0,1]")
    from matplotlib import pyplot as plt
    from matplotlib.backend_bases import FigureCanvasBase

    figure, axes = plt.subplots(layout="constrained")
    if type(figure.canvas).start_event_loop is FigureCanvasBase.start_event_loop:
        plt.close(figure)
        raise RuntimeError("pick_km_curve requires an interactive Matplotlib backend")
    axes.imshow(image.pixels, origin="upper", interpolation="nearest")
    axes.set_xlim(-0.5, image.width - 0.5)
    axes.set_ylim(image.height - 0.5, -0.5)
    axes.set_aspect("equal")
    axes.set_title("Click left/right time anchors, then lower/upper survival anchors")
    anchor_pixels = figure.ginput(4, timeout=timeout_value, show_clicks=True)
    if len(anchor_pixels) != 4:
        raise ValueError("digitization ended before all four axis anchors were selected")
    selected = np.asarray(anchor_pixels, dtype=np.float64)
    calibration = km_axis_calibration(
        image,
        x_pixel_anchors=selected[:2, 0],
        x_values=x_data,
        y_pixel_anchors=selected[2:, 1],
        y_values=y_data,
        time_unit=time_unit,
    )
    axes.axvline(calibration.x_pixel_anchors[0], color="tab:blue", linewidth=0.8)
    axes.axvline(calibration.x_pixel_anchors[1], color="tab:blue", linewidth=0.8)
    axes.axhline(calibration.y_pixel_anchors[0], color="tab:red", linewidth=0.8)
    axes.axhline(calibration.y_pixel_anchors[1], color="tab:red", linewidth=0.8)
    axes.set_title("Click curve points in time order; middle-click or Enter to finish")
    clicks = figure.ginput(_MAX_CLICKED_POINTS, timeout=timeout_value, show_clicks=True)
    if not clicks:
        raise ValueError("at least one curve point must be selected")
    return digitize_km_points(clicks, calibration)


def plot_km_digitization(
    image: KMImage,
    calibration: KMAxisCalibration,
    curve: DigitizedKMCurve | None = None,
    *,
    axes: Axes | None = None,
) -> Axes:
    """Preview a bitmap, four calibration anchors, and manually clicked points."""
    if not isinstance(image, KMImage) or not isinstance(calibration, KMAxisCalibration):
        raise TypeError("image and calibration must be KMImage/KMAxisCalibration values")
    _validate_image_pixels(image)
    if (image.width, image.height) != calibration.image_size:
        raise ValueError("image dimensions do not match the calibration")
    if image.source != calibration.image_source:
        raise ValueError("image source does not match the calibration provenance")
    if curve is not None and curve.calibration != calibration:
        raise ValueError("curve and preview must use the same calibration")
    from matplotlib import pyplot as plt

    if axes is None:
        _, axes = plt.subplots(layout="constrained")
    axes.imshow(image.pixels, origin="upper", interpolation="nearest")
    for anchor in calibration.x_pixel_anchors:
        axes.axvline(anchor, color="tab:blue", linewidth=0.8, alpha=0.8)
    for anchor in calibration.y_pixel_anchors:
        axes.axhline(anchor, color="tab:red", linewidth=0.8, alpha=0.8)
    axes.plot([], [], color="tab:blue", label="time-axis anchors")
    axes.plot([], [], color="tab:red", label="survival-axis anchors")
    if curve is not None:
        axes.scatter(
            curve.pixel_points[:, 0],
            curve.pixel_points[:, 1],
            marker="o",
            s=14,
            label="clicked curve points",
        )
    axes.set_xlim(-0.5, image.width - 0.5)
    axes.set_ylim(image.height - 0.5, -0.5)
    axes.set_aspect("equal")
    axes.set_title(f"{image.source}: {calibration.time_unit} calibration")
    axes.legend(loc="best")
    return axes


def plot_ipdfromkm_diagnostics(
    prepared: PreparedKMCurve,
    reconstruction: ReconstructedIPD,
    *,
    time_unit: str = "time",
    axes: tuple[Axes, ...] | None = None,
) -> tuple[Axes, ...]:
    """Plot fitted/read-in KM points, coordinate errors, and risk counts if supplied.

    No confidence band is drawn: the native plot's ±5 RMSE ribbon is a visual
    tolerance, not a statistical confidence interval. ``ReconstructedIPD`` does
    not retain the original survival input, so the caller must pair this object
    with its corresponding prepared curve; matching times alone cannot verify
    that provenance.
    """
    if not isinstance(prepared, PreparedKMCurve) or not isinstance(
        reconstruction, ReconstructedIPD
    ):
        raise TypeError("prepared and reconstruction must be IPDfromKM result objects")
    if not isinstance(time_unit, str) or not time_unit.strip() or len(time_unit) > 40:
        raise ValueError("time_unit must be a nonempty label of at most 40 characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in time_unit):
        raise ValueError("time_unit must not contain control characters")
    if prepared.time.shape != reconstruction.curve_time.shape or not np.array_equal(
        prepared.time, reconstruction.curve_time
    ):
        raise ValueError("prepared curve and reconstruction coordinates must match")
    from matplotlib import pyplot as plt

    has_risk = reconstruction.risk_time.size > 1
    count_axes = 3 if has_risk else 2
    if axes is None:
        _, created = plt.subplots(
            count_axes,
            1,
            figsize=(8, 7),
            gridspec_kw={"height_ratios": [3, 1, 1] if has_risk else [3, 1]},
            layout="constrained",
        )
        chosen = tuple(np.atleast_1d(created))
    else:
        if len(axes) != count_axes or any(axis.figure is not axes[0].figure for axis in axes):
            raise ValueError(f"axes must contain {count_axes} axes from one figure")
        chosen = axes
    curve_ax, difference_ax = chosen[:2] if not has_risk else (chosen[0], chosen[2])
    fitted_curve = ipd_survival_summary(reconstruction.time, reconstruction.event)
    fitted_at_input = fitted_curve.at(prepared.time)
    curve_ax.step(
        fitted_curve.km.step_time,
        fitted_curve.km.step_survival,
        where="post",
        label="reconstructed IPD KM",
    )
    curve_ax.scatter(
        prepared.time, prepared.survival, s=18, marker="o", label="cleaned digitized points"
    )
    curve_ax.set(
        xlabel=f"Time ({time_unit.strip()})", ylabel="Survival probability", ylim=(0, 1.02)
    )
    curve_ax.legend()
    difference = fitted_at_input.survival - prepared.survival
    difference_ax.axhline(0, color="black", linewidth=0.8)
    difference_ax.scatter(prepared.time, difference, s=18)
    difference_ax.set(
        xlabel=f"Time ({time_unit.strip()})",
        ylabel="Fitted − digitized",
    )
    if has_risk:
        risk_ax = chosen[1]
        risk_ax.step(
            reconstruction.risk_time,
            reconstruction.reconstructed_risk,
            where="post",
            label="reconstructed risk",
        )
        risk_ax.scatter(
            reconstruction.risk_time,
            reconstruction.reported_risk,
            marker="o",
            label="reported risk",
        )
        risk_ax.set(xlabel=f"Time ({time_unit.strip()})", ylabel="At risk")
        risk_ax.legend()
    return chosen


def _pair(value: ArrayLike, name: str) -> tuple[float, float]:
    if isinstance(value, np.ndarray) and (value.ndim != 1 or value.size != 2):
        raise ValueError(f"{name} must contain exactly two real values")
    if isinstance(value, (list, tuple)) and (
        len(value) != 2 or any(not np.isscalar(item) for item in value)
    ):
        raise ValueError(f"{name} must contain exactly two real values")
    raw = finite(value, name)
    if raw.shape != (2,):
        raise ValueError(f"{name} must contain exactly two real values")
    return float(raw[0]), float(raw[1])


def _validate_image_pixels(image: KMImage) -> None:
    pixels = image.pixels
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.uint8
        or pixels.ndim != 3
        or pixels.shape[2] != 4
        or pixels.shape[0] < 1
        or pixels.shape[1] < 1
        or pixels.shape[0] * pixels.shape[1] > _MAX_IMAGE_PIXELS
    ):
        raise ValueError("KMImage pixels must be a bounded uint8 RGBA image")


def _point_matrix(value: ArrayLike) -> FloatArray:
    if isinstance(value, np.ndarray) and (
        value.ndim != 2 or value.shape[0] > _MAX_CLICKED_POINTS or value.shape[1] != 2
    ):
        raise ValueError("pixel_points must have shape (1..512, 2)")
    if isinstance(value, (list, tuple)) and (
        not 1 <= len(value) <= _MAX_CLICKED_POINTS
        or any(
            not isinstance(row, (list, tuple, np.ndarray))
            or len(row) != 2
            or any(not np.isscalar(item) for item in row)
            for row in value
        )
    ):
        raise ValueError("pixel_points must have shape (1..512, 2)")
    points = finite(value, "pixel_points")
    if points.ndim != 2 or not 1 <= points.shape[0] <= _MAX_CLICKED_POINTS or points.shape[1] != 2:
        raise ValueError("pixel_points must have shape (1..512, 2)")
    return points
