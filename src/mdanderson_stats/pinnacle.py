"""Bounded image-peak detection and quantification primitives for Pinnacle.

The input to peak detection is an already-denoised average image. This module
does not perform registration or denoising. Coordinates are row/column indices
in the original image, including when a rectangular region is selected.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import minimum_filter

from ._cdflib import _freeze
from ._validation import FloatArray

_MAX_IMAGES = 200
_MAX_PIXELS = 4_194_304
_MAX_OUTPUT_CELLS = 2_000_000
_MAX_WORK_PIXELS = 100_000_000
_MAX_CANDIDATES = 500_000


def _scalar(value: object, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _image(value: ArrayLike, name: str, *, nonnegative: bool) -> FloatArray:
    raw = np.asarray(value)
    if (
        raw.ndim != 2
        or raw.size == 0
        or 0 in raw.shape
        or raw.size > _MAX_PIXELS
        or raw.dtype.kind not in "iuf"
    ):
        raise ValueError(f"{name} must be a bounded real 2D image")
    image = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(image)) or (nonnegative and np.any(image < 0)):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{name} pixels must be {qualifier}")
    return image


def _iter_images(images: Iterable[ArrayLike], *, max_images: int) -> Iterable[ArrayLike]:
    if isinstance(images, np.ndarray) and images.ndim == 3:
        if images.shape[0] > max_images or images.shape[1] * images.shape[2] > _MAX_PIXELS:
            raise ValueError("image stack exceeds the configured image or pixel bound")
        yield from images
        return
    for index, image in enumerate(images):
        if index >= max_images:
            raise ValueError("image iterator exceeds max_images")
        yield image


def _region(
    region: tuple[int, int, int, int] | None, shape: tuple[int, int]
) -> tuple[slice, slice, tuple[int, int, int, int]]:
    rows, cols = shape
    if region is None:
        return slice(0, rows), slice(0, cols), (0, rows, 0, cols)
    if len(region) != 4 or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in region
    ):
        raise ValueError("region must be (row_start, row_stop, col_start, col_stop) integers")
    r0, r1, c0, c1 = (int(value) for value in region)
    if not (0 <= r0 < r1 <= rows and 0 <= c0 < c1 <= cols):
        raise ValueError("region must be a nonempty in-bounds rectangle")
    return slice(r0, r1), slice(c0, c1), (r0, r1, c0, c1)


def _freeze_int(value: ArrayLike) -> NDArray[np.int64]:
    array = np.asarray(value, dtype=np.int64)
    return np.frombuffer(array.tobytes(), dtype=np.int64).reshape(array.shape)


@dataclass(frozen=True)
class PinnaclePeaks:
    """Detected peaks with immutable row/column coordinates and source metadata."""

    coordinates: NDArray[np.int64]
    intensities: FloatArray
    threshold: float
    region: tuple[int, int, int, int]
    suppression_radius: int


@dataclass(frozen=True)
class PinnacleQuantification:
    """Per-image pinnacle measurements and selected normalization factors."""

    coordinates: NDArray[np.int64]
    raw: FloatArray
    background: FloatArray
    corrected: FloatArray
    normalized: FloatArray
    normalization_factors: FloatArray
    background_method: str
    background_radius: int
    background_quantile: float
    normalization: str
    region: tuple[int, int, int, int]


def pinnacle_mean_image(
    images: Iterable[ArrayLike],
    *,
    max_images: int = _MAX_IMAGES,
    region: tuple[int, int, int, int] | None = None,
) -> FloatArray:
    """Stream the pixelwise mean of aligned, finite, nonnegative 2D images.

    A three-dimensional NumPy stack is accepted without creating another
    stack. Other iterables are consumed once and stopped at ``max_images``.
    An optional half-open ``region`` returns a rectangular crop. The result is
    limited to 4,194,304 pixels.
    """
    if isinstance(max_images, bool) or not isinstance(max_images, (int, np.integer)):
        raise ValueError("max_images must be an integer")
    if not 2 <= max_images <= _MAX_IMAGES:
        raise ValueError(f"max_images must be between 2 and {_MAX_IMAGES}")
    mean: FloatArray | None = None
    source_shape: tuple[int, int] | None = None
    count = 0
    for item in _iter_images(images, max_images=int(max_images)):
        current = _image(item, f"images[{count}]", nonnegative=True)
        if source_shape is None:
            source_shape = current.shape
            rs, cs, _ = _region(region, source_shape)
        else:
            if current.shape != source_shape:
                raise ValueError("all images must have identical dimensions")
            rs, cs, _ = _region(region, source_shape)
        current_crop = current[rs, cs]
        if mean is None:
            mean = current_crop.copy()
        else:
            # Online update avoids stacking and limits accumulated rounding.
            mean += (current_crop - mean) / (count + 1)
        count += 1
    if count < 2 or mean is None:
        raise ValueError("at least two aligned images are required")
    if not np.all(np.isfinite(mean)):
        raise ArithmeticError("mean image is not representable")
    return _freeze(mean)


def pinnacle_detect_peaks(
    image: ArrayLike,
    *,
    threshold_quantile: float = 0.75,
    suppression_radius: int = 2,
    region: tuple[int, int, int, int] | None = None,
) -> PinnaclePeaks:
    """Find cross-direction local maxima and greedily suppress nearby peaks.

    A candidate is at least as high as its existing north/south/east/west
    neighbors and strictly above the selected-image quantile threshold.
    At borders only existing neighbors are compared. Candidates are considered
    by descending intensity then ascending row/column; accepting one suppresses
    the clipped square of the requested radius. This deterministic rule may
    retain multiple members of a broad plateau when they are farther apart
    than the suppression window.
    """
    source = _image(image, "image", nonnegative=False)
    quantile = _scalar(threshold_quantile, "threshold_quantile")
    if not 0 <= quantile <= 1:
        raise ValueError("threshold_quantile must be in [0,1]")
    if isinstance(suppression_radius, bool) or not isinstance(
        suppression_radius, (int, np.integer)
    ):
        raise ValueError("suppression_radius must be a nonnegative integer")
    radius = int(suppression_radius)
    if not 0 <= radius <= 100:
        raise ValueError("suppression_radius must be in [0,100]")
    rs, cs, bounds = _region(region, source.shape)
    cropped = source[rs, cs]
    threshold = float(np.quantile(cropped, quantile))
    candidate = cropped > threshold
    if cropped.shape[0] > 1:
        candidate[1:, :] &= cropped[1:, :] >= cropped[:-1, :]
        candidate[:-1, :] &= cropped[:-1, :] >= cropped[1:, :]
    if cropped.shape[1] > 1:
        candidate[:, 1:] &= cropped[:, 1:] >= cropped[:, :-1]
        candidate[:, :-1] &= cropped[:, :-1] >= cropped[:, 1:]
    rows, cols = np.nonzero(candidate)
    if rows.size > _MAX_CANDIDATES:
        raise ValueError("candidate peak count exceeds the bounded suppression workload")
    values = cropped[rows, cols]
    order = np.lexsort((cols, rows, -values))
    occupied = np.zeros(cropped.shape, dtype=bool)
    chosen: list[tuple[int, int]] = []
    for candidate_index in order:
        row, col = int(rows[candidate_index]), int(cols[candidate_index])
        if occupied[row, col]:
            continue
        chosen.append((row + bounds[0], col + bounds[2]))
        occupied[
            max(0, row - radius) : row + radius + 1, max(0, col - radius) : col + radius + 1
        ] = True
    coordinates = np.asarray(chosen, dtype=np.int64).reshape((-1, 2))
    intensities = source[coordinates[:, 0], coordinates[:, 1]] if coordinates.size else np.empty(0)
    return PinnaclePeaks(_freeze_int(coordinates), _freeze(intensities), threshold, bounds, radius)


def _window(
    image: FloatArray, row: int, col: int, radius: int, bounds: tuple[int, int, int, int]
) -> FloatArray:
    r0, r1, c0, c1 = bounds
    return image[
        max(r0, row - radius) : min(r1, row + radius + 1),
        max(c0, col - radius) : min(c1, col + radius + 1),
    ]


def pinnacle_quantify(
    images: Iterable[ArrayLike],
    peaks: PinnaclePeaks | ArrayLike,
    *,
    peak_radius: int = 2,
    background: Literal[
        "none", "local_minimum", "local_quantile", "global_quantile"
    ] = "local_minimum",
    background_radius: int = 100,
    background_quantile: float = 0.0,
    normalization: Literal[
        "none", "mean_pinnacle", "image_volume", "pinnacle_sum"
    ] = "mean_pinnacle",
    region: tuple[int, int, int, int] | None = None,
    max_images: int = _MAX_IMAGES,
    max_output_cells: int = _MAX_OUTPUT_CELLS,
) -> PinnacleQuantification:
    """Quantify local pinnacle maxima with explicit background and normalization.

    ``mean_pinnacle`` divides each gel's corrected peaks by their mean, as in
    the 2008 article. ``pinnacle_sum`` divides by their sum. ``image_volume``
    divides by the raw pixel sum in the selected region, following the later
    manual. Background-corrected values are not clipped at zero.
    """
    if background not in {"none", "local_minimum", "local_quantile", "global_quantile"}:
        raise ValueError("unsupported background method")
    if normalization not in {"none", "mean_pinnacle", "image_volume", "pinnacle_sum"}:
        raise ValueError("unsupported normalization")
    for value, name, maximum in (
        (peak_radius, "peak_radius", 100),
        (background_radius, "background_radius", 1000),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or not 0 <= value <= maximum
        ):
            raise ValueError(f"{name} must be an integer in [0,{maximum}]")
    quantile = _scalar(background_quantile, "background_quantile")
    if not 0 <= quantile <= 1:
        raise ValueError("background_quantile must be in [0,1]")
    if (
        isinstance(max_images, bool)
        or not isinstance(max_images, (int, np.integer))
        or not 1 <= max_images <= _MAX_IMAGES
    ):
        raise ValueError(f"max_images must be in [1,{_MAX_IMAGES}]")
    if (
        isinstance(max_output_cells, bool)
        or not isinstance(max_output_cells, (int, np.integer))
        or not 0 <= max_output_cells <= _MAX_OUTPUT_CELLS
    ):
        raise ValueError(f"max_output_cells must be in [0,{_MAX_OUTPUT_CELLS}]")
    if isinstance(peaks, PinnaclePeaks):
        coordinates = np.asarray(peaks.coordinates)
        if region is None:
            region = peaks.region
    else:
        coordinates = np.asarray(peaks)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1:] != (2,)
        or coordinates.shape[0] > _MAX_OUTPUT_CELLS
    ):
        raise ValueError("peaks must be a bounded (n_peaks,2) row/column array")
    if coordinates.dtype.kind not in "iu" or coordinates.dtype.kind == "b":
        raise ValueError("peak coordinates must be integer row/column indices")
    if coordinates.size and (
        np.any(coordinates[:, 0] < 0)
        or np.any(coordinates[:, 0] >= _MAX_PIXELS)
        or np.any(coordinates[:, 1] < 0)
        or np.any(coordinates[:, 1] >= _MAX_PIXELS)
    ):
        raise ValueError("peak coordinate is outside the supported image range")
    coords = np.asarray(coordinates, dtype=np.int64)
    iterator = iter(_iter_images(images, max_images=int(max_images)))
    try:
        first_item = next(iterator)
    except StopIteration as exc:
        raise ValueError("at least one image is required") from exc
    first = _image(first_item, "images[0]", nonnegative=True)
    rs, cs, bounds = _region(region, first.shape)
    if coords.size and (
        np.any(coords[:, 0] < bounds[0])
        or np.any(coords[:, 0] >= bounds[1])
        or np.any(coords[:, 1] < bounds[2])
        or np.any(coords[:, 1] >= bounds[3])
    ):
        raise ValueError("every peak must lie within the selected region")
    raw_rows: list[FloatArray] = []
    background_rows: list[FloatArray] = []
    factor_values: list[float] = []
    work = 0
    image_count = 0
    for image_index, image_item in enumerate(chain((first_item,), iterator)):
        image = (
            first
            if image_index == 0
            else _image(image_item, f"images[{image_index}]", nonnegative=True)
        )
        if image_index and image.shape != first.shape:
            raise ValueError("all images must have identical dimensions")
        if image_index >= int(max_images):
            raise ValueError("image iterator exceeds max_images")
        if 4 * (image_index + 1) * coords.shape[0] > int(max_output_cells):
            raise ValueError("combined image-by-peak outputs exceed max_output_cells")
        work += image.shape[0] * image.shape[1]
        if work > _MAX_WORK_PIXELS:
            raise ValueError("quantification exceeds the bounded pixel-work limit")
        cropped = image[rs, cs]
        local_values = np.empty(coords.shape[0], dtype=float)
        backgrounds = np.zeros(coords.shape[0], dtype=float)
        minimum_map: FloatArray | None = None
        if background == "local_minimum" and coords.shape[0]:
            minimum_map = minimum_filter(
                cropped, size=2 * int(background_radius) + 1, mode="nearest"
            )
            work += cropped.size
            if work > _MAX_WORK_PIXELS:
                raise ValueError("background quantification exceeds the bounded pixel-work limit")
        for peak_index, (row_value, col_value) in enumerate(coords):
            row, col = int(row_value), int(col_value)
            spot = _window(image, row, col, int(peak_radius), bounds)
            work += spot.size
            if work > _MAX_WORK_PIXELS:
                raise ValueError("peak quantification exceeds the bounded pixel-work limit")
            local_values[peak_index] = float(np.max(spot))
            if background == "local_minimum":
                assert minimum_map is not None
                backgrounds[peak_index] = float(minimum_map[row - bounds[0], col - bounds[2]])
            elif background == "local_quantile":
                bg_window = _window(image, row, col, int(background_radius), bounds)
                work += bg_window.size
                if work > _MAX_WORK_PIXELS:
                    raise ValueError(
                        "background quantification exceeds the bounded pixel-work limit"
                    )
                backgrounds[peak_index] = float(np.quantile(bg_window, quantile))
        if background == "global_quantile":
            global_bg = float(np.quantile(cropped, quantile))
            backgrounds.fill(global_bg)
        elif background == "none":
            backgrounds.fill(0.0)
        raw_rows.append(local_values)
        background_rows.append(backgrounds)
        corrected = local_values - backgrounds
        if normalization == "none":
            factor = 1.0
        elif normalization == "mean_pinnacle":
            if corrected.size == 0:
                raise ValueError("mean_pinnacle normalization requires at least one peak")
            factor = float(np.mean(corrected))
        elif normalization == "pinnacle_sum":
            if corrected.size == 0:
                raise ValueError("pinnacle_sum normalization requires at least one peak")
            factor = float(np.sum(corrected))
        else:
            factor = float(np.sum(cropped))
        if not isfinite(factor) or factor <= 0:
            if normalization == "none":
                raise ArithmeticError("normalization factor is not representable")
            raise ValueError(f"{normalization} normalization requires a positive finite factor")
        factor_values.append(factor)
        image_count += 1
    if image_count == 0:
        raise ValueError("at least one image is required")
    output_raw = np.vstack(raw_rows)
    output_bg = np.vstack(background_rows)
    factors = np.asarray(factor_values, dtype=float)
    corrected_matrix = output_raw - output_bg
    normalized = corrected_matrix / factors[:, None]
    if not np.all(np.isfinite(normalized)):
        raise ArithmeticError("normalized pinnacle measurements are not representable")
    return PinnacleQuantification(
        _freeze_int(coords),
        _freeze(output_raw),
        _freeze(output_bg),
        _freeze(corrected_matrix),
        _freeze(normalized),
        _freeze(factors),
        background,
        int(background_radius),
        quantile,
        normalization,
        bounds,
    )
