"""Explicit peak selection, re-quantification, and CSV export for Pinnacle."""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .pinnacle import (
    _MAX_OUTPUT_CELLS,
    PinnacleQuantification,
    _freeze_int,
    _image,
    _region,
    pinnacle_quantify,
)

if TYPE_CHECKING:
    from .pinnacle_wavelet import PinnacleDenoiseSettings

_MAX_SELECTED_PEAKS = min(_MAX_OUTPUT_CELLS, 500_000)
_MAX_EXPORT_ROWS = _MAX_OUTPUT_CELLS


def _image_shape(value: object) -> tuple[int, int]:
    dimensions: tuple[object, object]
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.shape != (2,):
            raise ValueError("image_shape must contain exactly two dimensions")
        dimensions = (value[0], value[1])
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError("image_shape must contain exactly two dimensions")
        dimensions = (value[0], value[1])
    else:
        raise ValueError("image_shape must be a bounded two-item sequence")
    parsed: list[int] = []
    for dimension in dimensions:
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, (int, np.integer)):
            raise ValueError("image dimensions must be positive integers")
        parsed.append(int(dimension))
    rows, columns = parsed
    if rows < 1 or columns < 1 or rows * columns > 4_194_304:
        raise ValueError("image_shape must be positive and within the Pinnacle pixel limit")
    return rows, columns


def _coordinate_array(value: object) -> NDArray[np.int64]:
    if isinstance(value, np.ndarray):
        if value.ndim != 2 or value.shape[1:] != (2,) or value.shape[0] > _MAX_SELECTED_PEAKS:
            raise ValueError("coordinates must be a bounded (n_peaks,2) array")
        if value.dtype.kind not in "iu":
            raise ValueError("coordinates must contain integer row and column indices")
        if value.size and np.any(value > np.iinfo(np.int64).max):
            raise ValueError("coordinate exceeds the supported integer range")
        return np.asarray(value, dtype=np.int64)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("coordinates must be a bounded sequence of integer pairs")
    if len(value) > _MAX_SELECTED_PEAKS:
        raise ValueError("coordinate count exceeds the selected-peak limit")
    rows: list[tuple[int, int]] = []
    for item in value:
        if isinstance(item, np.ndarray):
            if item.ndim != 1 or item.shape != (2,) or item.dtype.kind not in "iu":
                raise ValueError("each coordinate must be an integer (row,column) pair")
            pair = (int(item[0]), int(item[1]))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if len(item) != 2:
                raise ValueError("each coordinate must contain exactly row and column")
            if any(
                isinstance(part, (bool, np.bool_)) or not isinstance(part, (int, np.integer))
                for part in item
            ):
                raise ValueError("coordinates must contain integer row and column indices")
            pair = (int(item[0]), int(item[1]))
        else:
            raise ValueError("each coordinate must be an integer (row,column) pair")
        if any(part < np.iinfo(np.int64).min or part > np.iinfo(np.int64).max for part in pair):
            raise ValueError("coordinate exceeds the supported integer range")
        rows.append(pair)
    return np.asarray(rows, dtype=np.int64).reshape((-1, 2))


@dataclass(frozen=True)
class PinnaclePeakSelection:
    """Ordered, immutable exact pixel coordinates selected by the caller.

    Coordinates are zero-based ``(row, column)`` values in the original image.
    They are used exactly as supplied; this object does not snap to nearby local
    maxima. Duplicates are rejected. A move is expressed by removing one
    coordinate and adding another.
    """

    coordinates: NDArray[np.int64]
    image_shape: tuple[int, int]
    region: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        shape = _image_shape(self.image_shape)
        _, _, bounds = _region(self.region, shape)
        points = _coordinate_array(self.coordinates)
        if points.size and (
            np.any(points[:, 0] < bounds[0])
            or np.any(points[:, 0] >= bounds[1])
            or np.any(points[:, 1] < bounds[2])
            or np.any(points[:, 1] >= bounds[3])
        ):
            raise ValueError("every selected coordinate must lie within the image region")
        if points.shape[0] > 1 and np.unique(points, axis=0).shape[0] != points.shape[0]:
            raise ValueError("duplicate peak coordinates are not allowed")
        object.__setattr__(self, "coordinates", _freeze_int(points))
        object.__setattr__(self, "image_shape", shape)
        object.__setattr__(self, "region", bounds)

    @classmethod
    def from_coordinates(
        cls,
        coordinates: ArrayLike,
        image_shape: Sequence[int] | NDArray[Any],
        *,
        region: tuple[int, int, int, int] | None = None,
    ) -> PinnaclePeakSelection:
        shape = _image_shape(image_shape)
        _, _, bounds = _region(region, shape)
        return cls(coordinates, shape, bounds)  # type: ignore[arg-type]

    def add_peak(self, coordinate: Sequence[int]) -> PinnaclePeakSelection:
        point = _coordinate_array((coordinate,))
        if point[0, 0] < self.region[0] or point[0, 0] >= self.region[1]:
            raise ValueError("selected coordinate is outside the image region")
        if point[0, 1] < self.region[2] or point[0, 1] >= self.region[3]:
            raise ValueError("selected coordinate is outside the image region")
        if self.coordinates.shape[0] >= _MAX_SELECTED_PEAKS:
            raise ValueError("selected peak count exceeds the configured limit")
        if np.any(np.all(self.coordinates == point[0], axis=1)):
            raise ValueError("duplicate peak coordinates are not allowed")
        return PinnaclePeakSelection.from_coordinates(
            np.vstack((self.coordinates, point)), self.image_shape, region=self.region
        )

    def remove_peak(self, coordinate: Sequence[int]) -> PinnaclePeakSelection:
        point = _coordinate_array((coordinate,))[0]
        matches = np.flatnonzero(np.all(self.coordinates == point, axis=1))
        if matches.size != 1:
            raise ValueError("coordinate is not present in the selected peaks")
        remaining = np.delete(self.coordinates, int(matches[0]), axis=0)
        return PinnaclePeakSelection.from_coordinates(
            remaining, self.image_shape, region=self.region
        )


@dataclass(frozen=True)
class PinnacleSelectionAnalysis:
    """Quantification tied to the exact caller-selected peak order and settings."""

    selection: PinnaclePeakSelection
    quantification: PinnacleQuantification
    peak_radius: int


def quantify_selected_peaks(
    images: Iterable[ArrayLike],
    selection: PinnaclePeakSelection,
    *,
    peak_radius: int = 2,
    background: Literal[
        "none", "local_minimum", "local_quantile", "global_quantile"
    ] = "local_minimum",
    background_radius: int | tuple[int, int] = 100,
    background_quantile: float = 0.0,
    normalization: Literal[
        "none", "mean_pinnacle", "image_volume", "pinnacle_sum"
    ] = "mean_pinnacle",
    max_images: int = 200,
    max_output_cells: int = _MAX_OUTPUT_CELLS,
    denoising: PinnacleDenoiseSettings | None = None,
) -> PinnacleSelectionAnalysis:
    """Re-quantify exact selected coordinates using the existing bounded kernel."""
    if not isinstance(selection, PinnaclePeakSelection):
        raise ValueError("selection must be a PinnaclePeakSelection")
    if selection.coordinates.shape[0] == 0:
        raise ValueError("at least one selected peak is required for quantification")

    def checked_images() -> Iterable[ArrayLike]:
        for index, item in enumerate(images):
            image = _image(item, f"images[{index}]", nonnegative=True)
            if image.shape != selection.image_shape:
                raise ValueError("input image dimensions do not match the recorded selection")
            yield image

    measured = pinnacle_quantify(
        checked_images(),
        selection.coordinates,
        peak_radius=peak_radius,
        background=background,
        background_radius=background_radius,
        background_quantile=background_quantile,
        normalization=normalization,
        region=selection.region,
        max_images=max_images,
        max_output_cells=max_output_cells,
        denoising=denoising,
    )
    if measured.coordinates.shape != selection.coordinates.shape or not np.array_equal(
        measured.coordinates, selection.coordinates
    ):
        raise ArithmeticError("quantification changed the selected coordinate order")
    return PinnacleSelectionAnalysis(selection, measured, int(peak_radius))


def write_pinnacle_selection_csv(
    analysis: PinnacleSelectionAnalysis,
    path: str | os.PathLike[str],
) -> Path:
    """Atomically write long-form Python quantification results as UTF-8 CSV."""
    if not isinstance(analysis, PinnacleSelectionAnalysis):
        raise ValueError("analysis must be a PinnacleSelectionAnalysis")
    result = analysis.quantification
    rows, peaks = result.normalized.shape
    if rows * peaks > _MAX_EXPORT_ROWS:
        raise ValueError("peak quantification exceeds the bounded CSV export size")
    if not np.array_equal(result.coordinates, analysis.selection.coordinates):
        raise ValueError("analysis coordinates do not match the recorded selection")
    destination = Path(path)
    if not destination.parent.is_dir():
        raise ValueError("CSV parent directory must already exist")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "image_index",
                    "peak_index",
                    "row",
                    "column",
                    "raw",
                    "background",
                    "corrected",
                    "normalized",
                    "normalization_factor",
                    "peak_radius",
                    "background_method",
                    "background_radius",
                    "background_quantile",
                    "normalization",
                    "region_row_start",
                    "region_row_stop",
                    "region_column_start",
                    "region_column_stop",
                    "denoising_filter_length",
                    "denoising_threshold_multiplier",
                    "denoising_requested_levels",
                    "denoising_convention",
                    "denoising_max_work_bytes",
                    "denoising_noise_estimate",
                    "denoising_threshold",
                )
            )
            radius = result.background_radius
            radius_text = f"{radius[0]},{radius[1]}" if isinstance(radius, tuple) else str(radius)
            denoising = result.denoising
            for image_index in range(rows):
                for peak_index in range(peaks):
                    row, column = map(int, analysis.selection.coordinates[peak_index])
                    writer.writerow(
                        (
                            image_index,
                            peak_index,
                            row,
                            column,
                            repr(float(result.raw[image_index, peak_index])),
                            repr(float(result.background[image_index, peak_index])),
                            repr(float(result.corrected[image_index, peak_index])),
                            repr(float(result.normalized[image_index, peak_index])),
                            repr(float(result.normalization_factors[image_index])),
                            analysis.peak_radius,
                            result.background_method,
                            radius_text,
                            repr(float(result.background_quantile)),
                            result.normalization,
                            *result.region,
                            "" if denoising is None else denoising.filter_length,
                            "" if denoising is None else repr(denoising.threshold_multiplier),
                            ""
                            if denoising is None or denoising.levels is None
                            else denoising.levels,
                            "" if denoising is None else denoising.convention,
                            "" if denoising is None else denoising.max_work_bytes,
                            ""
                            if result.denoising_noise_estimates is None
                            else repr(float(result.denoising_noise_estimates[image_index])),
                            ""
                            if result.denoising_thresholds is None
                            else repr(float(result.denoising_thresholds[image_index])),
                        )
                    )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
