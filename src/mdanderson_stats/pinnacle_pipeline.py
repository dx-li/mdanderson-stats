"""Replayable, bounded Pinnacle image analysis workflow."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray
from .pinnacle import (
    _MAX_IMAGES,
    _MAX_OUTPUT_CELLS,
    PinnaclePeaks,
    PinnacleQuantification,
    _freeze_int,
    _image,
    _region,
    pinnacle_detect_peaks,
    pinnacle_mean_image,
    pinnacle_quantify,
)
from .pinnacle_wavelet import (
    PinnacleDenoiseResult,
    PinnacleDenoiseSettings,
    _denoise_level_count,
    _denoise_work_bytes,
    _integer,
    pinnacle_denoise,
)


@dataclass(frozen=True)
class PinnacleAnalysis:
    """Two-pass analysis output; no input image stack is retained."""

    average_image: FloatArray
    denoising: PinnacleDenoiseResult
    peaks: PinnaclePeaks
    quantification: PinnacleQuantification
    image_count: int


class _Hasher(Protocol):
    def update(self, data: bytes | bytearray | memoryview, /) -> None: ...

    def digest(self) -> bytes: ...


def _digest_image(hasher: _Hasher, image: FloatArray) -> None:
    contiguous = np.ascontiguousarray(image, dtype="<f8")
    hasher.update(np.asarray(image.shape, dtype="<i8").tobytes())
    hasher.update(memoryview(contiguous).cast("B"))


def _image_pass(
    factory: Callable[[], Iterable[ArrayLike]],
    *,
    expected_shape: tuple[int, int] | None = None,
    max_images: int,
) -> tuple[Iterable[FloatArray], _Hasher, list[int], list[tuple[int, int]]]:
    hasher = hashlib.blake2b(digest_size=32)
    count = [0]
    shape_holder: list[tuple[int, int]] = []

    def images() -> Iterable[FloatArray]:
        source = factory()
        for index, raw in enumerate(source):
            if index >= max_images:
                raise ValueError("image factory produced more than max_images")
            image = _image(raw, f"images[{index}]", nonnegative=True)
            shape = (int(image.shape[0]), int(image.shape[1]))
            if expected_shape is not None and shape != expected_shape:
                raise ValueError("replayed image dimensions differ from the first pass")
            if shape_holder and shape != shape_holder[0]:
                raise ValueError("all factory images must have identical dimensions")
            if not shape_holder:
                shape_holder.append(shape)
            _digest_image(hasher, image)
            count[0] += 1
            yield image

    return images(), hasher, count, shape_holder


def run_pinnacle(
    images_factory: Callable[[], Iterable[ArrayLike]],
    *,
    region: tuple[int, int, int, int] | None = None,
    threshold_quantile: float = 0.75,
    suppression_radius: int = 2,
    peak_radius: int = 2,
    background: Literal[
        "none", "local_minimum", "local_quantile", "global_quantile"
    ] = "local_minimum",
    background_radius: int | tuple[int, int] = 100,
    background_quantile: float = 0.0,
    normalization: Literal[
        "none", "mean_pinnacle", "image_volume", "pinnacle_sum"
    ] = "mean_pinnacle",
    filter_length: int = 8,
    threshold_multiplier: float = 2.0,
    levels: int | None = None,
    convention: str = "paper",
    quantification_denoising: PinnacleDenoiseSettings | None = None,
    max_images: int = _MAX_IMAGES,
    max_output_cells: int = _MAX_OUTPUT_CELLS,
    max_work_bytes: int = 512 * 1024 * 1024,
) -> PinnacleAnalysis:
    """Run two-pass Pinnacle processing with a replayable image factory.

    The first pass averages RAW images and denoises that single average, as in
    the paper. The second pass quantifies original raw gels at those detected
    coordinates, optionally denoising each gel independently. The factory
    must replay the same ordered aligned images; a bounded content digest
    verifies count, dimensions and pixel identity
    between passes. The input stack is never cached.
    """
    if not callable(images_factory):
        raise ValueError("images_factory must be callable and return an image iterable")
    if isinstance(max_images, bool) or not isinstance(max_images, (int, np.integer)):
        raise ValueError("max_images must be an integer")
    if not 2 <= max_images <= _MAX_IMAGES:
        raise ValueError(f"max_images must be between 2 and {_MAX_IMAGES}")
    max_work_bytes = _integer(max_work_bytes, "max_work_bytes", 1, 1024 * 1024 * 1024)

    first_images, first_hash, first_count, first_shape = _image_pass(
        images_factory, max_images=int(max_images)
    )
    average = pinnacle_mean_image(first_images, max_images=int(max_images), region=region)
    if not first_shape or first_count[0] < 2:
        raise ValueError("at least two images are required")
    source_shape = first_shape[0]
    _, _, crop_bounds = _region(region, source_shape)
    denoise_levels = _denoise_level_count(levels, average.shape)
    held_image_bytes = average.size * 8
    average_work_bytes = _denoise_work_bytes(average.shape, denoise_levels)
    if average_work_bytes + 2 * held_image_bytes > max_work_bytes:
        raise ValueError("average-image denoising exceeds the combined max_work_bytes bound")
    denoised = pinnacle_denoise(
        average,
        filter_length=filter_length,
        threshold_multiplier=threshold_multiplier,
        levels=levels,
        convention=convention,
        max_work_bytes=max_work_bytes - 2 * held_image_bytes,
    )
    denoised = replace(
        denoised,
        origin=(crop_bounds[0], crop_bounds[2]),
        region=crop_bounds,
    )
    detected = pinnacle_detect_peaks(
        denoised.image,
        threshold_quantile=threshold_quantile,
        suppression_radius=suppression_radius,
    )
    offset = np.array([crop_bounds[0], crop_bounds[2]], dtype=np.int64)
    coordinates = np.asarray(detected.coordinates) + offset
    detected_peaks = PinnaclePeaks(
        _freeze_int(coordinates),
        detected.intensities,
        detected.threshold,
        crop_bounds,
        detected.suppression_radius,
    )

    second_images, second_hash, second_count, second_shape = _image_pass(
        images_factory, expected_shape=source_shape, max_images=int(max_images)
    )
    quantified = pinnacle_quantify(
        second_images,
        detected_peaks,
        peak_radius=peak_radius,
        background=background,
        background_radius=background_radius,
        background_quantile=background_quantile,
        normalization=normalization,
        region=crop_bounds,
        max_images=int(max_images),
        max_output_cells=max_output_cells,
        denoising=quantification_denoising,
        _reserved_work_bytes=2 * average.size * 8,
    )
    if (
        first_count[0] != second_count[0]
        or first_shape != second_shape
        or first_hash.digest() != second_hash.digest()
    ):
        raise ValueError("image factory must replay the same count, shapes and ordered pixel data")
    return PinnacleAnalysis(average, denoised, detected_peaks, quantified, first_count[0])
