"""Redundant Daubechies wavelet transform and Pinnacle image denoising.

The transform follows Rice Wavelet Toolbox 2.4 periodization and axis order.
The implementation is an independent NumPy adaptation; see the package's
Rice Wavelet Toolbox notice for attribution and license terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite, log2

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .pinnacle import _MAX_PIXELS, _region, _scalar

_DEFAULT_WORK_BYTES = 512 * 1024 * 1024
_MAX_WORK_BYTES = 1024 * 1024 * 1024
_MIN_FILTER_LENGTH = 2
_MAX_FILTER_LENGTH = 20


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iu" or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    result = int(raw)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum},{maximum}]")
    return result


def _filter_length(value: object) -> int:
    length = _integer(value, "filter_length", _MIN_FILTER_LENGTH, _MAX_FILTER_LENGTH)
    if length % 2:
        raise ValueError("filter_length must be even")
    return length


def _levels_for_shape(shape: tuple[int, int]) -> int:
    def power_two_divisor(value: int) -> int:
        count = 0
        while value % 2 == 0:
            count += 1
            value //= 2
        return count

    return min(power_two_divisor(shape[0]), power_two_divisor(shape[1]))


def _level_count(levels: object | None, shape: tuple[int, int]) -> int:
    maximum = _levels_for_shape(shape)
    if levels is None:
        result = maximum
    else:
        result = _integer(levels, "levels", 1, 20)
    if result < 1 or result > maximum:
        raise ValueError("levels must be positive and divide both image dimensions")
    divisor = 1 << result
    if shape[0] % divisor or shape[1] % divisor:
        raise ValueError("2**levels must divide both image dimensions; padding is not applied")
    return result


def _denoise_level_count(levels: object | None, shape: tuple[int, int]) -> int:
    if levels is None:
        # RWT denoise.m uses floor(log2(min(shape))); incompatible dimensions
        # are rejected instead of padded implicitly.
        result = floor(log2(min(shape)))
    else:
        result = _integer(levels, "levels", 1, 20)
    if result < 1 or result > 20:
        raise ValueError("image dimensions do not support a denoising level")
    divisor = 1 << result
    if shape[0] % divisor or shape[1] % divisor:
        raise ValueError("2**levels must divide both dimensions; denoising does not pad")
    return result


def _check_work(shape: tuple[int, int], levels: int, work_bytes: object) -> int:
    budget = _integer(work_bytes, "max_work_bytes", 1, _MAX_WORK_BYTES)
    pixels = shape[0] * shape[1]
    if pixels > _MAX_PIXELS:
        raise ValueError(f"image exceeds the {_MAX_PIXELS}-pixel limit")
    # All level details remain live; sixteen additional image buffers bound
    # analysis/synthesis temporaries and immutable-output overhead.
    estimate = pixels * (3 * levels + 16) * np.dtype(float).itemsize
    if estimate > budget:
        raise ValueError("wavelet transform exceeds max_work_bytes")
    return budget


def _denoise_work_bytes(shape: tuple[int, int], levels: int) -> int:
    """Conservative full-image-buffer estimate used by pipeline accounting."""
    return shape[0] * shape[1] * (3 * levels + 16) * np.dtype(float).itemsize


def pinnacle_daubechies_filter(filter_length: int = 8) -> tuple[FloatArray, FloatArray]:
    """Return the minimum-phase orthonormal scaling and wavelet taps (sum sqrt(2)).

    Even filter lengths from 2 through 20 are supported. Polynomial roots are
    selected by increasing magnitude, matching the minimum-phase factor.
    """
    length = _filter_length(filter_length)
    half = length // 2
    coefficient = 1.0
    p = np.array([1.0])
    q = np.array([1.0])
    scaling = np.array([1.0, 1.0])
    for index in range(1, half):
        coefficient = -coefficient * 0.25 * (index + half - 1) / index
        scaling = np.pad(scaling, (1, 0)) + np.pad(scaling, (0, 1))
        p = np.pad(-p, (1, 0)) + np.pad(p, (0, 1))
        p = np.pad(-p, (1, 0)) + np.pad(p, (0, 1))
        q = np.pad(q, (1, 1)) + coefficient * p
    roots = np.roots(q)
    ordered = roots[np.argsort(np.abs(roots), kind="stable")]
    inside = ordered[: half - 1]
    if np.any(np.abs(inside) >= 1 + 1e-7):
        raise ArithmeticError("minimum-phase Daubechies roots are not representable")
    scaling = np.convolve(scaling, np.real(np.poly(inside)))
    scaling *= np.sqrt(2.0) / float(np.sum(scaling))
    if scaling.size != length or abs(float(np.dot(scaling, scaling)) - 1.0) > 1e-7:
        raise ArithmeticError("Daubechies filter construction is numerically unstable")
    wavelet = scaling[::-1].copy()
    wavelet[1::2] *= -1
    return _freeze(scaling), _freeze(wavelet)


def _filter_axis(image: FloatArray, taps: FloatArray, axis: int, stride: int) -> FloatArray:
    result = np.zeros_like(image)
    with np.errstate(over="ignore", invalid="ignore"):
        for index, tap in enumerate(taps):
            result += float(tap) * np.roll(image, -index * stride, axis=axis)
    return result


def _synthesis_axis(
    low: FloatArray,
    high: FloatArray,
    scaling: FloatArray,
    wavelet: FloatArray,
    axis: int,
    stride: int,
) -> FloatArray:
    result = np.zeros_like(low)
    with np.errstate(over="ignore", invalid="ignore"):
        for index, (lo, hi) in enumerate(zip(scaling, wavelet, strict=True)):
            shift = index * stride
            result += 0.5 * (
                float(lo) * np.roll(low, shift, axis=axis)
                + float(hi) * np.roll(high, shift, axis=axis)
            )
    return result


def _decompose(
    image: FloatArray,
    scaling: FloatArray,
    wavelet: FloatArray,
    levels: int,
    *,
    freeze_details: bool = False,
) -> tuple[FloatArray, tuple[tuple[FloatArray, FloatArray, FloatArray], ...]]:
    low_low = image.copy()
    details: list[tuple[FloatArray, FloatArray, FloatArray]] = []
    for level in range(levels):
        stride = 1 << level
        row_low = _filter_axis(low_low, scaling, axis=1, stride=stride)
        row_high = _filter_axis(low_low, wavelet, axis=1, stride=stride)
        ll = _filter_axis(row_low, scaling, axis=0, stride=stride)
        lh = _filter_axis(row_low, wavelet, axis=0, stride=stride)
        hl = _filter_axis(row_high, scaling, axis=0, stride=stride)
        hh = _filter_axis(row_high, wavelet, axis=0, stride=stride)
        if not all(np.all(np.isfinite(band)) for band in (ll, lh, hl, hh)):
            raise ArithmeticError("wavelet coefficients exceed floating-point range")
        triplet = (lh, hl, hh)
        if freeze_details:
            frozen_triplet: tuple[FloatArray, FloatArray, FloatArray] = (
                _freeze(lh),
                _freeze(hl),
                _freeze(hh),
            )
            details.append(frozen_triplet)
        else:
            details.append(triplet)
        low_low = ll
    return low_low, tuple(details)


def _synthesize(
    low_low: FloatArray,
    details: tuple[tuple[FloatArray, FloatArray, FloatArray], ...],
    scaling: FloatArray,
    wavelet: FloatArray,
) -> FloatArray:
    current = low_low.copy()
    for level in range(len(details) - 1, -1, -1):
        lh, hl, hh = details[level]
        stride = 1 << level
        row_low = _synthesis_axis(current, lh, scaling, wavelet, axis=0, stride=stride)
        row_high = _synthesis_axis(hl, hh, scaling, wavelet, axis=0, stride=stride)
        current = _synthesis_axis(row_low, row_high, scaling, wavelet, axis=1, stride=stride)
    if not np.all(np.isfinite(current)):
        raise ArithmeticError("inverse wavelet transform is not finite")
    return current


@dataclass(frozen=True)
class PinnacleWaveletTransform:
    """Immutable redundant-wavelet coefficients; details are ordered finest first."""

    low_low: FloatArray
    details: tuple[tuple[FloatArray, FloatArray, FloatArray], ...]
    scaling_filter: FloatArray
    wavelet_filter: FloatArray
    shape: tuple[int, int]
    levels: int
    max_work_bytes: int


def pinnacle_rdwt(
    image: ArrayLike,
    *,
    filter_length: int = 8,
    levels: int | None = None,
    max_work_bytes: int = _DEFAULT_WORK_BYTES,
) -> PinnacleWaveletTransform:
    """Compute a 2D periodized redundant Daubechies transform.

    Analysis filters rows first, then columns; detail bands are ``(LH, HL, HH)``
    at each level. Dimensions must be divisible by ``2**levels``; there is no
    implicit padding. Default levels use the largest common power-of-two divisor.
    """
    raw = np.asarray(image)
    if raw.ndim != 2 or raw.size == 0 or raw.dtype.kind not in "iuf":
        raise ValueError("image must be a nonempty real 2D array")
    if raw.size > _MAX_PIXELS:
        raise ValueError("image exceeds the bounded pixel count")
    shape = (int(raw.shape[0]), int(raw.shape[1]))
    level_count = _level_count(levels, shape)
    budget = _check_work(shape, level_count, max_work_bytes)
    length = _filter_length(filter_length)
    source = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(source)):
        raise ValueError("image values must be finite")
    scaling, wavelet = pinnacle_daubechies_filter(length)
    low_low, details = _decompose(source, scaling, wavelet, level_count, freeze_details=True)
    return PinnacleWaveletTransform(
        _freeze(low_low),
        details,
        scaling,
        wavelet,
        shape,
        level_count,
        budget,
    )


def pinnacle_irdwt(transform: PinnacleWaveletTransform) -> FloatArray:
    """Reconstruct an image, validating coefficient shapes and finiteness first."""
    if not isinstance(transform, PinnacleWaveletTransform):
        raise ValueError("transform must be a PinnacleWaveletTransform")
    shape = transform.shape
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape)
        or shape[0] * shape[1] > _MAX_PIXELS
    ):
        raise ValueError("transform shape is invalid or exceeds the pixel bound")
    levels = _level_count(transform.levels, shape)
    _check_work(shape, levels, transform.max_work_bytes)
    scaling_raw = np.asarray(transform.scaling_filter)
    wavelet_raw = np.asarray(transform.wavelet_filter)
    if (
        scaling_raw.ndim != 1
        or scaling_raw.size % 2
        or not 2 <= scaling_raw.size <= _MAX_FILTER_LENGTH
        or wavelet_raw.shape != scaling_raw.shape
        or scaling_raw.dtype.kind not in "iuf"
        or wavelet_raw.dtype.kind not in "iuf"
        or not np.all(np.isfinite(scaling_raw))
        or not np.all(np.isfinite(wavelet_raw))
    ):
        raise ValueError("transform filters are malformed")
    scaling = np.asarray(scaling_raw, dtype=float)
    wavelet = np.asarray(wavelet_raw, dtype=float)
    if not isinstance(transform.details, tuple):
        raise ValueError("transform details must be a tuple of coefficient triples")
    low_raw = np.asarray(transform.low_low)
    if (
        low_raw.shape != shape
        or low_raw.dtype.kind not in "iuf"
        or not np.all(np.isfinite(low_raw))
        or len(transform.details) != levels
    ):
        raise ValueError("transform low-pass or level data are malformed")
    low = np.asarray(low_raw, dtype=float)
    detail_arrays: list[tuple[FloatArray, FloatArray, FloatArray]] = []
    for triplet in transform.details:
        if len(triplet) != 3:
            raise ValueError("each level must contain LH, HL and HH bands")
        bands_raw = tuple(np.asarray(band) for band in triplet)
        if any(
            band.shape != shape or band.dtype.kind not in "iuf" or not np.all(np.isfinite(band))
            for band in bands_raw
        ):
            raise ValueError("wavelet detail bands must match the image and be finite")
        bands = tuple(np.asarray(band, dtype=float) for band in bands_raw)
        detail_arrays.append(bands)  # type: ignore[arg-type]
    reconstructed = _synthesize(low, tuple(detail_arrays), scaling, wavelet)
    return _freeze(reconstructed)


@dataclass(frozen=True)
class PinnacleDenoiseSettings:
    """Explicit wavelet settings reusable for individual-gel quantification."""

    filter_length: int
    threshold_multiplier: float
    convention: str
    levels: int | None = None
    max_work_bytes: int = _DEFAULT_WORK_BYTES

    def __post_init__(self) -> None:
        _filter_length(self.filter_length)
        multiplier = _scalar(self.threshold_multiplier, "threshold_multiplier")
        if multiplier < 0:
            raise ValueError("threshold_multiplier must be finite and nonnegative")
        if self.convention not in {"paper", "rwt"}:
            raise ValueError("convention must be 'paper' or 'rwt'")
        if self.levels is not None:
            _integer(self.levels, "levels", 1, 20)
        _integer(self.max_work_bytes, "max_work_bytes", 1, _MAX_WORK_BYTES)


@dataclass(frozen=True)
class PinnacleDenoiseResult:
    """Denoised image and the explicit wavelet/noise-threshold conventions used."""

    image: FloatArray
    origin: tuple[int, int]
    region: tuple[int, int, int, int]
    scaling_filter: FloatArray
    levels: int
    filter_length: int
    noise_estimate: float
    threshold: float
    threshold_multiplier: float
    convention: str


def pinnacle_denoise(
    image: ArrayLike,
    *,
    filter_length: int = 8,
    threshold_multiplier: float = 2.0,
    levels: int | None = None,
    convention: str = "paper",
    region: tuple[int, int, int, int] | None = None,
    max_work_bytes: int = _DEFAULT_WORK_BYTES,
) -> PinnacleDenoiseResult:
    """Denoise a 2D image with undecimated hard-thresholded details.

    ``paper`` interprets the article's MAD as median absolute deviation about
    the finest HH median, divides by 0.6745, and retains coefficients equal to
    threshold. ``rwt`` follows RWT 2.4 literally: median(abs(HH))/.67 and strict
    ``abs(coefficient) > threshold``. The two are exposed separately because
    the published prose and toolbox differ.
    """
    raw = np.asarray(image)
    if raw.ndim != 2 or raw.size == 0 or raw.size > _MAX_PIXELS or raw.dtype.kind not in "iuf":
        raise ValueError("image must be a bounded nonempty real 2D array")
    rs, cs, bounds = _region(region, (int(raw.shape[0]), int(raw.shape[1])))
    crop_raw = raw[rs, cs]
    shape = (int(crop_raw.shape[0]), int(crop_raw.shape[1]))
    level_count = _denoise_level_count(levels, shape)
    _check_work(shape, level_count, max_work_bytes)
    length = _filter_length(filter_length)
    multiplier = _scalar(threshold_multiplier, "threshold_multiplier")
    if not isfinite(multiplier) or multiplier < 0:
        raise ValueError("threshold_multiplier must be finite and nonnegative")
    if convention not in {"paper", "rwt"}:
        raise ValueError("convention must be 'paper' or 'rwt'")
    source = np.asarray(crop_raw, dtype=float)
    if not np.all(np.isfinite(source)):
        raise ValueError("image values must be finite")
    scaling, wavelet = pinnacle_daubechies_filter(length)
    low, details = _decompose(source, scaling, wavelet, level_count)
    finest_hh = details[0][2]
    if convention == "paper":
        center = float(np.median(finest_hh))
        noise = float(np.median(np.abs(finest_hh - center)) / 0.6745)
        keep_equal = True
    else:
        noise = float(np.median(np.abs(finest_hh)) / 0.67)
        keep_equal = False
    threshold = multiplier * noise
    if not isfinite(noise) or not isfinite(threshold):
        raise ArithmeticError("wavelet noise estimate or threshold is not representable")
    for triplet in details:
        for band in triplet:
            if keep_equal:
                band[np.abs(band) < threshold] = 0.0
            else:
                band[np.abs(band) <= threshold] = 0.0
    denoised = _synthesize(low, details, scaling, wavelet)
    return PinnacleDenoiseResult(
        _freeze(denoised),
        (bounds[0], bounds[2]),
        bounds,
        scaling,
        level_count,
        length,
        noise,
        threshold,
        multiplier,
        convention,
    )
