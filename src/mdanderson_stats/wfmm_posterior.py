"""Reconstruct WFMM coefficient draws and summarize functional effects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray, scalar
from .wfmm_basis import WFMMBasis, wfmm_inverse

_MAX_INPUT_CELLS = 2_000_000
_MAX_CURVE_CELLS = 2_000_000
_MAX_WORK = 200_000_000
_INVERSE_CHUNK_ROWS = 128
_MAX_EFFECTS = 128
_MAX_SUMMARY_CELLS = 2_000_000


@dataclass(frozen=True)
class WFMMPosteriorSummary:
    """Posterior functional summaries; draw order is chain then iteration.

    Curves are optionally retained in ``(chain,draw,effect_contrast,time_region)``
    order. Pointwise quantiles use NumPy's linear interpolation convention.
    Simultaneous bands use each draw's maximum absolute standardized deviation
    over time regions, separately for each effect contrast.
    """

    mean: FloatArray
    standard_deviation: FloatArray
    quantiles: FloatArray
    quantile_probabilities: FloatArray
    effect_size_probability: FloatArray
    effect_sizes: FloatArray
    sign_tail_score: FloatArray
    simultaneous_lower: FloatArray
    simultaneous_upper: FloatArray
    simultaneous_critical_value: FloatArray
    simbas_probability: FloatArray
    effect_contrast: FloatArray | None
    time_contrast: FloatArray | None
    coefficient_scale: NDArray[np.int64]
    coefficient_partition: NDArray[np.int64]
    confidence: float
    curves: FloatArray | None


def _real_matrix(value: ArrayLike, name: str, *, ndim: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.ndim != ndim or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real {ndim}D array")
    array = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _safe_moments(
    draws: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Scale each output cell before means, sample SDs and quantiles."""
    scale = np.max(np.abs(draws), axis=0)
    normalized = np.divide(draws, scale, out=np.zeros_like(draws), where=scale > 0.0)
    mean_normalized = np.mean(normalized, axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = scale * mean_normalized
    centered = normalized - mean_normalized
    standard_normalized = np.sqrt(np.sum(centered * centered, axis=0) / (draws.shape[0] - 1))
    with np.errstate(over="ignore", invalid="ignore"):
        standard_deviation = scale * standard_normalized
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(standard_deviation)):
        raise ArithmeticError("WFMM posterior moments are not representable")
    return scale, normalized, mean_normalized, mean, standard_normalized


def wfmm_summarize(
    coefficients: ArrayLike,
    basis: WFMMBasis,
    *,
    effect_contrast: ArrayLike | None = None,
    time_contrast: ArrayLike | None = None,
    confidence: float = 0.95,
    quantiles: ArrayLike = (0.025, 0.5, 0.975),
    effect_sizes: ArrayLike = 0.0,
    retain_curves: bool = False,
) -> WFMMPosteriorSummary:
    """Inverse-transform posterior coefficients and calculate function summaries.

    ``coefficients`` has shape ``(chains,draws,effects,K)``. The optional
    effect contrast is ``(contrasts,effects)``; the optional time contrast is
    ``(time_points,regions)``. They act as ``L @ curves @ LT``. Effect-size
    probabilities have an explicit leading threshold axis; a scalar threshold
    is treated as a length-one vector. All transforms and posterior outputs
    are checked against fixed cell/work bounds before expansion.
    """
    if not isinstance(basis, WFMMBasis):
        raise ValueError("basis must be a WFMMBasis")
    raw = np.asarray(coefficients)
    if raw.ndim != 4 or raw.dtype.kind not in "iuf":
        raise ValueError("coefficients must be a real (chains,draws,effects,K) array")
    chain_count, draw_count, effect_count, coefficient_count = map(int, raw.shape)
    if (
        chain_count < 1
        or draw_count < 2
        or not 1 <= effect_count <= _MAX_EFFECTS
        or coefficient_count != basis.time_count
        or raw.size > _MAX_INPUT_CELLS
    ):
        raise ValueError("coefficient dimensions exceed supported WFMM summary bounds")
    if not np.all(np.isfinite(raw)):
        raise ValueError("coefficients must be finite")

    if not isinstance(retain_curves, (bool, np.bool_)):
        raise ValueError("retain_curves must be boolean")
    raw_confidence = np.asarray(confidence)
    if raw_confidence.ndim != 0 or raw_confidence.dtype.kind not in "iuf":
        raise ValueError("confidence must be a real scalar")
    confidence_value = scalar(confidence, "confidence")
    if not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    raw_quantiles = np.asarray(quantiles)
    if raw_quantiles.ndim != 1 or raw_quantiles.size > 128:
        raise ValueError("quantiles must be a bounded one-dimensional probability array")
    quantile_values = _real_matrix(raw_quantiles, "quantiles", ndim=1)
    if (
        quantile_values.size < 1
        or quantile_values.size > 128
        or np.any((quantile_values <= 0.0) | (quantile_values >= 1.0))
        or np.any(np.diff(quantile_values) <= 0.0)
    ):
        raise ValueError("quantiles must be strictly increasing probabilities in (0,1)")

    effect_matrix: FloatArray | None = None
    if effect_contrast is not None:
        raw_effect = np.asarray(effect_contrast)
        if (
            raw_effect.ndim != 2
            or raw_effect.shape[1] != effect_count
            or raw_effect.shape[0] < 1
            or raw_effect.shape[0] > _MAX_EFFECTS
            or raw_effect.dtype.kind not in "iuf"
        ):
            raise ValueError("effect_contrast must have shape (1..128,effects)")
        if raw_effect.size > _MAX_INPUT_CELLS:
            raise ValueError("effect_contrast exceeds the bounded matrix size")
        effect_matrix = _real_matrix(raw_effect, "effect_contrast", ndim=2)
    output_effects = effect_count if effect_matrix is None else effect_matrix.shape[0]

    time_matrix: FloatArray | None = None
    if time_contrast is not None:
        raw_time = np.asarray(time_contrast)
        if (
            raw_time.ndim != 2
            or raw_time.shape[0] != basis.time_count
            or raw_time.shape[1] < 1
            or raw_time.shape[1] > basis.time_count
            or raw_time.dtype.kind not in "iuf"
        ):
            raise ValueError("time_contrast must have shape (time_points,1..time_points)")
        if raw_time.size > _MAX_INPUT_CELLS:
            raise ValueError("time_contrast exceeds the bounded matrix size")
        time_matrix = _real_matrix(raw_time, "time_contrast", ndim=2)
    output_times = basis.time_count if time_matrix is None else time_matrix.shape[1]
    sample_count = chain_count * draw_count
    output_cells = sample_count * output_effects * output_times
    if output_cells > _MAX_CURVE_CELLS:
        raise ValueError("reconstructed posterior curves exceed the 2000000-cell limit")
    intermediate_cells = sample_count * output_effects * basis.time_count
    if intermediate_cells > _MAX_CURVE_CELLS:
        raise ValueError("pre-time-contrast posterior curves exceed the 2000000-cell limit")
    if effect_sizes is None:
        raise ValueError("effect_sizes must be a nonnegative scalar or vector")
    raw_sizes = np.asarray(effect_sizes)
    if (
        raw_sizes.dtype.kind not in "iuf"
        or raw_sizes.ndim > 1
        or raw_sizes.size == 0
        or raw_sizes.size > 128
    ):
        raise ValueError("effect_sizes must be a nonnegative scalar or vector")
    size_values = np.asarray(raw_sizes, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(size_values)) or np.any(size_values < 0.0):
        raise ValueError("effect_sizes must be finite and nonnegative")
    summary_cells = output_effects * output_times * (quantile_values.size + size_values.size + 7)
    contrast_cells = 0
    if effect_matrix is not None:
        contrast_cells += 2 * effect_matrix.size
    if time_matrix is not None:
        contrast_cells += 2 * time_matrix.size
    total_output_cells = summary_cells + contrast_cells + (output_cells if retain_curves else 0)
    if total_output_cells > _MAX_SUMMARY_CELLS:
        raise ValueError("posterior summary arrays exceed the 2000000-cell limit")

    rows_to_inverse = sample_count * effect_count
    if basis.transform == "wavelet":
        per_row_work = 2 * basis.time_count * int(basis.filter_length or 1) * basis.levels
    elif basis.transform == "custom":
        per_row_work = basis.time_count * basis.time_count
    else:
        per_row_work = basis.time_count
    inverse_work = rows_to_inverse * per_row_work
    effect_work = (
        0
        if effect_matrix is None
        else sample_count * output_effects * effect_count * basis.time_count
    )
    threshold_work = size_values.size * output_cells
    time_work = (
        0
        if time_matrix is None
        else sample_count * output_effects * basis.time_count * output_times
    )
    if inverse_work + effect_work + time_work + threshold_work > _MAX_WORK:
        raise ValueError(
            "posterior transform and contrast work exceeds the 200000000-operation limit"
        )
    chunk_rows = min(_INVERSE_CHUNK_ROWS, basis.max_work // per_row_work)
    if chunk_rows < 1:
        raise ValueError("basis max_work is too small for one posterior inverse row")

    source = np.asarray(raw, dtype=np.float64).reshape(
        sample_count * effect_count, coefficient_count
    )
    reconstructed = np.empty((sample_count * effect_count, basis.time_count), dtype=np.float64)
    for start in range(0, source.shape[0], chunk_rows):
        end = min(start + chunk_rows, source.shape[0])
        reconstructed[start:end] = wfmm_inverse(source[start:end], basis)
    curves = reconstructed.reshape(sample_count, effect_count, basis.time_count)
    if effect_matrix is not None:
        curves = np.einsum("le,set->slt", effect_matrix, curves, optimize=True)
    if time_matrix is not None:
        curves = np.einsum("slt,tr->slr", curves, time_matrix, optimize=True)
    if not np.all(np.isfinite(curves)):
        raise ArithmeticError("posterior contrasts produced nonfinite functional values")

    scale, normalized, mean_normalized, mean, sd_normalized = _safe_moments(curves)
    with np.errstate(over="ignore", invalid="ignore"):
        standard_deviation = scale * sd_normalized
    if not np.all(np.isfinite(standard_deviation)):
        raise ArithmeticError("posterior standard deviations are not representable")
    pointwise = np.quantile(normalized, quantile_values, axis=0, method="linear") * scale
    if not np.all(np.isfinite(pointwise)):
        raise ArithmeticError("posterior quantiles are not representable")

    effect_probability = np.empty((size_values.size, output_effects, output_times))
    absolute_draws = np.abs(curves)
    for index, size in enumerate(size_values):
        effect_probability[index] = np.mean(absolute_draws > size, axis=0)
    positive_probability = np.mean(curves > 0.0, axis=0)
    negative_probability = np.mean(curves < 0.0, axis=0)
    sign_score = 2.0 * np.minimum(positive_probability, negative_probability)

    centered = normalized - mean_normalized
    z_draws = np.divide(
        centered,
        sd_normalized,
        out=np.zeros_like(centered),
        where=sd_normalized > 0.0,
    )
    max_z = np.max(np.abs(z_draws), axis=2)
    critical = np.quantile(max_z, confidence_value, axis=0, method="linear")
    with np.errstate(over="ignore", invalid="ignore"):
        half_width = critical[:, None] * standard_deviation
        simultaneous_lower = mean - half_width
        simultaneous_upper = mean + half_width
    if not all(
        np.all(np.isfinite(value))
        for value in (
            effect_probability,
            sign_score,
            critical,
            simultaneous_lower,
            simultaneous_upper,
        )
    ):
        raise ArithmeticError("posterior probability or simultaneous interval is not representable")

    standardized_mean = np.divide(
        np.abs(mean_normalized), sd_normalized, out=np.zeros_like(mean), where=sd_normalized > 0.0
    )
    constant_nonzero = (sd_normalized == 0.0) & (mean != 0.0)
    standardized_mean[constant_nonzero] = np.inf
    # For a constant zero curve, both sides of the empirical event are zero,
    # so P(max_Z >= 0) is exactly one.
    simbas = np.mean(max_z[:, :, None] >= standardized_mean[None, :, :], axis=0)
    simbas[constant_nonzero] = 0.0

    result_curves = None
    if retain_curves:
        result_curves = _freeze(
            curves.reshape(chain_count, draw_count, output_effects, output_times)
        )
    return WFMMPosteriorSummary(
        _freeze(mean),
        _freeze(standard_deviation),
        _freeze(pointwise),
        _freeze(quantile_values),
        _freeze(effect_probability),
        _freeze(size_values),
        _freeze(sign_score),
        _freeze(simultaneous_lower),
        _freeze(simultaneous_upper),
        _freeze(critical),
        _freeze(simbas),
        None if effect_matrix is None else _freeze(effect_matrix),
        None if time_matrix is None else _freeze(time_matrix),
        basis.coefficient_scale,
        basis.coefficient_partition,
        confidence_value,
        result_curves,
    )
