"""Reconstruct WFMM covariance functions from coefficient variances."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray
from .wfmm_basis import WFMMBasis, wfmm_inverse

_MAX_INPUT_CELLS = 2_000_000
_MAX_OUTPUT_CELLS = 2_000_000
_MAX_LIVE_CELLS = 4_000_000
_MAX_WORK = 250_000_000
_BASIS_CHUNK_ROWS = 128


@dataclass(frozen=True)
class WFMMCovarianceSummary:
    """Posterior summaries of coefficient variances and covariance functions.

    ``correlation_from_mean_variance`` is the correlation obtained by
    normalizing the covariance reconstructed from posterior mean omega. It is
    a plug-in correlation, not the posterior mean of draw-wise correlations.
    Covariance draw arrays are retained only when explicitly requested.
    """

    variance_mean: FloatArray
    variance_standard_deviation: FloatArray
    variance_quantiles: FloatArray
    variance_function_mean: FloatArray
    variance_function_standard_deviation: FloatArray
    variance_function_quantiles: FloatArray
    quantile_probabilities: FloatArray
    covariance_mean: FloatArray | None
    correlation_from_mean_variance: FloatArray | None
    covariance_draws: FloatArray | None
    coefficient_scale: NDArray[np.int64]
    coefficient_partition: NDArray[np.int64]


def _validate_variances(variances: ArrayLike, basis: WFMMBasis) -> FloatArray:
    if not isinstance(basis, WFMMBasis):
        raise ValueError("basis must be a WFMMBasis")
    raw = np.asarray(variances)
    if raw.ndim < 1 or raw.shape[-1] != basis.time_count or raw.size > _MAX_INPUT_CELLS:
        raise ValueError("variances must end in K=time_count and fit the bounded input size")
    if raw.dtype.kind not in "iuf":
        raise ValueError("variances must be real-valued")
    values = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("variances must be finite and nonnegative")
    return values


def _per_basis_row_work(basis: WFMMBasis) -> int:
    if basis.transform == "wavelet":
        if basis.filter_length is None:
            raise ValueError("wavelet basis is missing its filter length")
        return 2 * basis.time_count * basis.filter_length * basis.levels
    if basis.transform == "custom":
        return basis.time_count * basis.time_count
    return basis.time_count


def _inverse_basis_rows(basis: WFMMBasis, start: int, end: int) -> FloatArray:
    time_count = basis.time_count
    indices = np.arange(start, end)
    identity_rows = np.zeros((end - start, time_count), dtype=np.float64)
    identity_rows[np.arange(end - start), indices] = 1.0
    return wfmm_inverse(identity_rows, basis)


def _basis_rows(basis: WFMMBasis) -> FloatArray:
    """Return inverse-transform basis rows (coefficient index,time index)."""
    if basis.transform == "identity":
        return np.eye(basis.time_count, dtype=np.float64)
    if basis.transform == "custom":
        return basis.synthesis_matrix
    rows = np.empty((basis.time_count, basis.time_count), dtype=np.float64)
    chunk_rows = min(
        _BASIS_CHUNK_ROWS,
        basis.max_work // _per_basis_row_work(basis),
    )
    if chunk_rows < 1:
        raise ValueError("basis max_work is too small to reconstruct one inverse-basis row")
    for start in range(0, basis.time_count, chunk_rows):
        end = min(start + chunk_rows, basis.time_count)
        rows[start:end] = _inverse_basis_rows(basis, start, end)
    return rows


def wfmm_covariance(variances: ArrayLike, basis: WFMMBasis) -> FloatArray:
    """Reconstruct ``W diag(omega) W.T`` for each supplied coefficient vector.

    The input can have arbitrary leading dimensions and ends with K coefficient
    variances. Output has the same leading dimensions followed by ``(T,T)``.
    Dense covariance output is limited to two million cells including the
    supplied coefficient values and a work matrix.
    """
    values = _validate_variances(variances, basis)
    leading = values.shape[:-1]
    vector_count = int(np.prod(leading, dtype=np.int64)) if leading else 1
    time_count = basis.time_count
    output_cells = vector_count * time_count * time_count
    if output_cells > _MAX_OUTPUT_CELLS:
        raise ValueError("dense WFMM covariance output exceeds 2000000 cells")
    covariance_work = (
        vector_count * time_count**2
        if basis.transform == "identity"
        else vector_count * time_count**3
    )
    basis_work = time_count * _per_basis_row_work(basis) if basis.transform == "wavelet" else 0
    work = covariance_work + basis_work
    if work > _MAX_WORK:
        raise ValueError("WFMM covariance reconstruction exceeds the work limit")
    workspace_cells = 0 if basis.transform == "identity" else 3 * time_count * time_count
    live_cells = 2 * values.size + 2 * output_cells + workspace_cells
    if live_cells > _MAX_LIVE_CELLS:
        raise ValueError("WFMM covariance reconstruction exceeds the live-cell limit")

    flattened = values.reshape(vector_count, time_count)
    output = np.empty((vector_count, time_count, time_count), dtype=np.float64)
    if basis.transform == "identity":
        output.fill(0.0)
        diagonal = np.arange(time_count)
        output[:, diagonal, diagonal] = flattened
    else:
        inverse_rows = _basis_rows(basis)
        for index, omega in enumerate(flattened):
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                weighted_rows = np.sqrt(omega)[:, None] * inverse_rows
                covariance = weighted_rows.T @ weighted_rows
            covariance = 0.5 * covariance + 0.5 * covariance.T
            if not np.all(np.isfinite(covariance)):
                raise ArithmeticError("WFMM covariance is not representable")
            output[index] = covariance
    if not np.all(np.isfinite(output)):
        raise ArithmeticError("WFMM covariance is not representable")
    return _freeze(output.reshape((*leading, time_count, time_count)))


def _draw_variance_functions(variances: FloatArray, basis: WFMMBasis) -> FloatArray:
    """Apply only the covariance diagonal transform, without dense draws."""
    chains, draws, components, coefficient_count = variances.shape
    sample_count = chains * draws
    output = np.empty((sample_count, components, coefficient_count), dtype=np.float64)
    flattened = variances.reshape(sample_count, components, coefficient_count)
    if basis.transform == "identity":
        output[:] = flattened
        return output
    rows_per_call = min(
        _BASIS_CHUNK_ROWS,
        basis.max_work // _per_basis_row_work(basis),
    )
    if rows_per_call < 1:
        raise ValueError("basis max_work is too small to reconstruct one inverse-basis row")
    output.fill(0.0)
    for start in range(0, coefficient_count, rows_per_call):
        end = min(start + rows_per_call, coefficient_count)
        if basis.transform == "custom":
            inverse_rows = basis.synthesis_matrix[start:end]
        else:
            inverse_rows = _inverse_basis_rows(basis, start, end)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            for offset, basis_row in enumerate(inverse_rows):
                root_weight = np.sqrt(flattened[:, :, start + offset])[:, :, None]
                contribution_root = root_weight * basis_row[None, None, :]
                output += contribution_root * contribution_root
    if not np.all(np.isfinite(output)):
        raise ArithmeticError("WFMM variance function is not representable")
    return output


def _summaries(draws: FloatArray, probabilities: FloatArray) -> tuple[FloatArray, ...]:
    scale = np.max(draws, axis=0)
    normalized = np.divide(draws, scale, out=np.zeros_like(draws), where=scale > 0.0)
    mean_normalized = np.mean(normalized, axis=0)
    mean = scale * mean_normalized
    centered = normalized - mean_normalized
    np.square(centered, out=centered)
    sd_normalized = np.sqrt(np.sum(centered, axis=0) / (draws.shape[0] - 1))
    with np.errstate(over="ignore", invalid="ignore"):
        standard_deviation = scale * sd_normalized
        quantile_values = np.quantile(normalized, probabilities, axis=0, method="linear") * scale
    if not all(np.all(np.isfinite(value)) for value in (mean, standard_deviation, quantile_values)):
        raise ArithmeticError("WFMM covariance summaries are not representable")
    return mean, standard_deviation, quantile_values


def wfmm_summarize_covariance(
    variance_draws: ArrayLike,
    basis: WFMMBasis,
    *,
    quantiles: ArrayLike = (0.025, 0.5, 0.975),
    include_covariance: bool = False,
    retain_covariance_draws: bool = False,
) -> WFMMCovarianceSummary:
    """Summarize posterior coefficient variances and data-time covariance.

    ``variance_draws`` has shape ``(chains,draws,components,K)``. Variance
    functions are each draw's data-time covariance diagonal. Covariance means
    and correlations are optional; retained full covariance draws require
    ``include_covariance=True``. Aggregate intermediate, retained and output
    arrays are preflighted against fixed cell and work limits.
    """
    values = _validate_variances(variance_draws, basis)
    if values.ndim != 4 or values.shape[0] < 1 or values.shape[1] < 2 or values.shape[2] < 1:
        raise ValueError("variance_draws must have shape (chains,draws,components,K)")
    if not isinstance(include_covariance, (bool, np.bool_)):
        raise ValueError("include_covariance must be boolean")
    if not isinstance(retain_covariance_draws, (bool, np.bool_)):
        raise ValueError("retain_covariance_draws must be boolean")
    if retain_covariance_draws and not include_covariance:
        raise ValueError("retaining covariance draws requires include_covariance=True")
    raw_quantiles = np.asarray(quantiles)
    if raw_quantiles.ndim != 1 or raw_quantiles.size < 1 or raw_quantiles.size > 128:
        raise ValueError("quantiles must be a bounded one-dimensional probability vector")
    if raw_quantiles.dtype.kind not in "iuf":
        raise ValueError("quantiles must be real-valued")
    probabilities = np.asarray(raw_quantiles, dtype=np.float64)
    if (
        not np.all(np.isfinite(probabilities))
        or np.any((probabilities <= 0.0) | (probabilities >= 1.0))
        or np.any(np.diff(probabilities) <= 0.0)
    ):
        raise ValueError("quantiles must be strictly increasing probabilities in (0,1)")

    chains, draws, components, coefficient_count = values.shape
    sample_count = chains * draws
    time_count = basis.time_count
    row_work = _per_basis_row_work(basis)
    basis_chunk_rows = min(_BASIS_CHUNK_ROWS, basis.max_work // row_work)
    if basis.transform != "identity" and basis_chunk_rows < 1:
        raise ValueError("basis max_work is too small to reconstruct inverse-basis rows")
    input_cells = values.size
    variance_function_cells = sample_count * components * time_count
    coefficient_summary_cells = components * coefficient_count * (3 + probabilities.size)
    function_summary_cells = components * time_count * (3 + probabilities.size)
    covariance_cells = components * time_count * time_count if include_covariance else 0
    retained_covariance_cells = (
        sample_count * components * time_count * time_count if retain_covariance_draws else 0
    )
    retained_output_cells = (
        coefficient_summary_cells
        + function_summary_cells
        + covariance_cells * 2
        + retained_covariance_cells
    )
    maximum_basis_cells = (
        3 * time_count * time_count
        if include_covariance
        else 0
        if basis.transform == "identity"
        else 3 * basis_chunk_rows * time_count
    )
    if retained_output_cells > _MAX_OUTPUT_CELLS:
        raise ValueError("WFMM posterior covariance outputs exceed 2000000 cells")
    duplicate_output_cells = (
        coefficient_summary_cells
        + function_summary_cells
        + covariance_cells * 2
        + retained_covariance_cells
    )
    live_cells = (
        4 * input_cells
        + 4 * variance_function_cells
        + retained_output_cells
        + duplicate_output_cells
        + maximum_basis_cells
    )
    if live_cells > _MAX_LIVE_CELLS:
        raise ValueError("WFMM covariance summary exceeds the 4000000 live-cell limit")

    basis_reconstructions = 1 + int(include_covariance) * (1 + int(retain_covariance_draws))
    basis_generation_work = (
        0
        if basis.transform in ("identity", "custom")
        else basis_reconstructions * time_count * row_work
    )
    variance_function_work = (
        sample_count * components * time_count
        if basis.transform == "identity"
        else sample_count * components * coefficient_count * time_count
    )
    covariance_multiplicity = components * (1 + sample_count if retain_covariance_draws else 1)
    matrix_work = (
        0
        if basis.transform == "identity" or not include_covariance
        else covariance_multiplicity * time_count**3
    )
    if basis_generation_work + variance_function_work + matrix_work > _MAX_WORK:
        raise ValueError("WFMM covariance summary exceeds the 250000000-operation limit")
    flattened_draws = values.reshape(sample_count, components, coefficient_count)
    coefficient_mean, coefficient_sd, coefficient_quantiles = _summaries(
        flattened_draws, probabilities
    )
    variance_function_draws = _draw_variance_functions(values, basis)
    function_mean, function_sd, function_quantiles = _summaries(
        variance_function_draws, probabilities
    )

    covariance_mean = None
    correlation = None
    covariance_draws = None
    if include_covariance:
        covariance_mean = wfmm_covariance(coefficient_mean, basis)
        diagonal = np.diagonal(covariance_mean, axis1=-2, axis2=-1).copy()
        if np.any(diagonal <= 0.0):
            raise ValueError("plug-in correlation is undefined for a zero covariance diagonal")
        standard_deviation = np.sqrt(diagonal)
        correlation = (
            covariance_mean / standard_deviation[:, :, None] / standard_deviation[:, None, :]
        )
        correlation = 0.5 * (correlation + np.swapaxes(correlation, -1, -2))
        if not np.all(np.isfinite(correlation)):
            raise ArithmeticError("plug-in WFMM correlation is not representable")
        if not np.allclose(diagonal, function_mean, rtol=2e-12, atol=2e-14):
            raise ArithmeticError("covariance and variance-function means are inconsistent")
        if retain_covariance_draws:
            covariance_draws = wfmm_covariance(values, basis).reshape(
                chains, draws, components, time_count, time_count
            )

    return WFMMCovarianceSummary(
        _freeze(coefficient_mean),
        _freeze(coefficient_sd),
        _freeze(coefficient_quantiles),
        _freeze(function_mean),
        _freeze(function_sd),
        _freeze(function_quantiles),
        _freeze(probabilities),
        None if covariance_mean is None else _freeze(covariance_mean),
        None if correlation is None else _freeze(correlation),
        None if covariance_draws is None else _freeze(covariance_draws),
        basis.coefficient_scale,
        basis.coefficient_partition,
    )
