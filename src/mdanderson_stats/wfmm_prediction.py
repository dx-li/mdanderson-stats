"""Posterior predictive WFMM coefficients under explicit future designs.

This is a Python prediction interface for the fitted coefficient model, not a
native WFMM prediction-file workflow. Existing random levels use retained
conditional draws; new levels are drawn independently from their fitted
coefficient-specific random-effect variances.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray
from .wfmm_model import _MAX_COEFFICIENTS, _MAX_FIXED_EFFECTS, _MAX_ROWS, WFMMCoefficientFit

_MAX_PREDICTION_CELLS = 2_000_000
_MAX_PREDICTION_WORK = 250_000_000


def _matrix_input(
    value: ArrayLike | None, name: str, rows: int, columns: int | None
) -> NDArray[np.generic] | None:
    if value is None:
        return None
    raw = np.asarray(value)
    if (
        raw.ndim != 2
        or raw.shape[0] != rows
        or (columns is not None and raw.shape[1] != columns)
        or raw.size > _MAX_PREDICTION_CELLS
        or np.iscomplexobj(raw)
        or raw.dtype.kind not in "iuf"
        or not np.all(np.isfinite(raw))
    ):
        shape = f" with {columns} columns" if columns is not None else ""
        raise ValueError(f"{name} must be a finite real matrix with {rows} rows{shape}")
    return raw


def _label_input(value: ArrayLike | None, name: str, size: int, groups: int) -> NDArray[np.generic]:
    if value is None:
        raise ValueError(f"{name} must be supplied for stochastic components")
    raw = np.asarray(value)
    if raw.shape != (size,) or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer vector of length {size}")
    if np.any((raw < 0) | (raw >= groups)):
        raise ValueError(f"{name} labels must index the fitted variance components")
    return raw


def _freeze(value: ArrayLike) -> FloatArray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _bounded_integer(value: int, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return result


def wfmm_predict_coefficients(
    fit: WFMMCoefficientFit,
    fixed_design: ArrayLike,
    *,
    existing_random_design: ArrayLike | None = None,
    new_random_design: ArrayLike | None = None,
    new_random_strata: ArrayLike | None = None,
    include_residual: bool = False,
    residual_strata: ArrayLike | None = None,
    max_work: int = _MAX_PREDICTION_WORK,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Draw future curves in coefficient space from an explicit design.

    ``fixed_design`` has shape ``(new_rows, fixed_effects)``. The optional
    ``existing_random_design`` uses exactly the fitted random-level columns and
    multiplies them by retained conditional random-effect draws. The optional
    ``new_random_design`` has one column per new random level; rows sharing a
    column share its independently generated random effect. Its integer
    ``new_random_strata`` maps columns to fitted random-variance groups.

    With ``include_residual=False``, output contains latent means. For new
    random levels those draws include uncertainty in the new latent effects.
    With ``include_residual=True``, independent row-specific residual draws
    are added using ``residual_strata``. A Generator is required only when new
    random effects or residuals are drawn. The result has shape
    ``(chain, draw, new_row, coefficient)`` and can be passed to
    ``wfmm_summarize(result, basis)`` for curve-space summaries.

    The model assumes independent random levels and independent residual rows
    conditional on fitted variance components. This function implements that
    model under a Python prediction interface; native prediction-file parity
    is not established.
    """
    if not isinstance(fit, WFMMCoefficientFit):
        raise ValueError("fit must be a WFMMCoefficientFit")
    beta = np.asarray(fit.coefficients)
    if beta.ndim != 4:
        raise ValueError("fit.coefficients must have shape (chain,draw,fixed_effect,coefficient)")
    chains, draws, fixed_effects, coefficients = beta.shape
    if (
        chains < 1
        or draws < 2
        or not 1 <= fixed_effects <= _MAX_FIXED_EFFECTS
        or not 1 <= coefficients <= _MAX_COEFFICIENTS
        or not np.all(np.isfinite(beta))
    ):
        raise ValueError("fit coefficient draws are outside supported dimensions or nonfinite")

    raw_x = np.asarray(fixed_design)
    if (
        raw_x.ndim != 2
        or raw_x.shape[1] != fixed_effects
        or not 1 <= raw_x.shape[0] <= _MAX_ROWS
        or raw_x.size > _MAX_PREDICTION_CELLS
        or np.iscomplexobj(raw_x)
        or raw_x.dtype.kind not in "iuf"
        or not np.all(np.isfinite(raw_x))
    ):
        raise ValueError(
            "fixed_design must be a bounded finite real (new_rows,fixed_effects) matrix"
        )
    rows = int(raw_x.shape[0])

    random_variances = np.asarray(fit.random_variances)
    residual_variances = np.asarray(fit.residual_variances)
    if (
        random_variances.ndim != 4
        or random_variances.shape[:2] != (chains, draws)
        or random_variances.shape[-1] != coefficients
        or not np.all(np.isfinite(random_variances))
        or np.any(random_variances < 0.0)
    ):
        raise ValueError("fit.random_variances have an invalid shape or values")
    if (
        residual_variances.ndim != 4
        or residual_variances.shape[:2] != (chains, draws)
        or residual_variances.shape[-1] != coefficients
        or not np.all(np.isfinite(residual_variances))
        or np.any(residual_variances < 0.0)
    ):
        raise ValueError("fit.residual_variances have an invalid shape or values")
    if not isinstance(include_residual, (bool, np.bool_)):
        raise ValueError("include_residual must be boolean")

    existing_effects = None if fit.random_effects is None else np.asarray(fit.random_effects)
    if existing_random_design is not None and existing_effects is None:
        raise ValueError("existing random-effect draws were not retained in the fit")
    if existing_effects is None:
        existing_levels = 0
    else:
        if (
            existing_effects.ndim != 4
            or existing_effects.shape[:2] != (chains, draws)
            or existing_effects.shape[-1] != coefficients
            or not np.all(np.isfinite(existing_effects))
        ):
            raise ValueError("fit.random_effects have an invalid shape or values")
        existing_levels = int(existing_effects.shape[2])
    raw_z_existing = _matrix_input(
        existing_random_design, "existing_random_design", rows, existing_levels
    )

    raw_z_new = _matrix_input(new_random_design, "new_random_design", rows, None)
    if raw_z_new is not None and raw_z_new.shape[1] < 1:
        raise ValueError("new_random_design must have at least one level column")
    if new_random_design is None and new_random_strata is not None:
        raise ValueError("new_random_strata requires new_random_design")
    raw_new_labels = None
    if raw_z_new is not None:
        raw_new_labels = _label_input(
            new_random_strata,
            "new_random_strata",
            int(raw_z_new.shape[1]),
            int(random_variances.shape[2]),
        )

    if include_residual:
        residual_groups = int(residual_variances.shape[2])
        if residual_groups < 1:
            raise ValueError("fit has no residual variance groups")
        raw_residual_labels = _label_input(
            residual_strata, "residual_strata", rows, residual_groups
        )
    elif residual_strata is not None:
        raise ValueError("residual_strata is used only when include_residual=True")
    else:
        raw_residual_labels = None

    stochastic = raw_z_new is not None or bool(include_residual)
    if stochastic and not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit Generator when drawing new effects or residuals")
    if not stochastic and rng is not None and not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a NumPy Generator or None")

    fixed_cells = raw_x.size
    existing_cells = 0 if raw_z_existing is None else raw_z_existing.size
    new_cells = 0 if raw_z_new is None else raw_z_new.size
    output_cells = chains * draws * rows * coefficients
    new_levels = 0 if raw_z_new is None else int(raw_z_new.shape[1])
    live_cells = (
        2 * output_cells
        + 2 * (fixed_cells + existing_cells + new_cells)
        + 8 * rows
        + 4 * new_levels
    )
    work = (
        chains
        * draws
        * rows
        * coefficients
        * (
            fixed_effects
            + (0 if raw_z_existing is None else raw_z_existing.shape[1])
            + (0 if raw_z_new is None else raw_z_new.shape[1])
            + 1
        )
    )
    budget = _bounded_integer(max_work, "max_work", 1, _MAX_PREDICTION_WORK)
    if output_cells > _MAX_PREDICTION_CELLS or live_cells > 2 * _MAX_PREDICTION_CELLS:
        raise ValueError("WFMM predictive output and working arrays exceed the cell limit")
    if work > budget:
        raise ValueError("WFMM predictive computation exceeds max_work")

    # Make bounded copies only after all dimension, work and storage checks.
    x = np.array(raw_x, dtype=np.float64, copy=True)
    z_existing = (
        np.zeros((rows, 0), dtype=np.float64)
        if raw_z_existing is None
        else np.array(raw_z_existing, dtype=np.float64, copy=True)
    )
    z_new = (
        np.zeros((rows, 0), dtype=np.float64)
        if raw_z_new is None
        else np.array(raw_z_new, dtype=np.float64, copy=True)
    )
    new_labels = None if raw_new_labels is None else np.asarray(raw_new_labels, dtype=np.int64)
    residual_labels = (
        None if raw_residual_labels is None else np.asarray(raw_residual_labels, dtype=np.int64)
    )

    prediction = np.empty((chains, draws, rows, coefficients), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for chain in range(chains):
            for draw in range(draws):
                for coefficient in range(coefficients):
                    values = x @ beta[chain, draw, :, coefficient]
                    if existing_effects is not None and raw_z_existing is not None:
                        values = values + z_existing @ existing_effects[chain, draw, :, coefficient]
                    if new_labels is not None:
                        assert rng is not None
                        new_effect = np.sqrt(
                            random_variances[chain, draw, new_labels, coefficient]
                        ) * rng.normal(size=new_labels.size)
                        values = values + z_new @ new_effect
                    if residual_labels is not None:
                        assert rng is not None
                        residual = np.sqrt(
                            residual_variances[chain, draw, residual_labels, coefficient]
                        ) * rng.normal(size=rows)
                        values = values + residual
                    prediction[chain, draw, :, coefficient] = values
    if not np.all(np.isfinite(prediction)):
        raise ArithmeticError("WFMM predictive coefficients are not representable")
    return _freeze(prediction)
