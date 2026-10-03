"""Wild-bootstrap resampling for Kong and Lee's semiparametric surface.

This implements the documented residual/multiplier construction and refitting
workflow. The returned draw standard deviation is a descriptive sample SD
(centered on the draw mean, denominator B-1); it is not a source-defined
normal-interval convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray
from .synergy_surface import (
    _MAX_WORK,
    SynergySurfaceFit,
    _dose_vectors,
    fit_synergy_surface,
)

_MAX_BOOTSTRAP_REPLICATES = 2_000
_MAX_BOOTSTRAP_STORAGE_CELLS = 1_000_000
_MAX_BOOTSTRAP_WORK = 2_000_000_000
_SQRT5 = float(np.sqrt(5.0))
_MAMMEN_VALUES = np.array([(1.0 - _SQRT5) / 2.0, (1.0 + _SQRT5) / 2.0])
_MAMMEN_PROBABILITIES = np.array([(_SQRT5 + 1.0) / (2.0 * _SQRT5), (_SQRT5 - 1.0) / (2.0 * _SQRT5)])


@dataclass(frozen=True)
class SynergySurfaceBootstrap:
    """Original fit and source-style wild-bootstrap departure draws.

    ``departure_draws`` is ``None`` when ``store_draws=False``. The mean and
    sample SD summarize draws row-wise at the original observations; SD uses
    the Python descriptive convention (draw mean, ``ddof=1``), not a native
    interval formula. Replicate smoothing diagnostics retain one row per draw.
    """

    fit: SynergySurfaceFit
    original_departure: FloatArray
    departure_mean: FloatArray
    departure_standard_deviation: FloatArray
    departure_draws: FloatArray | None
    replicate_smoothing_parameters: FloatArray
    replicate_reml_objectives: FloatArray
    replicate_residual_variances_scaled: FloatArray
    replicate_response_scales: FloatArray
    replicate_optimizer_evaluations: np.ndarray
    replicate_smoothing_at_boundary: np.ndarray
    lower_smoothing_fit: SynergySurfaceFit
    upper_smoothing_fit: SynergySurfaceFit
    replicates: int
    multiplier_tape_used: bool
    estimated_work: int


def _fit_work(n: int, knots: int, max_iterations: int) -> int:
    reml_profile_budget = 61 + 59 * max_iterations
    return knots**3 + n**3 + n * knots**2 + knots * n**2 + n * reml_profile_budget


def _validate_tape(value: ArrayLike, replicates: int, observations: int) -> FloatArray:
    expected = (replicates, observations)
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.shape != expected or raw.dtype.kind not in "fiu":
        raise ValueError(f"multiplier_tape must be a real numeric array with shape {expected}")
    values = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("multiplier_tape must contain only finite values")
    nearest = np.argmin(np.abs(values[..., None] - _MAMMEN_VALUES), axis=-1)
    close = np.isclose(values, _MAMMEN_VALUES[nearest], rtol=0.0, atol=1e-14)
    if not np.all(close):
        raise ValueError("multiplier_tape values must be the two Mammen multipliers")
    return np.array(_MAMMEN_VALUES[nearest], copy=True)


def _freeze(value: ArrayLike, dtype: np.dtype | type = np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def bootstrap_synergy_surface(
    dose1: ArrayLike,
    dose2: ArrayLike,
    response: ArrayLike,
    *,
    baseline: Literal["raw", "log"] = "raw",
    replicates: int = 100,
    smoothing_parameter: float | None = None,
    rng: np.random.Generator | int | None = None,
    multiplier_tape: ArrayLike | None = None,
    store_draws: bool = True,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> SynergySurfaceBootstrap:
    """Run the documented two-stage Mammen wild bootstrap serially.

    The original fit selects REML lambda unless ``smoothing_parameter`` is
    supplied. Residuals use the same baseline with a spline fit at lambda/2;
    pseudo-data are centered on the fit at 2*lambda. Every pseudo-dataset then
    refits the baseline and selects REML lambda. ``multiplier_tape`` provides
    bounded deterministic replay with shape ``(replicates, observations)``.

    The returned draw mean and ordinary sample SD are descriptive summaries,
    not confidence limits. No normal interval is produced because the exact
    source SD centering and denominator remain unverified.
    """
    d1, d2, y_optional = _dose_vectors(dose1, dose2, response)
    assert y_optional is not None
    y = y_optional
    if baseline not in ("raw", "log"):
        raise ValueError("baseline must be 'raw' or 'log'")
    if isinstance(replicates, (bool, np.bool_)) or not isinstance(replicates, (int, np.integer)):
        raise ValueError("replicates must be an integer")
    count = int(replicates)
    if not 2 <= count <= _MAX_BOOTSTRAP_REPLICATES:
        raise ValueError("replicates must be in [2, 2000] to define a sample SD")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or not 20 <= int(max_iterations) <= 1000
    ):
        raise ValueError("max_iterations must be an integer in [20, 1000]")
    if not isinstance(store_draws, (bool, np.bool_)):
        raise ValueError("store_draws must be a boolean")
    if baseline == "log" and np.any((d1 == 0.0) & (d2 == 0.0)):
        raise ValueError("log-dose bootstrap rejects the both-zero dose: its baseline is undefined")
    if multiplier_tape is not None and rng is not None:
        raise ValueError("pass either multiplier_tape or rng, not both")

    n = int(d1.size)
    knot_count = int(np.unique(np.column_stack((d1, d2)), axis=0).shape[0])
    per_fit_work = _fit_work(n, knot_count, int(max_iterations))
    # One original REML fit, two fixed-lambda anchor fits, and all REML refits.
    estimated_work = (count + 1) * per_fit_work + 2 * (
        knot_count**3 + n**3 + n * knot_count**2 + knot_count * n**2 + n
    )
    if per_fit_work > _MAX_WORK:
        raise ValueError("one surface fit exceeds its supported work budget")
    if estimated_work > _MAX_BOOTSTRAP_WORK:
        raise ValueError("bootstrap request exceeds the supported total work budget")
    if store_draws and count * n > _MAX_BOOTSTRAP_STORAGE_CELLS:
        raise ValueError("stored bootstrap draws exceed the supported cell budget")
    tape: FloatArray | None = None
    if multiplier_tape is not None:
        if count * n > _MAX_BOOTSTRAP_STORAGE_CELLS:
            raise ValueError("multiplier tape exceeds the supported cell budget")
        tape = _validate_tape(multiplier_tape, count, n)

    original = fit_synergy_surface(
        d1,
        d2,
        y,
        baseline=baseline,
        smoothing_parameter=smoothing_parameter,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if not original.optimizer_success:
        raise ValueError("original REML smoothing fit did not converge")
    lam = original.smoothing_parameter
    lower = fit_synergy_surface(
        d1,
        d2,
        y,
        baseline=baseline,
        smoothing_parameter=lam / 2.0,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    upper = fit_synergy_surface(
        d1,
        d2,
        y,
        baseline=baseline,
        smoothing_parameter=2.0 * lam,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        residual = y - lower.fitted_baseline - lower.fitted_surface
        center = upper.fitted_baseline + upper.fitted_surface
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(center)):
        raise ArithmeticError("bootstrap anchor fit is not finite")

    generator = None if tape is not None else np.random.default_rng(rng)
    draws: FloatArray | None = np.empty((count, n), dtype=np.float64) if store_draws else None
    mean_scaled: FloatArray = np.zeros(n, dtype=np.float64)
    sum_squared_scaled: FloatArray = np.zeros(n, dtype=np.float64)
    draw_scale: FloatArray = np.zeros(n, dtype=np.float64)
    smoothing: FloatArray = np.empty(count, dtype=np.float64)
    objectives: FloatArray = np.empty(count, dtype=np.float64)
    variances: FloatArray = np.empty(count, dtype=np.float64)
    response_scales: FloatArray = np.empty(count, dtype=np.float64)
    evaluations: NDArray[np.int64] = np.empty(count, dtype=np.int64)
    boundaries: NDArray[np.bool_] = np.empty(count, dtype=np.bool_)

    for index in range(count):
        if tape is None:
            assert generator is not None
            weights = generator.choice(_MAMMEN_VALUES, size=n, p=_MAMMEN_PROBABILITIES)
        else:
            weights = tape[index]
        with np.errstate(over="ignore", invalid="ignore"):
            synthetic = center + residual * weights
        if not np.all(np.isfinite(synthetic)):
            raise ArithmeticError(f"bootstrap replicate {index} produced a nonfinite response")
        try:
            replicate_fit = fit_synergy_surface(
                d1,
                d2,
                synthetic,
                baseline=baseline,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"bootstrap replicate {index} failed: {exc}") from exc
        if not replicate_fit.optimizer_success:
            raise ValueError(f"bootstrap replicate {index} REML smoothing fit did not converge")
        draw = replicate_fit.fitted_surface
        new_scale = np.maximum(draw_scale, np.abs(draw))
        rescale = np.divide(draw_scale, new_scale, out=np.zeros(n), where=new_scale > 0.0)
        mean_scaled *= rescale
        sum_squared_scaled *= np.square(rescale)
        draw_scale = new_scale
        scaled_draw = np.divide(draw, draw_scale, out=np.zeros(n), where=draw_scale > 0.0)
        delta = scaled_draw - mean_scaled
        mean_scaled += delta / (index + 1)
        sum_squared_scaled += delta * (scaled_draw - mean_scaled)
        if draws is not None:
            draws[index] = draw
        smoothing[index] = replicate_fit.smoothing_parameter
        objectives[index] = replicate_fit.reml_objective
        variances[index] = replicate_fit.residual_variance_scaled
        response_scales[index] = replicate_fit.response_scale
        evaluations[index] = replicate_fit.optimizer_evaluations
        boundaries[index] = replicate_fit.smoothing_at_boundary

    roundoff = 128.0 * np.finfo(float).eps
    if np.any(sum_squared_scaled < -roundoff):
        raise ArithmeticError("bootstrap draw variance is negative beyond roundoff")
    sum_squared_scaled = np.maximum(sum_squared_scaled, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = draw_scale * mean_scaled
        standard_deviation = (
            draw_scale * np.sqrt(sum_squared_scaled / (count - 1))
            if count > 1
            else np.full(n, np.nan)
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(standard_deviation)):
        raise ArithmeticError("bootstrap draw summaries are outside floating-point range")
    return SynergySurfaceBootstrap(
        original,
        _freeze(original.fitted_surface),
        _freeze(mean),
        _freeze(standard_deviation),
        None if draws is None else _freeze(draws),
        _freeze(smoothing),
        _freeze(objectives),
        _freeze(variances),
        _freeze(response_scales),
        _freeze(evaluations, np.int64),
        _freeze(boundaries, np.bool_),
        lower,
        upper,
        count,
        tape is not None,
        estimated_work,
    )
