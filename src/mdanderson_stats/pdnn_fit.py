"""Analytic-gradient fitting of the PDNN log-intensity objective."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import expit

from ._validation import FloatArray, count, finite, scalar
from .bayesian_monitoring import _integer
from .beta_binomial import _owned
from .pdnn import _pdnn_pairs


@dataclass(frozen=True)
class PDNNParameters:
    specific_stacking: FloatArray
    nonspecific_stacking: FloatArray
    specific_weights: FloatArray
    nonspecific_weights: FloatArray
    log_expression: FloatArray
    log_nonspecific_amount: float
    log_background: float

    def __post_init__(self) -> None:
        for name, shape in [
            ("specific_stacking", (4, 4)),
            ("nonspecific_stacking", (4, 4)),
            ("specific_weights", (24,)),
            ("nonspecific_weights", (24,)),
        ]:
            value = finite(getattr(self, name), name)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            object.__setattr__(self, name, _owned(value))
        expression = finite(self.log_expression, "log_expression")
        if expression.ndim != 1 or expression.size == 0:
            raise ValueError("log_expression must be a nonempty vector in sorted probeset order")
        object.__setattr__(self, "log_expression", _owned(expression))
        for name in ["log_nonspecific_amount", "log_background"]:
            object.__setattr__(self, name, scalar(getattr(self, name), name))

    def _pack(self) -> FloatArray:
        return np.concatenate(
            [
                self.specific_stacking.ravel(),
                self.nonspecific_stacking.ravel(),
                self.specific_weights,
                self.nonspecific_weights,
                self.log_expression,
                [self.log_nonspecific_amount, self.log_background],
            ]
        )


def _unpack(vector: FloatArray) -> PDNNParameters:
    return PDNNParameters(
        vector[:16].reshape(4, 4),
        vector[16:32].reshape(4, 4),
        vector[32:56],
        vector[56:80],
        vector[80:-2],
        float(vector[-2]),
        float(vector[-1]),
    )


def _objective(
    vector: FloatArray,
    pairs: NDArray[np.int64],
    groups: NDArray[np.int64],
    log_observed: FloatArray,
) -> tuple[float, FloatArray, FloatArray]:
    e = np.sum(vector[:16][pairs] * vector[32:56], axis=1)
    ns = np.sum(vector[16:32][pairs] * vector[56:80], axis=1)
    specific = vector[80:-2][groups] - np.logaddexp(0, e)
    nonspecific = vector[-2] - np.logaddexp(0, ns)
    fitted = np.logaddexp(np.logaddexp(specific, nonspecific), vector[-1])
    residual = fitted - log_observed
    value = float(np.mean(residual**2))
    scale = 2 * residual / len(residual)
    responsibility_specific = np.exp(specific - fitted)
    responsibility_ns = np.exp(nonspecific - fitted)
    gradient = np.empty_like(vector)
    gradient[80:-2] = np.bincount(
        groups, weights=scale * responsibility_specific, minlength=len(vector) - 82
    )
    gradient[-2] = np.sum(scale * responsibility_ns)
    gradient[-1] = np.sum(scale * np.exp(vector[-1] - fitted))
    for energies, weights, energy_slice, weight_slice in [
        (e, responsibility_specific, slice(0, 16), slice(32, 56)),
        (ns, responsibility_ns, slice(16, 32), slice(56, 80)),
    ]:
        energy_derivative = -scale * weights * expit(energies)
        gradient[energy_slice] = np.bincount(
            pairs.ravel(),
            weights=(energy_derivative[:, None] * vector[weight_slice]).ravel(),
            minlength=16,
        )
        gradient[weight_slice] = np.sum(
            energy_derivative[:, None] * vector[energy_slice][pairs], axis=0
        )
    if not np.isfinite(value) or np.any(~np.isfinite(gradient)) or np.any(~np.isfinite(fitted)):
        raise ArithmeticError("PDNN objective or gradient cannot be represented")
    return value, gradient, fitted


@dataclass(frozen=True)
class PDNNFit:
    parameters: PDNNParameters
    probeset_ids: NDArray[np.int64]
    fitted_signal: FloatArray
    initial_fitness: float
    fitness: float
    iterations: int
    gradient_max: float
    converged: bool
    message: str


class PDNNConvergenceError(ArithmeticError):
    """A local optimizer failed to converge; result retains its last valid fit."""

    def __init__(self, result: PDNNFit):
        self.result = result
        super().__init__(f"PDNN optimization did not converge: {result.message}")


def fit_pdnn(
    sequences: ArrayLike,
    intensities: ArrayLike,
    probeset_ids: ArrayLike,
    *,
    initial: PDNNParameters,
    fit_energies: bool = True,
    fit_weights: bool = True,
    max_iterations: int = 2000,
    tolerance: float = 1e-8,
) -> PDNNFit:
    """Fit paper equation (4), mean squared log-intensity error, to one array.

    Fits log expression plus global log nonspecific amount/background. With
    fit_energies, first fit stacking energies while holding position weights fixed;
    with fit_weights, follow with a joint stage. This uses deterministic L-BFGS-B,
    not the native Monte Carlo optimizer. A local solution is not necessarily global
    or uniquely parameterized. The initial log-expression vector follows sorted IDs.
    """
    if not isinstance(initial, PDNNParameters):
        raise TypeError("initial must be PDNNParameters")
    if (
        not isinstance(fit_energies, bool)
        or not isinstance(fit_weights, bool)
        or (fit_weights and not fit_energies)
    ):
        raise ValueError("fit switches must be boolean; fitting weights requires fitting energies")
    iterations = _integer(max_iterations, "max_iterations")
    tol = scalar(tolerance, "tolerance")
    if not 1 <= iterations <= 100000 or not 1e-12 <= tol <= 1e-3:
        raise ValueError("max_iterations must be 1..100000 and tolerance 1e-12..1e-3")
    pairs = _pdnn_pairs(sequences)
    x, ids = finite(intensities, "intensities"), count(probeset_ids, "probeset_ids")
    if x.shape != (len(pairs),) or ids.shape != x.shape or np.any(x <= 0):
        raise ValueError("require one positive intensity and probeset ID per probe")
    labels, groups = np.unique(ids.astype(np.int64), return_inverse=True)
    if initial.log_expression.shape != labels.shape:
        raise ValueError("initial expression must contain one value per unique probeset")
    current = initial._pack()
    log_x = np.log(x)
    initial_fitness = _objective(current, pairs, groups, log_x)[0]
    active = np.r_[
        np.arange(32) if fit_energies else np.array([], dtype=int), np.arange(80, len(current))
    ]
    stages = [active]
    if fit_weights:
        stages.append(np.arange(len(current)))
    total_iterations = 0
    for active in stages:
        fixed = current.copy()

        def objective(values: FloatArray) -> tuple[float, FloatArray]:
            vector = fixed.copy()
            vector[active] = values
            value, gradient, _ = _objective(vector, pairs, groups, log_x)
            return value, gradient[active]

        solution = minimize(
            objective,
            current[active],
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": iterations, "ftol": tol * 1e-3, "gtol": tol, "maxls": 40},
        )
        current[active] = solution.x
        total_iterations += int(solution.nit)
        value, gradient, log_fitted = _objective(current, pairs, groups, log_x)
        with np.errstate(over="ignore", under="ignore"):
            fitted = np.exp(log_fitted)
        if np.any(~np.isfinite(fitted)) or np.any(fitted <= 0) or value > initial_fitness + 1e-10:
            raise ArithmeticError(
                "PDNN optimization returned invalid signals or worsened the objective"
            )
        labels.flags.writeable = False
        result = PDNNFit(
            _unpack(current),
            labels,
            _owned(fitted),
            initial_fitness,
            value,
            total_iterations,
            float(np.max(np.abs(gradient[active]))),
            bool(solution.success),
            str(solution.message),
        )
        if not solution.success:
            raise PDNNConvergenceError(result)
    return result
