"""Direct constrained likelihood fitting for MULTI's MLBETA model.

Uses analytic derivatives and L-BFGS-B instead of the archived David Gay solver.
The original model and parameter bounds are retained; see the MULTI notice.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.special import betaln, digamma

from ._validation import FloatArray, scalar
from .beta_mixture import BetaMixture, _logs
from .beta_mixture_fit import BetaMixtureFit, BetaMixtureFitError, beta_mixture_start
from .multiplicity import _pvalues


def _coordinates(model: BetaMixture) -> FloatArray:
    remaining = model.null_weight + np.cumsum(model.weights[::-1])[::-1]
    sticks = np.zeros_like(remaining)
    np.divide(model.weights, remaining, out=sticks, where=remaining > 0)
    return np.r_[np.clip(sticks, 0, 1), np.log(model.a), np.log(model.b)]


def _model(theta: FloatArray) -> BetaMixture:
    t, la, lb = np.split(theta, 3)
    remainder = np.r_[1.0, np.cumprod(1 - t)]
    return BetaMixture(float(remainder[-1]), remainder[:-1] * t, np.exp(la), np.exp(lb))


def _objective(theta: FloatArray, lx: FloatArray, ly: FloatArray) -> tuple[float, FloatArray]:
    """Average negative likelihood and its gradient in stick/log-shape coordinates."""
    t, la, lb = np.split(theta, 3)
    a, b = np.exp(la), np.exp(lb)
    component = lx[:, None] * (a - 1) + ly[:, None] * (b - 1) - betaln(a, b)
    with np.errstate(divide="ignore"):
        lt, lr = np.log(t), np.log1p(-t)
    prefix = np.r_[0.0, np.cumsum(lr[:-1])]
    tail = np.zeros((lx.size, t.size + 1))
    for j in range(t.size - 1, -1, -1):
        tail[:, j] = np.logaddexp(lt[j] + component[:, j], lr[j] + tail[:, j + 1])
    total = tail[:, 0]
    with np.errstate(over="ignore"):
        weight_derivative = np.exp(prefix + component - total[:, None]) - np.exp(
            prefix + tail[:, 1:] - total[:, None]
        )
    posterior = np.exp(prefix + lt + component - total[:, None])
    psi_total = digamma(a + b)
    da = posterior * (lx[:, None] - digamma(a) + psi_total) * a
    db = posterior * (ly[:, None] - digamma(b) + psi_total) * b
    gradient = -np.r_[weight_derivative.mean(axis=0), da.mean(axis=0), db.mean(axis=0)]
    objective = -float(total.mean())
    if not np.isfinite(objective) or not np.all(np.isfinite(gradient)):
        raise BetaMixtureFitError("Direct likelihood or gradient exceeds numerical range")
    return objective, gradient


def fit_beta_mixture_ml(
    pvalues: ArrayLike,
    initial: BetaMixture | None = None,
    *,
    tolerance: float = 1e-9,
    gradient_tolerance: float = 1e-6,
    max_iterations: int = 1000,
    max_evaluations: int = 15000,
    legacy_endpoints: bool = False,
) -> BetaMixtureFit:
    """Optimize MLBETA's likelihood with exact weight and shape constraints.

    Shape bounds are [1e-10,1e9], matching the original. Stick coordinates keep
    all weights nonnegative and normalized, including exact boundary weights;
    log coordinates improve shape scaling. L-BFGS-B replaces David Gay's solver,
    so tolerances, iteration paths, and selected local optima can differ.
    tolerance is relative change in average negative likelihood, and
    gradient_tolerance bounds the projected gradient in transformed coordinates.
    """
    x = _pvalues(pvalues)
    if x.ndim != 1:
        raise ValueError("Mixture fitting accepts one p-value vector")
    tolerance = scalar(tolerance, "tolerance")
    gradient_tolerance = scalar(gradient_tolerance, "gradient_tolerance")
    if not 0 < tolerance < 1 or not 0 < gradient_tolerance < 1:
        raise ValueError("Optimization tolerances must lie strictly between zero and one")
    for name, limit in (("max_iterations", max_iterations), ("max_evaluations", max_evaluations)):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"{name} must be a positive integer")
    model = beta_mixture_start(x) if initial is None else initial
    k = model.weights.size
    if x.size <= 3 * k + 1:
        raise BetaMixtureFitError("Mixture fitting requires n>3*k+1")
    if k == 0:
        return BetaMixtureFit(model, 0.0, float(model.cramer_von_mises(x)), 0, np.array([0.0]))
    if np.any((model.a < 1e-10) | (model.a > 1e9) | (model.b < 1e-10) | (model.b > 1e9)):
        raise ValueError("Initial beta shapes must lie in [1e-10,1e9]")
    if not legacy_endpoints and np.any((x == 0) | (x == 1)):
        raise ValueError("Likelihood fitting requires interior values unless legacy_endpoints=True")
    lx, ly = _logs(x, legacy_endpoints)
    theta = _coordinates(model)
    history = [-x.size * _objective(theta, lx, ly)[0]]

    def record(point: FloatArray) -> None:
        history.append(-x.size * _objective(point, lx, ly)[0])

    result = minimize(
        _objective,
        theta,
        args=(lx, ly),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, 1.0)] * k + [(float(np.log(1e-10)), float(np.log(1e9)))] * (2 * k),
        callback=record,
        options={
            "ftol": tolerance,
            "gtol": gradient_tolerance,
            "maxiter": max_iterations,
            "maxfun": max_evaluations,
            "maxls": 40,
        },
    )
    if not result.success:
        raise BetaMixtureFitError(f"Direct optimizer failed: {result.message}")
    model = _model(result.x)
    likelihood = float(model.log_likelihood(x, legacy_endpoints=legacy_endpoints))
    if not np.isfinite(likelihood) or likelihood < history[0] - 1e-8 * max(1, abs(history[0])):
        raise BetaMixtureFitError("Direct optimizer failed its likelihood check")
    history[-1] = likelihood
    return BetaMixtureFit(
        model, likelihood, float(model.cramer_von_mises(x)), int(result.nit), np.array(history)
    )
