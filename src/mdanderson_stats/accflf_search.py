"""Profile shape searches and model scans for ACCFLF."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar
from .accflf import accflf_shape
from .accflf_model import AccflfFit, _data, fit_accflf


def _check_data(
    time: ArrayLike, event: ArrayLike, covariates: ArrayLike | None, weights: ArrayLike | None
) -> int:
    y, e, x, _ = _data(time, event, covariates, weights)
    if not np.any(e == 1) or np.std(y) == 0 or np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("shape search requires failures, varying times and full-rank design")
    return y.size


@dataclass(frozen=True)
class AccflfGrid:
    p: FloatArray
    q: FloatArray
    log_likelihood: FloatArray
    fits: tuple[AccflfFit | None, ...]
    errors: tuple[str | None, ...]

    @property
    def best(self) -> AccflfFit | None:
        """Best successful grid point; None if every fit failed."""
        if not np.any(np.isfinite(self.log_likelihood)):
            return None
        return self.fits[int(np.nanargmax(self.log_likelihood))]


def scan_accflf(
    time: ArrayLike,
    event: ArrayLike,
    *,
    p: ArrayLike,
    q: ArrayLike,
    covariates: ArrayLike | None = None,
    weights: ArrayLike | None = None,
) -> AccflfGrid:
    """Refit sigma/intercept/covariates at every rectangular p,q grid point.

    Likelihood matrix has p rows and q columns. fits/errors follow row-major
    order. Failed fits remain NaN/None with their numerical error recorded.
    """
    n = _check_data(time, event, covariates, weights)
    if np.iscomplexobj(p) or np.iscomplexobj(q):
        raise ValueError("p and q grids must be real")
    pp, qq = finite(p, "p"), finite(q, "q")
    if (
        pp.ndim != 1
        or qq.ndim != 1
        or not 1 <= pp.size <= 100
        or not 1 <= qq.size <= 100
        or pp.size * qq.size * n > 20_000_000
    ):
        raise ValueError("require nonempty grids <=100 per axis and <=20 million row-grid pairs")
    for pv in pp:
        for qv in qq:
            accflf_shape(float(pv), float(qv))
    likelihood = np.full((pp.size, qq.size), np.nan)
    fits: list[AccflfFit | None] = []
    errors: list[str | None] = []
    for i, pv in enumerate(pp):
        for j, qv in enumerate(qq):
            try:
                fit = fit_accflf(
                    time, event, p=float(pv), q=float(qv), covariates=covariates, weights=weights
                )
            except ArithmeticError as exc:
                fits.append(None)
                errors.append(str(exc))
            else:
                likelihood[i, j] = fit.log_likelihood
                fits.append(fit)
                errors.append(None)
    return AccflfGrid(_freeze(pp), _freeze(qq), _freeze(likelihood), tuple(fits), tuple(errors))


@dataclass(frozen=True)
class AccflfSearchRun:
    start_p: float
    start_q: float
    fit: AccflfFit | None
    converged: bool
    evaluations: int
    message: str


@dataclass(frozen=True)
class AccflfShapeSearch:
    best: AccflfFit
    runs: tuple[AccflfSearchRun, ...]
    evaluated_shapes: int
    failed_shapes: tuple[tuple[float, float, str], ...]
    fixed_p: float | None

    @property
    def converged(self) -> bool:
        """Whether a run returning the selected best fit met simplex tolerances.

        This is local optimizer termination, not a global optimality certificate.
        """
        return any(run.fit is self.best and run.converged for run in self.runs)


def search_accflf(
    time: ArrayLike,
    event: ArrayLike,
    *,
    fixed_p: float | None = None,
    starts: ArrayLike | None = None,
    covariates: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    max_evaluations: int = 500,
    tolerance: float = 1e-6,
) -> AccflfShapeSearch:
    """Profile nuisance parameters while searching p,q, or q with p fixed.

    fixed_p=0 is the generalized-gamma boundary. Multiple local Nelder-Mead
    starts operate in log1p(p), signed-log1p(q) coordinates, within native
    search bounds. Every evaluation refits the conditional AFT model.
    best.covariance remains conditional on the selected shapes; it is not joint
    shape/coefficient uncertainty. Inspect converged, runs and failed_shapes.
    """
    n = _check_data(time, event, covariates, weights)
    if fixed_p is not None:
        fixed_p = scalar(fixed_p, "fixed_p")
        if not 0 <= fixed_p <= 1e10:
            raise ValueError("fixed_p must lie in [0,1e10]")
    if (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, int)
        or not 10 <= max_evaluations <= 5000
    ):
        raise ValueError("max_evaluations must be an integer in [10,5000] per start")
    tolerance = scalar(tolerance, "tolerance")
    if not 1e-10 <= tolerance <= 1e-2:
        raise ValueError("tolerance must lie in [1e-10,1e-2]")
    if starts is None:
        starts = np.array(
            [[0.5, 0.5], [0.5, -0.5], [5, 0]]
            if fixed_p is None
            else [[fixed_p, 0.5], [fixed_p, -0.5]]
        )
    assert starts is not None
    if np.iscomplexobj(starts):
        raise ValueError("starts must be real p,q pairs")
    points = finite(starts, "starts")
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not 1 <= points.shape[0] <= 20
        or points.shape[0] * max_evaluations * n > 20_000_000
    ):
        raise ValueError("require 1..20 starts and <=20 million row-evaluation pairs")
    if (
        np.any(points[:, 0] < (1e-10 if fixed_p is None else 0))
        or np.any(points[:, 0] > 1e10)
        or np.any(np.abs(points[:, 1]) > 1e10)
        or (fixed_p is not None and np.any(points[:, 0] != fixed_p))
    ):
        raise ValueError("starts must respect fixed_p and the native p,q search bounds")
    cache: dict[tuple[float, float], AccflfFit | None] = {}
    failures: list[tuple[float, float, str]] = []

    def decode(v: FloatArray) -> tuple[float, float]:
        p = float(np.clip(np.expm1(v[0]), 1e-10, 1e10)) if fixed_p is None else fixed_p
        q = float(np.clip(np.sign(v[-1]) * np.expm1(abs(v[-1])), -1e10, 1e10))
        return p, q

    def objective(v: FloatArray) -> float:
        p, q = decode(v)
        key = (p, q)
        if key not in cache:
            shape = accflf_shape(p, q)
            if shape.lower_clipped and shape.numerator_df == shape.denominator_df == 0.001:
                cache[key] = None
                failures.append((p, q, "both degrees of freedom clipped below native bound"))
            else:
                try:
                    cache[key] = fit_accflf(
                        time, event, p=p, q=q, covariates=covariates, weights=weights
                    )
                except ArithmeticError as exc:
                    cache[key] = None
                    failures.append((p, q, str(exc)))
        fit = cache[key]
        return np.inf if fit is None else -fit.log_likelihood

    bound = float(np.log1p(1e10))
    bounds = (
        [(float(np.log1p(1e-10)), bound), (-bound, bound)] if fixed_p is None else [(-bound, bound)]
    )
    runs = []
    for p, q in points:
        start = np.array([np.log1p(p), np.sign(q) * np.log1p(abs(q))])
        if fixed_p is not None:
            start = start[1:]
        # Give zero coordinates a substantive initial step as well.
        steps = np.where(start + 0.1 <= np.array(bounds)[:, 1], 0.1, -0.1)
        simplex = np.vstack((start, start + np.diag(steps)))
        result = minimize(
            objective,
            start,
            method="Nelder-Mead",
            bounds=bounds,
            options={
                "maxfev": max_evaluations,
                "xatol": tolerance,
                "fatol": tolerance,
                "initial_simplex": simplex,
            },
        )
        key = decode(result.x)
        fit = cache.get(key)
        runs.append(
            AccflfSearchRun(
                float(p),
                float(q),
                fit,
                bool(result.success) and fit is not None,
                int(result.nfev),
                str(result.message),
            )
        )
    candidates = [run.fit for run in runs if run.fit is not None]
    if not candidates:
        raise ArithmeticError("every shape-search run failed; try other starts or inspect a grid")
    best = max(candidates, key=lambda fit: fit.log_likelihood)
    return AccflfShapeSearch(best, tuple(runs), len(cache), tuple(failures), fixed_p)


@dataclass(frozen=True)
class AccflfModelResult:
    name: str
    fit: AccflfFit | None
    search: AccflfShapeSearch | None
    error: str | None


def compare_accflf(
    time: ArrayLike,
    event: ArrayLike,
    *,
    covariates: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    max_evaluations: int = 500,
    tolerance: float = 1e-6,
) -> tuple[AccflfModelResult, ...]:
    """Fit the six named models in the source's fit_all_models order.

    Generalized F/gamma retain full shape-search diagnostics. Numerical failures
    remain explicit model records; a comparison does not certify global maxima.
    """
    _check_data(time, event, covariates, weights)
    models: list[AccflfModelResult] = []
    for name, p in (("generalized_f", None), ("generalized_gamma", 0.0)):
        try:
            search = search_accflf(
                time,
                event,
                fixed_p=p,
                covariates=covariates,
                weights=weights,
                max_evaluations=max_evaluations,
                tolerance=tolerance,
            )
        except ArithmeticError as exc:
            models.append(AccflfModelResult(name, None, None, str(exc)))
        else:
            models.append(AccflfModelResult(name, search.best, search, None))
    for name, p, q, fixed in (
        ("weibull", 0, 1, None),
        ("exponential", 0, 1, 1),
        ("lognormal", 0, 0, None),
        ("loglogistic", 1, 0, None),
    ):
        try:
            fit = fit_accflf(
                time, event, p=p, q=q, fixed_sigma=fixed, covariates=covariates, weights=weights
            )
        except ArithmeticError as exc:
            models.append(AccflfModelResult(name, None, None, str(exc)))
        else:
            models.append(AccflfModelResult(name, fit, None, None))
    return tuple(models)
