"""Joint dose and allocation optimization for SINGLE response designs."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from ._validation import FloatArray, finite, scalar
from .single import _response_information
from .single_uniform import _local_criterion

DesignVectors = FloatArray | tuple[FloatArray, FloatArray]


@dataclass(frozen=True)
class SingleOptimizedDesign:
    doses: DesignVectors
    subjects: DesignVectors
    value: float
    initial_value: float
    stationarity: float
    iterations: int

    def report(self, *, digits: int = 8) -> str:
        """Return a TSV design and numerical summary with significant-digit formatting.

        Criterion values retain the objective chosen by the caller (SD/variance,
        arithmetic/harmonic). Zero allocations and original entry ordering are
        preserved. This report does not serialize the model or prior configuration.
        """
        if isinstance(digits, (bool, np.bool_)) or not isinstance(digits, (int, np.integer)):
            raise ValueError("digits must be an integer from 1 to 17")
        if not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 to 17")
        doses = self.doses if isinstance(self.doses, tuple) else (self.doses,)
        subjects = self.subjects if isinstance(self.subjects, tuple) else (self.subjects,)
        rows = ["SINGLE optimized design", "Group\tEntry\tDose\tSubjects"]
        for group, (x, n) in enumerate(zip(doses, subjects, strict=True), start=1):
            for entry, (dose, count) in enumerate(zip(x, n, strict=True), start=1):
                rows.append(f"{group}\t{entry}\t{dose:.{digits}g}\t{count:.{digits}g}")
        rows.extend(["", "Group\tTotal subjects"])
        for group, n in enumerate(subjects, start=1):
            rows.append(f"{group}\t{np.sum(n):.{digits}g}")
        rows.extend(
            [
                f"Overall total subjects\t{sum(float(np.sum(n)) for n in subjects):.{digits}g}",
                "",
                "Metric\tValue",
                f"Initial criterion\t{self.initial_value:.{digits}g}",
                f"Final criterion\t{self.value:.{digits}g}",
                f"Stationarity diagnostic\t{self.stationarity:.{digits}g}",
                f"Iterations\t{self.iterations}",
            ]
        )
        return "\n".join(rows) + "\n"

    def write_report(self, path: str | Path, *, digits: int = 8) -> Path:
        """Write the UTF-8 report, replacing the explicit destination if it exists."""
        destination = Path(path)
        destination.write_text(self.report(digits=digits), encoding="utf-8")
        return destination


def single_optimize_design(
    initial_doses: ArrayLike | tuple[ArrayLike, ArrayLike],
    parameters: ArrayLike,
    dose_bounds: ArrayLike,
    *,
    prior_weights: ArrayLike | None = None,
    total_subjects: float = 100,
    initial_subjects: ArrayLike | tuple[ArrayLike, ArrayLike] | None = None,
    criterion: str = "quantile",
    comparison: str | None = None,
    model: str = "logistic",
    form: str = "linear",
    measure: str = "sd",
    aggregation: str = "arithmetic",
    quantile: float = 0.05,
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
) -> SingleOptimizedDesign:
    """Locally optimize doses and continuous counts within shared dose bounds.

    Supply a point parameter vector or weighted parameter nodes. One sample
    uses a dose vector; two samples use a pair of vectors and comparison=location
    or slope. Total subjects is shared by both groups. The number of supplied
    dose entries stays fixed; zero allocations are allowed. Try multiple starts
    to assess local minima. Singular trial information is infeasible.
    """
    bounds = finite(dose_bounds, "dose_bounds")
    if bounds.shape != (2,) or bounds[0] >= bounds[1]:
        raise ValueError("dose_bounds must contain two finite increasing values")
    width = bounds[1] - bounds[0]
    if not np.isfinite(width):
        raise ValueError("Dose interval width overflowed")
    total, tol = scalar(total_subjects, "total_subjects"), scalar(tolerance, "tolerance")
    if total <= 0 or not 0 < tol < 1:
        raise ValueError("Require positive total_subjects and tolerance in (0,1)")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    if (
        criterion not in ("slope", "quantile")
        or measure not in ("sd", "variance")
        or aggregation not in ("arithmetic", "harmonic")
    ):
        raise ValueError("Invalid criterion, measure or aggregation")
    q = scalar(quantile, "quantile")
    dimension = 2 if comparison is None else 3
    b = finite(parameters, "parameters")
    if b.shape == (dimension,):
        b = b[None, :]
    if b.ndim != 2 or b.shape[1] != dimension or b.shape[0] == 0:
        raise ValueError("Parameter dimensions do not match the number of samples")
    if prior_weights is None:
        if b.shape[0] != 1:
            raise ValueError("prior_weights are required for multiple parameter nodes")
        prior = np.ones(1)
    else:
        prior = finite(prior_weights, "prior_weights")
    if (
        prior.shape != (b.shape[0],)
        or np.any(prior <= 0)
        or not np.isclose(prior.sum(), 1, rtol=1e-12, atol=0)
    ):
        raise ValueError("Positive prior_weights must match nodes and sum to one")
    prior = prior / prior.sum()
    groups: tuple[FloatArray, ...]
    if comparison is None:
        groups = (finite(np.asarray(initial_doses), "initial_doses"),)
    else:
        if not isinstance(initial_doses, (tuple, list, np.ndarray)) or len(initial_doses) != 2:
            raise ValueError("Require two initial dose vectors")
        groups = tuple(finite(v, "initial_doses") for v in initial_doses)
    if any(g.ndim != 1 or g.size == 0 for g in groups):
        raise ValueError("Initial dose vectors must be nonempty")
    lengths = [g.size for g in groups]
    x = np.concatenate(groups)
    size = x.size
    if np.any((x < bounds[0]) | (x > bounds[1])):
        raise ValueError("Initial doses must lie within dose_bounds")

    def unpack(v: FloatArray) -> DesignVectors:
        return v if comparison is None else (v[: lengths[0]], v[lengths[0] :])

    if initial_subjects is None:
        n = np.full(size, total / size)
    elif comparison is None:
        n = finite(np.asarray(initial_subjects), "initial_subjects")
    else:
        if (
            not isinstance(initial_subjects, (tuple, list, np.ndarray))
            or len(initial_subjects) != 2
        ):
            raise ValueError("Require two initial subject vectors")
        counts = [finite(v, "initial_subjects") for v in initial_subjects]
        if any(c.shape != g.shape for c, g in zip(counts, groups, strict=True)):
            raise ValueError("Initial subject vectors must match group doses")
        n = np.concatenate(counts)
    if n.shape != x.shape or np.any(n < 0) or not np.isclose(n.sum(), total, rtol=1e-12, atol=0):
        raise ValueError("Initial subjects must match doses, be nonnegative and sum to total")
    _local_criterion(
        unpack(x),
        unpack(n),
        b,
        f"{criterion}_{measure}" if comparison is None else measure,
        model,
        form,
        comparison,
        q,
    )
    compared = 0 if (form == "linear") == (comparison == "location") else 1
    if comparison is not None:
        target = np.zeros_like(b)
        target[:, compared], target[:, 2] = 1, -1
    elif criterion == "slope":
        target = np.broadcast_to([0.0, 1.0] if form == "linear" else [1.0, 0.0], b.shape)
    else:
        link = np.log(q) - np.log1p(-q) if model == "logistic" else -np.log(-np.log(q))
        target = (
            np.stack([-1 / b[:, 1], -(link - b[:, 0]) / b[:, 1] ** 2], axis=-1)
            if form == "linear"
            else np.stack([-link / b[:, 0] ** 2, np.ones(b.shape[0])], axis=-1)
        )
    power = 0.5 if measure == "sd" else 1.0

    def evaluate(z: FloatArray) -> tuple[float, FloatArray]:
        doses = bounds[0] + width * z[:size]
        fractions = z[size:]
        gradients, dose_gradients, weights, weight_derivatives = [], [], [], []
        start = 0
        for group, length in enumerate(lengths):
            points = doses[start : start + length]
            start += length
            indices = [0, 1]
            if group == 1:
                indices[compared] = 2
            first, second = b[:, indices[0], None], b[:, indices[1], None]
            if form == "linear":
                u = first + second * points
                derivatives = np.broadcast_arrays(np.ones_like(u), points)
                slope, changed = second, indices[1]
            else:
                u = first * (points - second)
                derivatives = np.broadcast_arrays(points - second, -first)
                slope, changed = first, indices[0]
            if not np.all(np.isfinite(u)):
                return np.inf, np.zeros(2 * size)
            g = np.zeros(u.shape + (dimension,))
            dg = np.zeros_like(g)
            dg[..., changed] = 1
            for index, derivative in zip(indices, derivatives, strict=True):
                g[..., index] = derivative
            p, w = _response_information(u, model)
            if model == "logistic":
                dw = w * (1 - 2 * p)
            else:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                    t = np.exp(-u)
                    ratio = np.ones_like(t)
                    np.divide(t, -np.expm1(-t), out=ratio, where=(t > 0) & np.isfinite(t))
                    dw = w * (ratio - 2)
            gradients.append(g)
            dose_gradients.append(dg)
            weights.append(w)
            weight_derivatives.append(dw * slope)
        gradient, dg = np.concatenate(gradients, axis=1), np.concatenate(dose_gradients, axis=1)
        w, dw = np.concatenate(weights, axis=1), np.concatenate(weight_derivatives, axis=1)
        effective = fractions * w
        if np.any(
            np.linalg.matrix_rank(np.where(effective[..., None] > 0, gradient, 0)) < dimension
        ):
            return np.inf, np.zeros(2 * size)
        information = np.swapaxes(gradient, -1, -2) @ (effective[..., None] * gradient)
        try:
            factor = np.linalg.cholesky(information)
            y = np.linalg.solve(factor, target[..., None])
            solution = np.linalg.solve(np.swapaxes(factor, -1, -2), y)
        except np.linalg.LinAlgError:
            return np.inf, np.zeros(2 * size)
        variance = np.sum(y[..., 0] ** 2, axis=-1)
        projection, dp = (gradient @ solution)[..., 0], (dg @ solution)[..., 0]
        df = -w * projection**2
        dx = -fractions * (dw * projection**2 + 2 * w * projection * dp) * width
        local = variance**power
        derivative = (power * variance ** (power - 1))[:, None] * np.concatenate([dx, df], axis=1)
        if aggregation == "arithmetic":
            return float(prior @ local), prior @ derivative
        minimum = np.min(local)
        value = float(minimum / (prior @ (minimum / local)))
        return value, (prior * (value / local) ** 2) @ derivative

    start = np.concatenate([(x - bounds[0]) / width, n / total])
    baseline, _ = evaluate(start)
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError("Initial design has invalid precision")

    def objective(z: FloatArray) -> tuple[float, FloatArray]:
        value, derivative = evaluate(z)
        return value / baseline, derivative / baseline

    result = minimize(
        objective,
        start,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * (2 * size),
        constraints={
            "type": "eq",
            "fun": lambda z: np.sum(z[size:]) - 1,
            "jac": lambda z: np.concatenate([np.zeros(size), np.ones(size)]),
        },
        options={"maxiter": int(max_iterations), "ftol": tol},
    )
    if not result.success:
        raise RuntimeError(f"Joint design optimization failed: {result.message}")
    z = result.x
    if np.any((z < 0) | (z > 1)) or abs(z[size:].sum() - 1) > 1e-8:
        raise RuntimeError("Joint optimizer returned an infeasible design")
    z[size:] /= z[size:].sum()
    value, derivative = evaluate(z)
    if not np.isfinite(value) or value <= 0 or not np.all(np.isfinite(derivative)):
        raise RuntimeError("Joint optimizer returned invalid precision or derivatives")
    dose_residual = derivative[:size].copy()
    dose_residual[z[:size] <= 1e-8] = np.minimum(dose_residual[z[:size] <= 1e-8], 0)
    dose_residual[z[:size] >= 1 - 1e-8] = np.maximum(dose_residual[z[:size] >= 1 - 1e-8], 0)
    stationarity = float(
        max(
            np.max(np.abs(dose_residual)) / value,
            np.max(-derivative[size:]) / (power * value) - 1,
            0,
        )
    )
    return SingleOptimizedDesign(
        unpack(bounds[0] + width * z[:size]),
        unpack(total * z[size:]),
        value / total**power,
        baseline / total**power,
        stationarity,
        int(result.nit),
    )
