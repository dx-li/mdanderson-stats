"""Fixed-coefficient WFMM covariance optimization and conditional widths.

The optimizer is independently implemented with analytic gradients. It does not
reproduce the original Brent trajectory or claim global likelihood optimality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq, minimize

from ._validation import FloatArray, scalar
from .wfmm_model import _component_array, _finite_real, _integer, _strata
from .wfmm_mom import wfmm_mom_initialization
from .wfmm_variance_init import _frozen, _frozen_bool, _frozen_int


@dataclass(frozen=True)
class WFMMConditionalVariance:
    """Conditional estimates, local optimality checks and proposal diagnostics.

    Widths vary one variance component with all others and beta fixed. They are
    not marginal/profile intervals or posterior uncertainty. ``artificial_lower``
    marks native negative-symmetry sentinels; ``frozen`` marks native MOM-floor
    components. Neither is silently a scientific confidence statement.
    """

    random_variance: FloatArray
    residual_variance: FloatArray
    fixed_coefficients: FloatArray
    log_likelihood: FloatArray
    lower: FloatArray
    upper: FloatArray
    proposal_sd: FloatArray
    frozen: np.ndarray
    artificial_lower: np.ndarray
    lower_truncated: np.ndarray
    optimizer_success: np.ndarray
    projected_gradient: FloatArray
    covariance_condition: FloatArray
    frozen_descent: np.ndarray
    evaluations: np.ndarray
    status: tuple[str, ...]
    messages: tuple[str, ...]
    profile: str
    positive_proposals: bool


def fit_wfmm_conditional_variances(
    coefficients: ArrayLike,
    fixed_design: ArrayLike,
    fixed_coefficients: ArrayLike,
    random_design: ArrayLike | None = None,
    *,
    random_variance: ArrayLike | None = None,
    residual_variance: ArrayLike,
    random_strata: ArrayLike | None = None,
    residual_strata: ArrayLike | None = None,
    profile: str = "stable",
    min_variance: float = 1e-6,
    proposal_multiplier: float = 1.5,
    max_iterations: int = 200,
) -> WFMMConditionalVariance:
    """Fit covariance components conditional on supplied fixed coefficients.

    Two bounded starts and analytic gradients reduce boundary-start artifacts;
    reported convergence requires a projected-gradient check in scaled variance
    coordinates. It is a local optimum check, not proof of a global maximum.
    Stable mode may leave the initial floor and truncates missing lower roots
    at the floor. Explicit legacy_3_1 freezes floor starts, emits zero bounds for
    those components and uses lower=-upper when no positive lower root exists.
    Native optimizer iteration/rounding parity is not claimed.
    """
    iterations = _integer(max_iterations, "max_iterations", 1, 1000)
    if profile not in ("stable", "legacy_3_1"):
        raise ValueError("profile must be stable or legacy_3_1")
    if isinstance(min_variance, (bool, np.bool_)) or isinstance(
        proposal_multiplier, (bool, np.bool_)
    ):
        raise ValueError("floor and proposal multiplier must be real scalars")
    floor = scalar(min_variance, "min_variance")
    multiplier = scalar(proposal_multiplier, "proposal_multiplier")
    if not 1e-100 <= floor <= 1e100 or not 0 < multiplier <= 1e100:
        raise ValueError("floor or proposal multiplier outside bounded numerical range")
    raw = np.asarray(coefficients)
    if raw.ndim == 2 and raw.shape[1] * (40 * iterations) * raw.shape[0] ** 3 > 250_000_000:
        raise ValueError("conditional covariance fit exceeds bounded work budget")
    # Reuse the public design/rank/work validation, including identifiability of
    # the complete trace system. Its one-step estimates are not used as starts.
    validation = wfmm_mom_initialization(
        coefficients,
        fixed_design,
        random_design,
        random_strata=random_strata,
        residual_strata=residual_strata,
        min_variance=floor,
        max_iterations=1,
    )
    y, x = _finite_real(coefficients, "coefficients"), _finite_real(fixed_design, "fixed_design")
    beta = _finite_real(fixed_coefficients, "fixed_coefficients")
    n, k = y.shape
    if beta.shape != (x.shape[1], k) or np.max(np.abs(beta)) > 1e100:
        raise ValueError("fixed_coefficients must be bounded P-by-K")
    z = np.empty((n, 0)) if random_design is None else _finite_real(random_design, "random_design")
    rg = _strata(random_strata, z.shape[1], "random_strata")
    sg = _strata(residual_strata, n, "residual_strata")
    h, c = validation.random_variance.shape[0], validation.residual_variance.shape[0]
    q = _component_array(
        random_variance, (h, k), "random_variance", positive=True, default_empty=h == 0
    )
    s = _component_array(residual_variance, (c, k), "residual_variance", positive=True)
    initial = np.vstack((q, s))
    if np.any(initial < floor) or np.max(initial) > 1e100:
        raise ValueError("starting variances must lie between min_variance and 1e100")
    components = [z[:, rg == group] @ z[:, rg == group].T for group in range(h)]
    components.extend(np.diag((sg == group).astype(float)) for group in range(c))
    matrices = np.array(components)
    # Each objective evaluation solves a dense covariance. Include root searches
    # and both starts in the conservative preflight; a runtime cap remains too.
    evaluation_limit = 40 * iterations + 160 * (h + c)
    if k * evaluation_limit * (n**3 + (h + c) * n * n) > 250_000_000:
        raise ValueError("conditional covariance fit exceeds bounded work budget")
    residual = y - x @ beta
    if not np.all(np.isfinite(residual)) or np.max(np.abs(residual)) > 1e100:
        raise ValueError("conditional residuals exceed bounded numerical range")
    variance = initial.copy()
    lower, upper = np.full_like(initial, np.nan), np.full_like(initial, np.nan)
    proposals = np.full_like(initial, np.nan)
    frozen = (initial <= floor) if profile == "legacy_3_1" else np.zeros_like(initial, dtype=bool)
    artificial, truncated = np.zeros_like(initial, dtype=bool), np.zeros_like(initial, dtype=bool)
    success, descent = np.zeros(k, dtype=bool), np.zeros_like(initial, dtype=bool)
    ll, gradients = np.full(k, np.nan), np.full(k, np.inf)
    conditions = np.full(k, np.inf)
    evaluations = np.zeros(k, dtype=np.int64)
    statuses, messages = [], []
    basis_scale = np.max(np.abs(matrices), axis=(1, 2))
    normalized_matrices = matrices / basis_scale[:, None, None]
    for column in range(k):
        response_scale = max(float(np.max(np.abs(residual[:, column]))), np.sqrt(floor))
        response = residual[:, column] / response_scale
        units = response_scale**2 / basis_scale
        if not np.all(np.isfinite(units)) or np.any(units <= 0):
            raise ArithmeticError("conditional variance scaling is not representable")
        start = initial[:, column] / units
        minimum = floor / units
        # Explicit finite search boundary, flagged if reached. Scaling keeps the
        # maximum well beyond two dispersed starts without log-floor sticking.
        maximum = np.maximum(1.0, start) * 1e6
        fixed = frozen[:, column]
        active = ~fixed
        count = 0

        def objective(t):
            nonlocal count
            count += 1
            if count > evaluation_limit:
                raise ArithmeticError("conditional covariance evaluation budget exhausted")
            covariance = np.einsum("i,ijk->jk", t, normalized_matrices)
            factor = cho_factor(covariance, lower=True, check_finite=False)
            inv_r = cho_solve(factor, response, check_finite=False)
            inverse = cho_solve(factor, np.eye(n), check_finite=False)
            value = 0.5 * (2 * np.log(np.diag(factor[0])).sum() + response @ inv_r)
            gradient = 0.5 * (
                np.einsum("ij,kji->k", inverse, normalized_matrices)
                - np.einsum("i,kij,j->k", inv_r, normalized_matrices, inv_r)
            )
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                raise ArithmeticError("conditional objective/gradient is nonfinite")
            return float(value), gradient

        def expand(a):
            t = start.copy()
            t[active] = a
            return t

        def active_objective(a):
            value, gradient = objective(expand(a))
            return value, gradient[active]

        candidates = []
        if np.any(active):
            for candidate in (
                start[active],
                np.maximum(minimum[active], np.full(np.count_nonzero(active), 1 / (h + c))),
            ):
                fit = minimize(
                    active_objective,
                    candidate,
                    jac=True,
                    method="L-BFGS-B",
                    bounds=list(zip(minimum[active], maximum[active], strict=True)),
                    options=dict(
                        maxiter=iterations, maxfun=20 * iterations, ftol=1e-14, gtol=1e-8, maxls=30
                    ),
                )
                candidates.append(
                    (float(fit.fun), expand(fit.x), bool(fit.success), str(fit.message))
                )
            _, fitted, optimizer_ok, message = min(candidates, key=lambda row: row[0])
        else:
            fitted = start.copy()
            optimizer_ok = True
            message = "all components explicitly frozen"
        value, gradient = objective(fitted)
        projected = gradient.copy()
        projected[(fitted <= minimum * (1 + 1e-7)) & (gradient > 0)] = 0
        projected[fixed] = 0
        pg = float(np.max(np.abs(projected)))
        at_upper = np.any(fitted[active] >= maximum[active] * (1 - 1e-7))
        condition = float(np.linalg.cond(np.einsum("i,ijk->jk", fitted, normalized_matrices)))
        conditions[column] = condition
        good = optimizer_ok and pg <= 1e-5 and not at_upper and condition <= 1e12
        variance[:, column] = fitted * units
        variance[fixed, column] = initial[fixed, column]
        ll[column] = -value - n * np.log(response_scale)
        gradients[column] = pg
        descent[:, column] = fixed & (gradient < -1e-5)
        success[column] = good
        messages.append(message)
        status = "converged" if good else "optimizer_hold"
        if good:
            for group in range(h + c):
                if fixed[group]:
                    lower[group, column] = upper[group, column] = proposals[group, column] = 0
                    continue

                def root(t):
                    point = fitted.copy()
                    point[group] = t
                    return objective(point)[0] - value - 1

                mid = fitted[group]
                high = max(mid * 2, minimum[group] * 2, 1e-8)
                try:
                    for _ in range(60):
                        if root(high) >= 0:
                            break
                        high *= 2
                    else:
                        raise ArithmeticError("upper likelihood root not bracketed")
                    high = brentq(root, mid, high, xtol=1e-12, rtol=1e-12, maxiter=100)
                    low_value = root(minimum[group])
                    if low_value >= 0 and mid > minimum[group]:
                        low = brentq(root, minimum[group], mid, xtol=1e-12, rtol=1e-12, maxiter=100)
                    elif profile == "legacy_3_1":
                        low = -high
                        artificial[group, column] = True
                    else:
                        low = minimum[group]
                        truncated[group, column] = True
                    lower[group, column], upper[group, column] = (
                        low * units[group],
                        high * units[group],
                    )
                    proposals[group, column] = multiplier * (high - low) * units[group] / 2
                except (ValueError, ArithmeticError, np.linalg.LinAlgError):
                    status = "scale_hold"
        statuses.append(status)
        evaluations[column] = count
    positive = bool(np.all(success) and np.all(np.isfinite(proposals)) and np.all(proposals > 0))
    return WFMMConditionalVariance(
        _frozen(variance[:h]),
        _frozen(variance[h:]),
        _frozen(beta),
        _frozen(ll),
        _frozen(lower),
        _frozen(upper),
        _frozen(proposals),
        _frozen_bool(frozen),
        _frozen_bool(artificial),
        _frozen_bool(truncated),
        _frozen_bool(success),
        _frozen(gradients),
        _frozen(conditions),
        _frozen_bool(descent),
        _frozen_int(evaluations),
        tuple(statuses),
        tuple(messages),
        profile,
        positive,
    )
