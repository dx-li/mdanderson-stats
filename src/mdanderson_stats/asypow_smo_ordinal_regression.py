"""SMO for proportional-odds and complementary-log-log ordinal regression."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from ._validation import FloatArray, finite
from .asypow_generic import asypow_smo_generic
from .asypow_ordinal import _category_logs
from .asypow_smo import SMOPower
from .asypow_smo_regression import _log_exp_remainder


def asypow_smo_ordinal_regression(
    parameters: ArrayLike,
    covariates: ArrayLike,
    *,
    constraints: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    quadratic: bool = False,
    link: str = "logistic",
    observations: ArrayLike = 1,
    group_size: ArrayLike = 1,
    subtract_df: bool = True,
    tolerance: float = 1e-8,
) -> SMOPower:
    """Rows contain ordered intercepts, then shared linear/quadratic slopes.

    Supply computational bounds preserving intercept order for all evaluations.
    Constraints use one-based indices in flattened group-by-group order.
    """
    theta = finite(parameters, "parameters")
    if theta.ndim == 1:
        theta = theta[None, :]
    if not isinstance(quadratic, (bool, np.bool_)):
        raise ValueError("quadratic must be boolean")
    slopes = 2 if quadratic else 1
    if theta.ndim != 2 or theta.shape[1] <= slopes or not 1 <= theta.size <= 500:
        raise ValueError("parameters need thresholds and slopes, with at most 500 entries")
    if link not in ("logistic", "cloglog"):
        raise ValueError("link must be logistic or cloglog")
    groups, width = theta.shape
    thresholds = width - slopes
    x = finite(covariates, "covariates")
    if x.ndim == 1:
        x = np.broadcast_to(x, (groups, x.size))
    if x.ndim != 2 or x.shape[0] != groups or not 1 <= x.size <= 10_000:
        raise ValueError("covariates need one row per group and 1..10000 entries")
    if x.size * (thresholds + 1) > 1_000_000:
        raise ValueError("design has more than one million category/point combinations")
    count = np.broadcast_to(finite(observations, "observations"), x.shape)
    allocation = np.broadcast_to(finite(group_size, "group_size"), (groups,))
    if np.any(count < 0) or np.any(allocation <= 0) or np.any(np.sum(count > 0, axis=1) == 0):
        raise ValueError("allocations must be positive; every group needs positive observations")
    used = count > 0
    index = np.nonzero(used)[0]
    with np.errstate(over="ignore", invalid="ignore"):
        design = np.vander(x[used], slopes + 1, increasing=True)
    design = finite(design, "polynomial design")
    for g in range(groups):
        block = design[index == g]
        scale = np.max(np.abs(block), axis=0)
        if np.any(scale == 0) or np.linalg.matrix_rank(block / scale) < slopes + 1:
            raise ValueError("each group's observed covariates must identify its slopes")
    log_weight = np.log(count[used]) + np.log(allocation[index])
    log_weight -= logsumexp(log_weight)

    def logs(q: FloatArray) -> tuple[FloatArray, FloatArray]:
        rows = q.reshape(theta.shape)
        if np.any(np.diff(rows[:, :thresholds], axis=1) <= 0):
            raise ValueError("ordinal intercepts must remain strictly ordered; review bounds")
        with np.errstate(over="ignore", invalid="ignore"):
            eta = (
                rows[index, :thresholds]
                + np.sum(design[:, 1:] * rows[index, thresholds:], axis=1)[:, None]
            )
        eta = finite(eta, "ordinal predictors")
        evaluated = [_category_logs(point, link) for point in eta]
        return np.stack([pair[0] for pair in evaluated]), np.stack([pair[1] for pair in evaluated])

    alternative, log_f = logs(theta.ravel())
    log_info = logsumexp(
        np.logaddexp(2 * log_f - alternative[:, :-1], 2 * log_f - alternative[:, 1:]), axis=1
    )
    log_scale = float(logsumexp(log_weight + log_info))
    if not np.isfinite(log_scale):
        raise ArithmeticError("ordinal predictor information is numerically unresolved")
    scaled_mass = log_weight[:, None] + alternative - log_scale

    def likelihood(_p: FloatArray, q: FloatArray) -> float:
        candidate, _ = logs(q)
        with np.errstate(over="ignore", under="ignore"):
            return -float(
                np.exp(logsumexp(scaled_mass + _log_exp_remainder(candidate - alternative)))
            )

    def gradient(_p: FloatArray, q: FloatArray) -> FloatArray:
        candidate, derivative = logs(q)
        ratio = alternative - candidate
        gap = ratio[:, :-1] - ratio[:, 1:]
        with np.errstate(divide="ignore", under="ignore", over="ignore"):
            log_difference = np.maximum(ratio[:, :-1], ratio[:, 1:]) + np.log(
                -np.expm1(-np.abs(gap))
            )
            threshold_score = np.sign(gap) * np.exp(
                log_weight[:, None] - log_scale + derivative + log_difference
            )
        point_gradient = np.column_stack(
            (threshold_score, threshold_score.sum(axis=1)[:, None] * design[:, 1:])
        )
        result = np.zeros_like(theta)
        np.add.at(result, index, point_gradient)
        return finite(result.ravel(), "ordinal gradient")

    fitted = asypow_smo_generic(
        theta,
        likelihood,
        gradient=gradient,
        constraints=constraints,
        lower=lower,
        upper=upper,
        subtract_df=subtract_df,
        tolerance=tolerance,
    )
    with np.errstate(divide="ignore", over="ignore", under="ignore"):
        w = float(np.exp(np.log(fitted.divergence_per_observation) + log_scale))
    if not np.isfinite(w):
        raise ArithmeticError("ordinal regression divergence is not representable")
    return SMOPower(w, fitted.degrees_of_freedom, fitted.null_parameters, fitted.subtract_df)
