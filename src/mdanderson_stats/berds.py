"""Backward elimination via repeated data splitting (MD Anderson BERDS)."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import stdtr

from ._validation import FloatArray, finite, scalar


def _readonly(values: ArrayLike) -> FloatArray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BackwardElimination:
    """Selected predictor indices are zero-based; coefficients exclude the intercept."""

    selected: tuple[int, ...]
    coefficients: FloatArray
    intercept: float
    residual_sum_squares: float
    mean_squared_error: float
    native_mean_squared_error: float
    r_squared: float
    mallows_cp: float
    full_model_mean_squared_error: float


@dataclass(frozen=True)
class BERDSResult:
    model: BackwardElimination
    alpha: float
    validation_sum_squares: float
    alpha_domain: tuple[float, float]
    alphas: FloatArray
    scores: FloatArray
    split_alphas: FloatArray
    split_scores: FloatArray
    validation_indices: NDArray[np.intp]
    scoring: str


def _data(response: ArrayLike, predictors: ArrayLike) -> tuple[FloatArray, FloatArray]:
    y = finite(response, "response")
    x = finite(predictors, "predictors")
    if y.ndim != 1 or x.ndim != 2 or x.shape[0] != y.size or x.shape[1] < 1:
        raise ValueError("response must be a vector and predictors a matching nonempty matrix")
    if x.size > 2_000_000 or x.shape[1] > 200:
        raise ValueError("at most 200 predictors and two million predictor entries are supported")
    if y.size <= x.shape[1] + 1:
        raise ValueError("positive residual degrees of freedom are required for the full model")
    return y, x


def _fit(
    y: FloatArray, x: FloatArray, selected: list[int]
) -> tuple[FloatArray, float, float, FloatArray]:
    ym = float(np.mean(y))
    centered_y = y - ym
    yscale = float(np.max(np.abs(centered_y)))
    if yscale == 0 or not np.isfinite(yscale):
        raise ValueError("response must have finite, nonzero variation")
    if not selected:
        rss = float(np.sum((centered_y / yscale) ** 2) * yscale * yscale)
        if not np.isfinite(rss) or rss <= 0:
            raise ArithmeticError("residual sum of squares is not representable; rescale response")
        return np.empty(0), ym, rss, np.empty(0)
    z = x[:, selected]
    xm = z.mean(axis=0)
    z = z - xm
    scale = np.max(np.abs(z), axis=0)
    if np.any(scale == 0) or not np.all(np.isfinite(scale)):
        raise ValueError("predictors must have finite variation in every elimination set")
    u, s, vt = np.linalg.svd(z / scale, full_matrices=False)
    if s[-1] <= np.finfo(float).eps * max(z.shape) * s[0]:
        raise ValueError("collinear or numerically rank-deficient elimination design")
    b = vt.T @ ((u.T @ (centered_y / yscale)) / s)
    residual = centered_y / yscale - (z / scale) @ b
    rss_scaled = float(residual @ residual)
    if rss_scaled <= (np.finfo(float).eps * max(z.shape)) ** 2:
        raise ValueError("residual variance is zero or numerically unresolved")
    se = np.sqrt(np.sum((vt.T / s) ** 2, axis=1) * rss_scaled / (y.size - len(selected) - 1))
    pvalues = 2 * stdtr(y.size - len(selected) - 1, -np.abs(b / se))
    coefficients = b * yscale / scale
    intercept = ym - float(xm @ coefficients)
    rss = rss_scaled * yscale * yscale
    if (
        rss <= 0
        or not np.isfinite(rss)
        or not np.all(np.isfinite(coefficients))
        or not np.isfinite(intercept)
    ):
        raise ArithmeticError("regression output is not representable; rescale the data")
    return coefficients, intercept, rss, pvalues


def backward_elimination(
    response: ArrayLike, predictors: ArrayLike, *, alpha: float = 0.05
) -> BackwardElimination:
    """Delete the largest two-sided t-test p-value while it exceeds alpha.

    Always fits an intercept. Ties delete the lowest original predictor index.
    Returns both residual-df MSE and the original BERDS n-minus-p MSE.
    """
    y, x = _data(response, predictors)
    alpha = scalar(alpha, "alpha")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    selected = list(range(x.shape[1]))
    b, intercept, rss, pvalues = _fit(y, x, selected)
    full_mse = rss / (y.size - x.shape[1] - 1)
    while selected and float(np.max(pvalues)) > alpha:
        del selected[int(np.argmax(pvalues))]
        b, intercept, rss, pvalues = _fit(y, x, selected)
    coefficients = np.zeros(x.shape[1])
    coefficients[selected] = b
    total = float(np.sum((y - y.mean()) ** 2))
    if not np.isfinite(total) or full_mse <= 0:
        raise ArithmeticError("regression diagnostics are not representable; rescale response")
    k = len(selected)
    return BackwardElimination(
        tuple(selected),
        _readonly(coefficients),
        intercept,
        rss,
        rss / (y.size - k - 1),
        rss / (y.size - k),
        1 - rss / total,
        rss / full_mse - (y.size - 2 * (k + 1)),
        full_mse,
    )


def berds(
    response: ArrayLike,
    predictors: ArrayLike,
    *,
    repetitions: int = 20,
    elimination_fraction: float = 0.5,
    trim: float = 0.1,
    truncation_quantile: float = 0.9,
    seed: int | None = None,
    validation_indices: ArrayLike | None = None,
    scoring: str = "refit",
) -> BERDSResult:
    """Select a backward-elimination threshold using repeated held-out prediction.

    Explicit validation rows (one equally sized row per split) allow exact replay.
    ``scoring='native'`` reproduces the original stale-coefficient score and
    threshold lookup; ``'refit'`` scores the refitted model selected at each alpha.
    Neither option uses validation responses to estimate regression coefficients.
    """
    y, x = _data(response, predictors)
    if (
        isinstance(repetitions, (bool, np.bool_))
        or not isinstance(repetitions, (int, np.integer))
        or not 2 <= repetitions <= 10_000
    ):
        raise ValueError("repetitions must be an integer in 2..10000")
    fraction = scalar(elimination_fraction, "elimination_fraction")
    trim = scalar(trim, "trim")
    quantile = scalar(truncation_quantile, "truncation_quantile")
    if not 0 < fraction < 1 or not 0 <= trim < 0.5 or not 0 <= quantile <= 1:
        raise ValueError("fraction must be in (0,1), trim in [0,.5), quantile in [0,1]")
    if scoring not in ("refit", "native"):
        raise ValueError("scoring must be refit or native")
    n, p = x.shape
    if repetitions * p > 100_000 or repetitions**2 * p > 20_000_000 or repetitions * n > 20_000_000:
        raise ValueError("split storage or score-aggregation workload exceeds supported limits")
    k = int((1 - fraction) * n)
    if min(k, n - k) < 3 or n - k <= p + 1:
        raise ValueError("each split needs at least three rows and positive full-model residual df")
    if validation_indices is None:
        rng = np.random.default_rng(seed)
        splits = np.stack([rng.permutation(n)[:k] for _ in range(repetitions)])
    else:
        if seed is not None:
            raise ValueError("seed and explicit validation_indices are mutually exclusive")
        raw = finite(validation_indices, "validation_indices")
        if (
            raw.shape != (repetitions, k)
            or np.any(raw != np.floor(raw))
            or np.any((raw < 0) | (raw >= n))
        ):
            raise ValueError(
                "validation_indices need shape (repetitions, validation size) "
                "and valid integer indices"
            )
        splits = raw.astype(np.intp)
        if any(np.unique(row).size != k for row in splits):
            raise ValueError("a validation split cannot repeat an observation")
    # Preserve the original asymmetric integer rounding for fractional trim counts.
    start, stop = int(trim * repetitions), int(repetitions - trim * repetitions)
    if stop - start < 2:
        raise ValueError("trimming must retain at least two split scores")
    thresholds = np.empty((repetitions, p))
    losses = np.empty((repetitions, p + 1))
    for i, validation in enumerate(splits):
        keep = np.ones(n, dtype=bool)
        keep[validation] = False
        ey, ex = y[keep], x[keep]
        vy, vx = y[validation], x[validation]
        selected = list(range(p))
        b, intercept, _, pv = _fit(ey, ex, selected)
        losses[i, 0] = np.sum((vy - intercept - vx[:, selected] @ b) ** 2)
        for deleted in range(p):
            worst = int(np.argmax(pv))
            thresholds[i, deleted] = pv[worst]
            del selected[worst]
            old_b = np.delete(b, worst)
            if scoring == "native":
                losses[i, deleted + 1] = np.sum((vy - intercept - vx[:, selected] @ old_b) ** 2)
            b, intercept, _, pv = _fit(ey, ex, selected)
            if scoring == "refit":
                losses[i, deleted + 1] = np.sum((vy - intercept - vx[:, selected] @ b) ** 2)
    if not np.all(np.isfinite(losses)):
        raise ArithmeticError("validation losses are not representable; rescale response")
    minima, maxima = thresholds.min(axis=1), thresholds.max(axis=1)
    lower = float(np.quantile(minima, quantile))
    upper = float(np.quantile(maxima, 1 - quantile))
    if maxima.min() <= minima.max():
        upper = 1.0
    alphas = np.unique(thresholds)
    alphas = alphas[(alphas >= lower) & (alphas <= upper)]
    if not alphas.size:
        raise ValueError("truncated alpha domain contains no candidate; change truncation quantile")
    scores = np.empty(alphas.size)
    effective_thresholds = np.minimum.accumulate(thresholds, axis=1)
    reverse = thresholds[:, ::-1]
    for j, alpha in enumerate(alphas):
        if scoring == "refit":
            deletions = np.sum(effective_thresholds > alpha, axis=1)
        else:
            accepted = reverse >= alpha
            position = np.where(accepted.any(axis=1), accepted.argmax(axis=1), p)
            deletions = p - position
        values = np.sort(losses[np.arange(repetitions), deletions])
        scores[j] = np.sum(values[start:stop] / (stop - start))
    best = scores.size - 1 - int(np.argmin(scores[::-1]))
    alpha = float(alphas[best])
    splits.setflags(write=False)
    return BERDSResult(
        backward_elimination(y, x, alpha=alpha),
        alpha,
        float(scores[best]),
        (lower, upper),
        _readonly(alphas),
        _readonly(scores),
        _readonly(thresholds),
        _readonly(losses),
        splits,
        scoring,
    )
