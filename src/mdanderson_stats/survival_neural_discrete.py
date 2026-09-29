"""Discrete-time neural survival objectives and their logit derivatives.

The label transformations and losses follow the pinned PyCox 0.3.0 source.
This module intentionally contains only the training loss kernels; prediction
and network optimization remain in their owning modules.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, logsumexp

FloatArray = NDArray[np.float64]
DiscreteFamily = Literal["loghaz", "deephit", "pchazard"]
_MAX_PAIR_TIME_WORK = 20_000_000
_LOG_MAX = float(np.log(np.finfo(np.float64).max))


def _real_finite(value: ArrayLike, name: str) -> FloatArray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real array") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _inputs(
    family: str, time: ArrayLike, event: ArrayLike, logits: ArrayLike, cuts: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    t = _real_finite(time, "time")
    e = _real_finite(event, "event")
    z = _real_finite(logits, "logits")
    cut = _real_finite(cuts, "cuts")
    if t.ndim != 1 or e.shape != t.shape or np.any(t < 0) or np.any((e != 0) & (e != 1)):
        raise ValueError("time and binary event must be aligned one-dimensional arrays")
    if z.ndim != 2 or z.shape[0] != t.size or not t.size:
        raise ValueError("logits must be a nonempty row-aligned matrix")
    if cut.ndim != 1 or cut.size < 2 or np.any(cut[1:] <= cut[:-1]):
        raise ValueError("cuts must be a strictly increasing vector of at least two values")
    if np.any(t < cut[0]):
        raise ValueError("cuts must start at or before every observed time")
    if family in ("loghaz", "deephit") and z.shape[1] != cut.size:
        raise ValueError("LogisticHazard and DeepHit logits must have one output per cut")
    if family == "pchazard" and z.shape[1] != cut.size - 1:
        raise ValueError("PCHazard logits must have one output per interval")
    if family == "pchazard":
        with np.errstate(over="ignore", invalid="ignore"):
            widths = cut[1:] - cut[:-1]
        if not np.all(np.isfinite(widths)):
            raise ValueError("PCHazard cut widths must be finite")
    return t, e, z, cut


def _labels(
    time: FloatArray, event: FloatArray, cuts: FloatArray, *, censor_side: Literal["left", "right"]
) -> tuple[FloatArray, FloatArray, NDArray[np.int64]]:
    """PyCox DiscretizeUnknownC labels, including its right-censor-at-max rule."""
    t = time.copy()
    e = event.copy()
    beyond = t > cuts[-1]
    t[beyond] = cuts[-1]
    e[beyond] = 0.0
    # PyCox's discretize(side="right") means round an interval observation up
    # while retaining an exact cut itself; NumPy searchsorted(side="left")
    # implements that boundary convention.
    event_index = np.searchsorted(cuts, t, side="left")
    censor_index = np.searchsorted(cuts, t, side="left")
    exact = (censor_index < cuts.size) & (cuts[np.minimum(censor_index, cuts.size - 1)] == t)
    if censor_side == "left":
        censor_index -= (~exact).astype(np.int64)
    # `right` means round a censor to the next cut, like an event, but the
    # equality-aware search above leaves exact-cut observations on that cut.
    index = np.where(e == 1.0, event_index, censor_index)
    if np.any((index < 0) | (index >= cuts.size)):
        raise ValueError("time cannot be represented on the supplied cut grid")
    return t, e, index.astype(np.int64, copy=False)


def _log_softplus(z: FloatArray) -> FloatArray:
    """Stable log(softplus(z)); the asymptote is used only below -35."""
    result = np.array(z, copy=True)
    moderate = z >= -35.0
    result[moderate] = np.log(np.logaddexp(0.0, z[moderate]))
    return result


def _softplus_log_derivative(z: FloatArray) -> FloatArray:
    """Derivative of log(softplus(z)), stable when softplus underflows."""
    result = np.ones_like(z)
    moderate = z >= -35.0
    result[moderate] = expit(z[moderate]) / np.logaddexp(0.0, z[moderate])
    return result


def _loghaz(
    t: FloatArray, e: FloatArray, z: FloatArray, cuts: FloatArray
) -> tuple[float, FloatArray]:
    _, transformed_event, index = _labels(t, e, cuts, censor_side="left")
    bins = np.arange(z.shape[1])[None, :]
    observed = bins <= index[:, None]
    targets = np.zeros_like(z)
    event_rows = np.flatnonzero(transformed_event == 1.0)
    targets[event_rows, index[event_rows]] = 1.0
    loss_terms = np.logaddexp(0.0, z) * observed - targets * z
    gradient = (expit(z) * observed - targets) / t.size
    loss = float(np.sum(loss_terms, dtype=np.float64) / t.size)
    if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
        raise ArithmeticError("LogisticHazard loss or gradient is not representable")
    return loss, gradient


def _deephit(
    t: FloatArray,
    e: FloatArray,
    z: FloatArray,
    cuts: FloatArray,
    *,
    alpha: float,
    sigma: float,
) -> tuple[float, FloatArray]:
    _, transformed_event, index = _labels(t, e, cuts, censor_side="left")
    n, outputs = z.shape
    work = n * n * outputs
    if work > _MAX_PAIR_TIME_WORK:
        raise ValueError("DeepHit pair-by-time work exceeds the bounded loss budget")
    logits = np.column_stack((z, np.zeros(n, dtype=np.float64)))
    log_partition = logsumexp(logits, axis=1)
    logp = logits - log_partition[:, None]
    probs = np.exp(logp)
    nll_gradient = np.zeros_like(z)
    nll_loss = 0.0
    for row in range(n):
        at = int(index[row])
        if transformed_event[row] == 1.0:
            nll_loss -= float(logp[row, at]) / n
            nll_gradient[row] += probs[row, :-1] / n
            nll_gradient[row, at] -= 1.0 / n
        else:
            # NLL-PMF uses probability strictly after the censor index; the
            # fixed appended tail cell makes censoring at the last cut finite.
            log_tail = float(logsumexp(logits[row, at + 1 :]) - log_partition[row])
            nll_loss -= log_tail / n
            conditional = np.exp(logits[row, at + 1 :] - logsumexp(logits[row, at + 1 :]))
            nll_gradient[row] += probs[row, :-1] / n
            first = at + 1
            last = min(outputs, logits.shape[1] - 1)
            if first < last:
                nll_gradient[row, first:last] -= conditional[: last - first] / n

    rank_gradient = np.zeros_like(z)
    rank_loss = 0.0
    if alpha < 1.0:
        weight = (1.0 - alpha) / (n * n)
        cdf = np.cumsum(probs[:, :outputs], axis=1)
        for i in range(n):
            if transformed_event[i] != 1.0:
                continue
            k = int(index[i])
            eligible = (index > k) | ((index == k) & (transformed_event == 0.0))
            js = np.flatnonzero(eligible)
            if not js.size:
                continue
            cdf_i = float(cdf[i, k])
            cdf_j = cdf[js, k]
            tail_i = float(np.sum(probs[i, k + 1 :]))
            tail_j = np.sum(probs[js, k + 1 :], axis=1)
            differences = cdf_i - cdf_j
            with np.errstate(over="ignore", divide="ignore", under="ignore", invalid="ignore"):
                exponents = -differences / sigma
                terms = weight * np.exp(exponents)
                derivatives = -terms / sigma
            if np.any(exponents > _LOG_MAX) or not np.all(np.isfinite(derivatives)):
                raise ArithmeticError(
                    "DeepHit ranking loss or gradient exceeds floating-point range"
                )
            rank_loss += float(np.sum(terms, dtype=np.float64))
            # Derivatives of F_i(k) and -F_j(k) with respect to their logits.
            gi = np.empty(outputs, dtype=np.float64)
            gj = np.empty((js.size, outputs), dtype=np.float64)
            gi[: k + 1] = probs[i, : k + 1] * tail_i
            gj[:, : k + 1] = -probs[js, : k + 1] * tail_j[:, None]
            if k + 1 < outputs:
                gi[k + 1 :] = -probs[i, k + 1 : outputs] * cdf_i
                gj[:, k + 1 :] = probs[js, k + 1 : outputs] * cdf_j[:, None]
            rank_gradient[i] += float(np.sum(derivatives)) * gi
            rank_gradient[js] += derivatives[:, None] * gj

    loss = alpha * nll_loss + rank_loss
    gradient = alpha * nll_gradient + rank_gradient
    if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
        raise ArithmeticError("DeepHit loss or gradient is not representable")
    return float(loss), gradient


def _pchazard(
    time: FloatArray, event: FloatArray, z: FloatArray, cuts: FloatArray
) -> tuple[float, FloatArray]:
    transformed_time, transformed_event, index = _labels(time, event, cuts, censor_side="right")
    if np.any((transformed_time == cuts[0]) & (transformed_event == 1.0)):
        raise ValueError("PCHazard cannot represent an event at its initial cut")
    keep = transformed_time > cuts[0]
    if not np.any(keep):
        raise ValueError("PCHazard has no observations after its initial cut")
    t = transformed_time[keep]
    e = transformed_event[keep]
    idx = index[keep] - 1
    if np.any((idx < 0) | (idx >= z.shape[1])):
        raise ValueError("PCHazard times must be after the first cut")
    widths = np.diff(cuts)
    frac = np.clip((t - cuts[idx]) / widths[idx], 0.0, 1.0)
    rates = np.logaddexp(0.0, z[keep])
    gradient = np.zeros_like(z)
    normalizer = int(np.count_nonzero(keep))
    total_loss = 0.0
    softplus_derivative = expit(z[keep])
    retained_rows = np.flatnonzero(keep)
    for local_row, (row, j_value) in enumerate(zip(retained_rows, idx, strict=True)):
        j = int(j_value)
        exposure = np.zeros(z.shape[1], dtype=np.float64)
        exposure[:j] = 1.0
        exposure[j] = frac[local_row]
        total_loss += float(rates[local_row] @ exposure)
        gradient[row, :j] += softplus_derivative[local_row, :j] / normalizer
        gradient[row, j] += softplus_derivative[local_row, j] * frac[local_row] / normalizer
        if e[local_row] == 1.0:
            total_loss -= float(_log_softplus(np.array([z[row, j]]))[0])
            gradient[row, j] -= _softplus_log_derivative(np.array([z[row, j]]))[0] / normalizer
    loss = total_loss / normalizer
    gradient[~keep] = 0.0
    if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
        raise ArithmeticError("PCHazard loss or gradient is not representable")
    return float(loss), gradient


def _loss_gradient(
    family: str,
    time: ArrayLike,
    event: ArrayLike,
    logits: ArrayLike,
    cuts: ArrayLike,
    *,
    alpha: float = 0.2,
    rank_sigma: float = 0.1,
) -> tuple[float, FloatArray, FloatArray, FloatArray]:
    """Return mean loss and derivative with respect to discrete model logits.

    ``alpha`` is PyCox DeepHit's NLL weight in
    ``alpha*NLL + (1-alpha)*ranking``. It is ignored for the other families.
    Observations after the last cut are administratively censored at that cut,
    matching PyCox ``right_censor=True``. Times before the first cut are
    rejected rather than mapped outside the trained time domain. PCHazard
    censors exactly at its first cut contribute no likelihood, matching the
    pinned loss mask; an event exactly there is rejected as a stricter Python
    input rule instead of being silently converted and dropped by PyCox.
    """
    if family not in ("loghaz", "deephit", "pchazard"):
        raise ValueError("family must be loghaz, deephit, or pchazard")
    t, e, z, cut = _inputs(family, time, event, logits, cuts)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    if not np.isfinite(rank_sigma) or rank_sigma <= 0.0:
        raise ValueError("rank_sigma must be finite and positive")
    if family == "loghaz":
        loss, gradient = _loghaz(t, e, z, cut)
    elif family == "deephit":
        loss, gradient = _deephit(t, e, z, cut, alpha=alpha, sigma=rank_sigma)
    else:
        loss, gradient = _pchazard(t, e, z, cut)
    return loss, gradient, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
