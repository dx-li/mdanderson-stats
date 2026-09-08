"""CTA RELRISK odds ratios with corrected or original confidence limits."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr, ndtri_exp

from ._validation import FloatArray, finite, scalar


@dataclass(frozen=True)
class OddsRatio:
    odds: FloatArray
    odds_ratio: FloatArray
    log_odds_ratio: FloatArray
    standard_error: FloatArray
    lower: FloatArray
    upper: FloatArray
    log_lower: FloatArray
    log_upper: FloatArray
    alpha: float
    multiplier: float
    risk_factor: str
    response_index: int
    legacy: bool


def odds_ratio(
    observed: ArrayLike,
    *,
    risk_factor: Literal["rows", "columns"] = "columns",
    response_index: int = 0,
    alpha: float = 0.05,
    legacy: bool = False,
) -> OddsRatio:
    """Compare response odds in risk group zero versus one on final 2x2 axes.

    Positive finite cells (including fractional counts) are required. The
    response index selects the response on the axis opposite the risk factor.
    Standard error refers to the log odds ratio. Default intervals are log-Wald
    intervals; legacy uses CTA's incorrect normal CDF multiplier Phi(alpha/2).
    Leading dimensions form batches. Unrepresentable exponentials become 0/inf;
    corresponding log estimates and limits remain available.
    """
    ob = finite(observed, "observed")
    if ob.ndim < 2 or ob.shape[-2:] != (2, 2) or np.any(ob <= 0):
        raise ValueError("observed must contain strictly positive 2x2 tables")
    if risk_factor not in ("rows", "columns"):
        raise ValueError("risk_factor must be rows or columns")
    if (
        isinstance(response_index, (bool, np.bool_))
        or not isinstance(response_index, (int, np.integer))
        or response_index not in (0, 1)
    ):
        raise ValueError("response_index must be zero or one")
    if not isinstance(legacy, (bool, np.bool_)):
        raise ValueError("legacy must be boolean")
    alpha = scalar(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    pos, neg = int(response_index), 1 - int(response_index)
    logs = np.log(ob)
    log_odds = (
        logs[..., pos, :] - logs[..., neg, :]
        if risk_factor == "columns"
        else logs[..., :, pos] - logs[..., :, neg]
    )
    log_ratio = np.asarray(log_odds[..., 0] - log_odds[..., 1])
    # Taking square roots before reciprocals and using hypot avoids overflow
    # even when cells are subnormal and the reciprocal variance is infinite.
    se = np.asarray(np.hypot.reduce((1 / np.sqrt(ob)).reshape((*ob.shape[:-2], 4)), axis=-1))
    multiplier = float(ndtr(alpha / 2) if legacy else -ndtri_exp(np.log(alpha) - np.log(2)))
    log_lower = np.asarray(log_ratio - multiplier * se)
    log_upper = np.asarray(log_ratio + multiplier * se)
    with np.errstate(over="ignore", under="ignore"):
        odds, ratio, lower, upper = map(np.exp, (log_odds, log_ratio, log_lower, log_upper))
    arrays = [
        np.asarray(x) for x in (odds, ratio, log_ratio, se, lower, upper, log_lower, log_upper)
    ]
    for value in arrays:
        value.flags.writeable = False
    return OddsRatio(
        arrays[0],
        arrays[1],
        arrays[2],
        arrays[3],
        arrays[4],
        arrays[5],
        arrays[6],
        arrays[7],
        alpha,
        multiplier,
        risk_factor,
        pos,
        bool(legacy),
    )
