"""MULTI desktop BMPVPB/BMURPB reciprocal-density decisions."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray
from .beta_mixture import BetaMixture
from .multiplicity import _alpha, _pvalues


@dataclass(frozen=True)
class BetaMixtureTestingResult:
    """Arrays retain input order; order lists the sequence used for prefix maxima.

    Scores are historical desktop diagnostics, not posterior probabilities or
    calibrated multiple-testing adjusted p-values.
    """

    scores: FloatArray
    reject: NDArray[np.bool_]
    log_density: FloatArray
    order: NDArray[np.intp]
    sequence: str
    alpha: float


def beta_mixture_testing(
    pvalues: ArrayLike,
    model: BetaMixture,
    *,
    alpha: float = 0.05,
    sequence: str = "rank",
    legacy_endpoints: bool = False,
) -> BetaMixtureTestingResult:
    """Run desktop reciprocal-density decisions along the last axis.

    rank uses ascending p-values as BMPVPB does; input preserves the entered
    order as BMURPB does. Scores are min(1, prefix-max(1/mixture density)).
    Reject when score<=alpha, including equality despite the source comment.
    Stable log arithmetic avoids overflowing reciprocal densities. Mathematical
    zero density maps to score one and infinite density to score zero; request
    legacy_endpoints explicitly for the archived INITLN approximation.
    """
    x, alpha = _pvalues(pvalues), _alpha(alpha)
    if sequence not in ("rank", "input"):
        raise ValueError("sequence must be 'rank' or 'input'")
    if sequence == "rank":
        order = np.argsort(x, axis=-1, kind="stable")
    else:
        order = np.broadcast_to(np.arange(x.shape[-1]), x.shape).copy()
    log_density = model.logpdf(x, legacy_endpoints=legacy_endpoints)
    # Clamp in log space before exponentiating; never form infinity from 1/f.
    raw = np.exp(np.minimum(-log_density, 0))
    scores = np.maximum.accumulate(np.take_along_axis(raw, order, axis=-1), axis=-1)
    inverse = np.argsort(order, axis=-1)
    scores = np.take_along_axis(scores, inverse, axis=-1)
    return BetaMixtureTestingResult(scores, scores <= alpha, log_density, order, sequence, alpha)
