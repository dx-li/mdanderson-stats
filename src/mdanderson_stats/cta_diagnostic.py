"""CTA sensitivity, specificity and predictive values with explicit table axes."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite


@dataclass(frozen=True)
class DiagnosticAccuracy:
    sensitivity: FloatArray
    specificity: FloatArray
    positive_predictive_value: FloatArray
    negative_predictive_value: FloatArray
    standard_errors: FloatArray
    denominators: FloatArray
    standard: str
    positive_index: int
    legacy: bool


def diagnostic_accuracy(
    observed: ArrayLike,
    *,
    standard: Literal["rows", "columns"] = "columns",
    positive_index: int = 0,
    legacy: bool = False,
) -> DiagnosticAccuracy:
    """Analyze final 2x2 axes; return probabilities and conditional binomial SEs.

    Standard identifies the reference-classification axis, and positive_index
    identifies the positive class on both axes (zero-based). Standard errors
    and denominators have final order sensitivity, specificity, PPV, NPV.
    Legacy returns CTA's count SD sqrt(n*p*(1-p)) mislabeled as a probability SE.
    A zero denominator gives NaN for the corresponding probability and error.
    """
    ob = finite(observed, "observed")
    if ob.ndim < 2 or ob.shape[-2:] != (2, 2) or np.any(ob < 0):
        raise ValueError("observed must contain nonnegative 2x2 tables")
    if standard not in ("rows", "columns"):
        raise ValueError("standard must be rows or columns")
    if (
        isinstance(positive_index, (bool, np.bool_))
        or not isinstance(positive_index, (int, np.integer))
        or positive_index not in (0, 1)
    ):
        raise ValueError("positive_index must be zero or one")
    if not isinstance(legacy, (bool, np.bool_)):
        raise ValueError("legacy must be boolean")
    pos, neg = int(positive_index), 1 - int(positive_index)
    tp, tn = ob[..., pos, pos], ob[..., neg, neg]
    fn, fp = (
        (ob[..., neg, pos], ob[..., pos, neg])
        if standard == "columns"
        else (ob[..., pos, neg], ob[..., neg, pos])
    )
    successes = np.stack((tp, tn, tp, tn), axis=-1)
    failures = np.stack((fn, fp, fp, fn), axis=-1)
    with np.errstate(over="ignore"):
        denominators = successes + failures
    if not np.all(np.isfinite(denominators)):
        raise ValueError("classification margins must be finite")
    valid = denominators > 0
    probabilities = np.divide(
        successes, denominators, out=np.full_like(successes, np.nan), where=valid
    )
    complements = np.divide(failures, denominators, out=np.full_like(failures, np.nan), where=valid)
    # Square roots first prevent overflowing the count-SD variance product.
    scale = np.sqrt(denominators)
    product = np.sqrt(probabilities) * np.sqrt(complements)
    with np.errstate(divide="ignore", invalid="ignore"):
        errors = product * scale if legacy else product / scale
    estimates = [np.asarray(probabilities[..., i]) for i in range(4)]
    for value in (*estimates, errors, denominators):
        value.flags.writeable = False
    return DiagnosticAccuracy(
        estimates[0],
        estimates[1],
        estimates[2],
        estimates[3],
        errors,
        denominators,
        standard,
        pos,
        bool(legacy),
    )
