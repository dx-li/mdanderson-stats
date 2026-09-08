"""Batched contingency-table chi-square analysis from CTA's CHISQT workflow."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaincc

from ._validation import FloatArray, finite, scalar


@dataclass(frozen=True)
class ContingencyChiSquare:
    observed: FloatArray
    expected: FloatArray
    contributions: FloatArray
    row_percent: FloatArray
    column_percent: FloatArray
    statistic: FloatArray
    pvalue: FloatArray
    degrees_of_freedom: int
    yates_statistic: FloatArray | None
    yates_pvalue: FloatArray | None
    cochran_statistic: FloatArray | None
    cochran_pvalue: FloatArray | None
    minimum_expected: FloatArray
    percent_small_expected: FloatArray
    expected_threshold: float
    legacy: bool


def contingency_chi_square(
    observed: ArrayLike,
    *,
    expected_threshold: float = 5,
    legacy: bool = False,
) -> ContingencyChiSquare:
    """Analyze final two axes as independent row/column contingency tables.

    Leading dimensions form a batch. Finite nonnegative fractional frequencies
    are allowed as in CTA; every margin must be positive. Default Yates is
    clipped at zero and supplied only for 2x2 tables. Legacy mode subtracts .5
    without clipping, for every table shape, matching CHISQT. The source-specific
    Cochran statistic is supplied for 2x2 tables in either mode.
    """
    ob = finite(observed, "observed").copy()
    if ob.ndim < 2 or min(ob.shape[-2:]) < 2 or np.any(ob < 0):
        raise ValueError(
            "observed must contain nonnegative tables with at least two rows and columns"
        )
    threshold = scalar(expected_threshold, "expected_threshold")
    if threshold < 0:
        raise ValueError("expected_threshold must be nonnegative")
    if not isinstance(legacy, (bool, np.bool_)):
        raise ValueError("legacy must be boolean")
    with np.errstate(over="ignore"):
        rows, cols = ob.sum(axis=-1, keepdims=True), ob.sum(axis=-2, keepdims=True)
        total = rows.sum(axis=-2, keepdims=True)
    if np.any(rows <= 0) or np.any(cols <= 0) or not np.all(np.isfinite(total)):
        raise ValueError("every row/column margin must be positive and totals finite")
    expected = (rows / total) * cols
    if np.any(expected <= 0):
        raise ValueError("expected frequencies underflowed to zero")
    delta = np.abs(ob - expected)
    with np.errstate(over="ignore"):
        contributions = (delta / np.sqrt(expected)) ** 2
        statistic = contributions.sum(axis=(-2, -1))
    df = (ob.shape[-2] - 1) * (ob.shape[-1] - 1)
    yates = yates_p = cochran = cochran_p = None
    is_two = ob.shape[-2:] == (2, 2)
    if is_two or legacy:
        corrected = delta - 0.5 if legacy else np.maximum(delta - 0.5, 0)
        with np.errstate(over="ignore"):
            yates = np.asarray(((corrected / np.sqrt(expected)) ** 2).sum(axis=(-2, -1)))
        yates_p = np.asarray(gammaincc(df / 2, yates / 2))
    if is_two:
        index = expected.reshape(*ob.shape[:-2], 4).argmin(axis=-1)
        flat_ob, flat_ex = ob.reshape(*ob.shape[:-2], 4), expected.reshape(*ob.shape[:-2], 4)
        cell = np.take_along_axis(flat_ob, index[..., None], axis=-1)[..., 0]
        smallest = np.take_along_axis(flat_ex, index[..., None], axis=-1)[..., 0]
        difference = np.abs(cell - smallest)
        adjustment = np.where(
            cell / 2 <= smallest,
            np.floor(difference) + np.floor(2 * (difference - np.floor(difference))) / 2,
            difference - 0.5,
        )
        with np.errstate(over="ignore"):
            cochran = np.asarray(
                ((adjustment[..., None, None] / np.sqrt(expected)) ** 2).sum(axis=(-2, -1))
            )
        cochran_p = np.asarray(gammaincc(0.5, cochran / 2))
    row_percent, column_percent = ob / rows * 100, ob / cols * 100
    statistic = np.asarray(statistic)
    pvalue = np.asarray(gammaincc(df / 2, statistic / 2))
    minimum = np.asarray(expected.min(axis=(-2, -1)))
    percent_small = np.asarray((expected <= threshold).mean(axis=(-2, -1)) * 100)
    for value in (
        ob,
        expected,
        contributions,
        row_percent,
        column_percent,
        statistic,
        pvalue,
        yates,
        yates_p,
        cochran,
        cochran_p,
        minimum,
        percent_small,
    ):
        if value is not None:
            value.flags.writeable = False
    return ContingencyChiSquare(
        ob,
        expected,
        contributions,
        row_percent,
        column_percent,
        statistic,
        pvalue,
        df,
        yates,
        yates_p,
        cochran,
        cochran_p,
        minimum,
        percent_small,
        threshold,
        bool(legacy),
    )
