"""DIAG population projections and prevalence-dependent predictive values."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, count, finite
from .cta_diagnostic import diagnostic_accuracy


def _owned(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=float, copy=True)
    result.flags.writeable = False
    return result


def _probability(value: ArrayLike, name: str) -> FloatArray:
    p = finite(value, name)
    if np.any((p < 0) | (p > 1)):
        raise ValueError(f"{name} must be in [0, 1]")
    return p


@dataclass(frozen=True)
class DiagnosticPopulation:
    """Final table axes are test (+,-) rows and disease (+,-) columns.

    Counts are expected counts, not rounded individuals. A predictive value is
    NaN if its conditioning test outcome has probability zero.
    """

    sensitivity: FloatArray
    specificity: FloatArray
    prevalence: FloatArray
    false_positive_rate: FloatArray
    false_negative_rate: FloatArray
    positive_predictive_value: FloatArray
    negative_predictive_value: FloatArray
    false_discovery_rate: FloatArray
    joint_probabilities: FloatArray
    expected_counts: FloatArray


def _from_logs(
    log_sensitivity: ArrayLike,
    log_specificity: ArrayLike,
    log_false_positive: ArrayLike,
    log_false_negative: ArrayLike,
    prevalence: ArrayLike,
    population: ArrayLike,
) -> DiagnosticPopulation:
    se, sp, fp, fn, p, n = np.broadcast_arrays(
        log_sensitivity,
        log_specificity,
        log_false_positive,
        log_false_negative,
        _probability(prevalence, "prevalence"),
        count(population, "population"),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        lp, lq = np.log(p), np.log1p(-p)
        tp, fpj, fnj, tn = se + lp, fp + lq, fn + lp, sp + lq
        positive, negative = np.logaddexp(tp, fpj), np.logaddexp(fnj, tn)
        ppv, npv, fdr = np.exp(tp - positive), np.exp(tn - negative), np.exp(fpj - positive)
    logs = np.stack((np.stack((tp, fpj), -1), np.stack((fnj, tn), -1)), -2)
    joint = np.exp(logs)
    # Retain counts that can be represented after multiplying a tiny probability
    # by a population, even when the standalone probability underflows.
    with np.errstate(divide="ignore"):
        counts = np.exp(logs + np.log(n)[..., None, None])
    return DiagnosticPopulation(
        *[
            _owned(x)
            for x in (
                np.exp(se),
                np.exp(sp),
                p,
                np.exp(fp),
                np.exp(fn),
                ppv,
                npv,
                fdr,
                joint,
                counts,
            )
        ]
    )


def diagnostic_population(
    sensitivity: ArrayLike,
    specificity: ArrayLike,
    prevalence: ArrayLike,
    *,
    population: ArrayLike = 10000,
) -> DiagnosticPopulation:
    """Apply Bayes' rule at one or many prevalences; all arguments broadcast."""
    se, sp = _probability(sensitivity, "sensitivity"), _probability(specificity, "specificity")
    with np.errstate(divide="ignore"):
        return _from_logs(
            np.log(se), np.log(sp), np.log1p(-sp), np.log1p(-se), prevalence, population
        )


def diagnostic_population_from_counts(
    observed: ArrayLike, *, prevalence: ArrayLike | None = None, population: ArrayLike = 10000
) -> DiagnosticPopulation:
    """Estimate from 2x2 integer tables, optionally substituting a target prevalence.

    Both disease groups must be observed to estimate sensitivity and specificity.
    A prevalence override changes the target population, not the observed table.
    """
    table = count(observed, "observed")
    accuracy = diagnostic_accuracy(table, standard="columns", positive_index=0)
    if np.any(~np.isfinite(accuracy.sensitivity)) or np.any(~np.isfinite(accuracy.specificity)):
        raise ValueError("both diseased and non-diseased groups must be observed")
    n = table.sum(axis=(-2, -1))
    if np.any(n >= 2**53):
        raise ValueError("total observed count must be smaller than 2**53")
    p = table[..., :, 0].sum(axis=-1) / n if prevalence is None else prevalence
    # Obtain complements from the observed cells, avoiding subtraction from a
    # rounded sensitivity/specificity at very unbalanced table margins.
    diseased = table[..., :, 0].sum(axis=-1)
    healthy = table[..., :, 1].sum(axis=-1)
    with np.errstate(divide="ignore"):
        return _from_logs(
            np.log(accuracy.sensitivity),
            np.log(accuracy.specificity),
            np.log(table[..., 0, 1] / healthy),
            np.log(table[..., 1, 0] / diseased),
            p,
            population,
        )
