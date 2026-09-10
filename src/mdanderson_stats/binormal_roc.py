"""DTROC's continuous-marker binormal ROC model, cut-points and predictive values."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import log_ndtr, ndtr, ndtri

from ._validation import FloatArray, finite
from .diagnostic_population import DiagnosticPopulation, _from_logs, _owned, _probability


def _standardize(value: FloatArray, mean: FloatArray, sd: FloatArray) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        delta = value - mean
        direct = delta / sd
        fallback = value / sd - mean / sd
    return np.where(np.isfinite(delta) | np.isinf(value), direct, fallback)


@dataclass(frozen=True)
class BinormalROCPoint:
    threshold: FloatArray
    auc: FloatArray
    diagnostic: DiagnosticPopulation


@dataclass(frozen=True, init=False)
class BinormalROC:
    """Independent Gaussian marker distributions; high marker values test positive."""

    control_mean: FloatArray
    control_sd: FloatArray
    disease_mean: FloatArray
    disease_sd: FloatArray

    def __init__(
        self,
        control_mean: ArrayLike = -1,
        control_sd: ArrayLike = 1,
        disease_mean: ArrayLike = 1,
        disease_sd: ArrayLike = 1,
    ):
        values = np.broadcast_arrays(
            *[
                finite(x, name)
                for x, name in (
                    (control_mean, "control_mean"),
                    (control_sd, "control_sd"),
                    (disease_mean, "disease_mean"),
                    (disease_sd, "disease_sd"),
                )
            ]
        )
        if np.any(values[1] <= 0) or np.any(values[3] <= 0):
            raise ValueError("marker standard deviations must be positive")
        for name, value in zip(
            ("control_mean", "control_sd", "disease_mean", "disease_sd"), values, strict=True
        ):
            object.__setattr__(self, name, _owned(value))

    @property
    def auc(self) -> FloatArray:
        scale = np.maximum(self.control_sd, self.disease_sd)
        denominator = np.hypot(self.control_sd / scale, self.disease_sd / scale)
        return ndtr(_standardize(self.disease_mean, self.control_mean, scale) / denominator)

    def at_threshold(
        self, threshold: ArrayLike, *, prevalence: ArrayLike = 0.5, population: ArrayLike = 10000
    ) -> BinormalROCPoint:
        """Evaluate thresholds; +/- infinity give the all-negative/all-positive limits."""
        cut = np.asarray(threshold, dtype=float)
        if np.any(np.isnan(cut)):
            raise ValueError("threshold must not contain NaN")
        zc = _standardize(cut, self.control_mean, self.control_sd)
        zd = _standardize(cut, self.disease_mean, self.disease_sd)
        diagnostic = _from_logs(
            log_ndtr(-zd), log_ndtr(zc), log_ndtr(-zc), log_ndtr(zd), prevalence, population
        )
        shape = diagnostic.prevalence.shape
        return BinormalROCPoint(
            _owned(np.broadcast_to(cut, shape)),
            _owned(np.broadcast_to(self.auc, shape)),
            diagnostic,
        )

    def at_percentile(
        self, percentile: ArrayLike, *, prevalence: ArrayLike = 0.5, population: ArrayLike = 10000
    ) -> BinormalROCPoint:
        """Choose a control-distribution quantile; percentile is a fraction in [0,1]."""
        p = _probability(percentile, "percentile")
        with np.errstate(over="ignore"):
            cut = self.control_mean + self.control_sd * ndtri(p)
        if np.any(~np.isfinite(cut) & (p > 0) & (p < 1)):
            raise ArithmeticError("finite control percentile threshold is not representable")
        achieved = ndtr(_standardize(cut, self.control_mean, self.control_sd))
        if np.any(np.abs(achieved - p) > 2e-10):
            raise ArithmeticError("requested percentile cannot be resolved at this marker scale")
        return self.at_threshold(cut, prevalence=prevalence, population=population)

    def curve(self, false_positive_rate: ArrayLike) -> FloatArray:
        """Sensitivity at specified FPR values, without subtracting FPR from one."""
        fpr = _probability(false_positive_rate, "false_positive_rate")
        scale = np.maximum(self.control_sd, self.disease_sd)
        shift = _standardize(self.disease_mean, self.control_mean, scale)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            z = (shift + (self.control_sd / scale) * ndtri(fpr)) / (self.disease_sd / scale)
            sensitivity = np.where(fpr == 0, 0, np.where(fpr == 1, 1, ndtr(z)))
        if np.any(np.isnan(sensitivity)):
            raise ArithmeticError("ROC sensitivity cannot be resolved at these scales")
        return sensitivity

    def densities(self, marker: ArrayLike) -> FloatArray:
        """Control and disease densities on a final axis of length two."""
        x = finite(marker, "marker")
        zc = _standardize(x, self.control_mean, self.control_sd)
        zd = _standardize(x, self.disease_mean, self.disease_sd)
        with np.errstate(over="ignore", under="ignore"):
            control = np.exp(-0.5 * zc**2 - 0.5 * np.log(2 * np.pi) - np.log(self.control_sd))
            disease = np.exp(-0.5 * zd**2 - 0.5 * np.log(2 * np.pi) - np.log(self.disease_sd))
        return _owned(np.stack(np.broadcast_arrays(control, disease), axis=-1))
