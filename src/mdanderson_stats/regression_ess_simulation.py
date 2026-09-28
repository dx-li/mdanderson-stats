"""Bounded covariate simulation for BayesESS regression effective sample size.

This ports the source ``ESS_RegressionCalc`` information-path calculation. It
simulates design covariates only (not responses), averages cumulative expected
curvature paths over replicates, and linearly interpolates the first crossing
of the prior-information gain. The native source's two fixed subvectors are
replaced by arbitrary zero-based parameter subsets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from ._validation import FloatArray, scalar
from .parameter_distribution import ParameterDistribution
from .regression_ess import _gain

_MAX_COVARIATE_CELLS = 2_000_000
_MAX_PATH_CELLS = 200_000
_MAX_WORK = 20_000_000


def _frozen(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _ordinary_information(log_values: ArrayLike, name: str) -> FloatArray:
    logs = np.asarray(log_values, dtype=float)
    if np.any(np.isnan(logs)) or np.any(np.isposinf(logs)):
        raise ArithmeticError(f"{name} is not representable")
    with np.errstate(over="ignore", under="ignore"):
        values = np.exp(logs)
    if not np.isfinite(values).all():
        raise ArithmeticError(f"{name} exceeds float64 range")
    return _frozen(values)


def _ordinary_signed_curvature(sign: ArrayLike, log_abs: ArrayLike, name: str) -> FloatArray:
    signs = np.asarray(sign, dtype=float)
    logs = np.asarray(log_abs, dtype=float)
    if np.any(np.isnan(logs)) or np.any(np.isposinf(logs)):
        raise ArithmeticError(f"{name} is not representable")
    with np.errstate(over="ignore", under="ignore"):
        values = signs * np.exp(logs)
    if not np.isfinite(values).all():
        raise ArithmeticError(f"{name} exceeds float64 range")
    return _frozen(values)


def _prior_curvatures(
    priors: tuple[ParameterDistribution, ...], inflation: float
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    prior_sign = np.empty(len(priors), dtype=float)
    prior_log_abs = np.empty(len(priors), dtype=float)
    epsilon_sign = np.empty(len(priors), dtype=float)
    epsilon_log_abs = np.empty(len(priors), dtype=float)
    for i, distribution in enumerate(priors):
        if distribution.family == "normal":
            log_variance = np.log(distribution.parameter2)
            prior_sign[i], prior_log_abs[i] = 1.0, -log_variance
            epsilon_sign[i], epsilon_log_abs[i] = 1.0, -log_variance - np.log(inflation)
        else:
            shape, scale = distribution.parameter1, distribution.parameter2
            prior_factor = shape - 1.0
            epsilon_factor = shape / inflation - 1.0
            common = -2.0 * (np.log(shape) + np.log(scale))
            prior_sign[i] = np.sign(prior_factor)
            prior_log_abs[i] = (
                -np.inf if prior_factor == 0.0 else np.log(abs(prior_factor)) + common
            )
            epsilon_sign[i] = np.sign(epsilon_factor)
            epsilon_log_abs[i] = (
                -np.inf if epsilon_factor == 0.0 else np.log(abs(epsilon_factor)) + common
            )
    gain = _gain(priors, inflation)
    if not np.isfinite(gain).all():
        raise ArithmeticError("prior information gain is not representable")
    return prior_sign, prior_log_abs, epsilon_sign, epsilon_log_abs, gain


@dataclass(frozen=True)
class RegressionESSTrigger:
    """First crossing result for the whole prior or a selected parameter subset."""

    indices: tuple[int, ...]
    status: Literal["exact", "interpolated", "not_reached"]
    estimate: float | None
    target_log_information_gain: float
    lower_patients: int | None
    upper_patients: int | None
    lower_log_information: float
    upper_log_information: float | None


@dataclass(frozen=True)
class RegressionESSSimulation:
    """Replicate-averaged regression curvature paths and ESS crossing access."""

    model: Literal["normal", "logistic"]
    parameter_names: tuple[str, ...]
    log_mean_cumulative_information: FloatArray
    prior_information_sign: FloatArray
    log_abs_prior_information: FloatArray
    epsilon_information_sign: FloatArray
    log_abs_epsilon_information: FloatArray
    log_prior_information_gain: FloatArray
    covariate_draws: FloatArray
    replicates: int
    max_patients: int
    variance_inflation: float

    @property
    def mean_cumulative_information(self) -> FloatArray:
        """Ordinary-scale mean path; raises if any value exceeds float64 range."""
        return _ordinary_information(
            self.log_mean_cumulative_information, "mean cumulative information"
        )

    @property
    def prior_information(self) -> FloatArray:
        """Ordinary-scale prior curvature; log/sign fields remain authoritative."""
        return _ordinary_signed_curvature(
            self.prior_information_sign, self.log_abs_prior_information, "prior information"
        )

    @property
    def epsilon_prior_information(self) -> FloatArray:
        """Ordinary-scale epsilon curvature; log/sign fields remain authoritative."""
        return _ordinary_signed_curvature(
            self.epsilon_information_sign,
            self.log_abs_epsilon_information,
            "epsilon-prior information",
        )

    def crossing(self, indices: ArrayLike | None = None) -> RegressionESSTrigger:
        """Return the first exact/interpolated crossing for chosen coefficient indices.

        ``indices=None`` means all model parameters. The first exact point on a
        flat target plateau is used; no extrapolation is performed when the
        simulated patient cap does not reach the target.
        """
        p = len(self.parameter_names)
        if indices is None:
            selected = np.arange(p, dtype=np.int64)
        else:
            raw = np.asarray(indices)
            if (
                raw.ndim != 1
                or raw.size == 0
                or raw.size > p
                or raw.dtype.kind not in "iu"
                or np.any(raw < 0)
                or np.any(raw >= p)
                or np.unique(raw).size != raw.size
            ):
                raise ValueError("indices must be distinct zero-based parameter indices")
            selected = raw.astype(np.int64, copy=False)
        log_target = float(logsumexp(self.log_prior_information_gain[selected]))
        log_path = logsumexp(self.log_mean_cumulative_information[:, selected], axis=1)
        crossed = np.flatnonzero(log_path >= log_target)
        if crossed.size == 0:
            last = self.max_patients
            return RegressionESSTrigger(
                tuple(int(i) for i in selected),
                "not_reached",
                None,
                log_target,
                last,
                None,
                float(log_path[-1]),
                None,
            )
        upper = int(crossed[0])
        if log_path[upper] == log_target or upper == 0:
            return RegressionESSTrigger(
                tuple(int(i) for i in selected),
                "exact",
                float(upper),
                log_target,
                upper,
                upper,
                float(log_path[upper]),
                float(log_path[upper]),
            )
        lower = upper - 1
        lower_log = float(log_path[lower])
        upper_log = float(log_path[upper])
        # Algebraically equivalent to linear interpolation on information
        # values, but remains stable when both information values are huge.
        fraction = (
            np.exp(log_target - upper_log)
            * (-np.expm1(lower_log - log_target))
            / (-np.expm1(lower_log - upper_log))
        )
        estimate = lower + float(fraction)
        if not np.isfinite(estimate) or not 0.0 <= fraction <= 1.0:
            raise ArithmeticError("information crossing interpolation is invalid")
        return RegressionESSTrigger(
            tuple(int(i) for i in selected),
            "interpolated",
            estimate,
            log_target,
            lower,
            upper,
            lower_log,
            upper_log,
        )

    @property
    def whole_model(self) -> RegressionESSTrigger:
        return self.crossing()


def simulate_regression_ess(
    model: Literal["normal", "logistic"],
    coefficient_priors: Sequence[ParameterDistribution],
    *,
    precision_prior: ParameterDistribution | None = None,
    max_patients: int = 100,
    replicates: int = 1000,
    variance_inflation: float = 10_000,
    rng: np.random.Generator | None = None,
    covariate_draws: ArrayLike | None = None,
) -> RegressionESSSimulation:
    """Simulate prior-curvature paths under uniform regression covariates.

    ``coefficient_priors`` includes the intercept as its first element. For a
    normal model, a gamma ``precision_prior`` is also required; its native-R
    rate convention is represented by ``ParameterDistribution``'s scale as
    the reciprocal rate. Supplied ``covariate_draws`` must exactly match
    ``(replicates, max_patients, non_intercept_covariates)`` and lie in
    ``[-1, 1]``. Generated covariates use the supplied NumPy Generator or a
    fresh default generator. Paths average cumulative expected information,
    including no response-generation randomness.
    """
    if model not in ("normal", "logistic"):
        raise ValueError("model must be 'normal' or 'logistic'")
    try:
        prior_count = len(coefficient_priors)
    except TypeError as error:
        raise TypeError("coefficient_priors must be a bounded sequence") from error
    if not 1 <= prior_count <= 11:
        raise ValueError("require 1..11 normal/gamma coefficient priors including intercept")
    priors = tuple(coefficient_priors)
    if (
        not 1 <= len(priors) <= 11
        or any(not isinstance(item, ParameterDistribution) for item in priors)
        or any(item.family not in ("normal", "gamma") for item in priors)
    ):
        raise ValueError("require 1..11 normal/gamma coefficient priors including intercept")
    if model == "normal":
        if (
            not isinstance(precision_prior, ParameterDistribution)
            or precision_prior.family != "gamma"
        ):
            raise ValueError("normal regression requires a gamma precision_prior")
        all_priors = (*priors, precision_prior)
    else:
        if precision_prior is not None:
            raise ValueError("precision_prior is only used for normal regression")
        all_priors = priors
    patient_value = scalar(max_patients, "max_patients")
    replicate_value = scalar(replicates, "replicates")
    if patient_value != np.floor(patient_value) or replicate_value != np.floor(replicate_value):
        raise ValueError("max_patients and replicates must be integers")
    patients, repetitions = int(patient_value), int(replicate_value)
    if not 1 <= patients <= 10_000 or not 1 <= repetitions <= 10_000:
        raise ValueError("max_patients and replicates must each lie in [1,10000]")
    inflation = scalar(variance_inflation, "variance_inflation")
    if inflation <= 1 or inflation > 1e12:
        raise ValueError("variance_inflation must lie in (1,1e12]")
    covariate_count = len(priors) - 1
    cov_cells = repetitions * patients * covariate_count
    parameter_count = len(all_priors)
    path_cells = (patients + 1) * parameter_count
    work = repetitions * patients * parameter_count
    if cov_cells > _MAX_COVARIATE_CELLS:
        raise ValueError("replicate covariate draws exceed the 2,000,000-cell limit")
    if path_cells > _MAX_PATH_CELLS:
        raise ValueError("cumulative information path exceeds the 200,000-cell limit")
    if work > _MAX_WORK:
        raise ValueError("regression ESS simulation exceeds the 20,000,000-work-unit limit")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None")

    coefficient_means = np.array([prior.mean for prior in priors], dtype=float)
    if not np.isfinite(coefficient_means).all():
        raise ValueError("coefficient prior means must be finite")
    if model == "logistic":
        with np.errstate(over="ignore"):
            maximum_abs_eta = float(np.sum(np.abs(coefficient_means), dtype=np.longdouble))
        if not np.isfinite(maximum_abs_eta):
            raise ArithmeticError("logistic predictor range from prior means is not representable")
    if model == "normal":
        assert precision_prior is not None
        mean_precision = precision_prior.mean
        if not np.isfinite(mean_precision) or mean_precision <= 0:
            raise ValueError("precision prior mean must be positive and finite")
    (
        prior_sign,
        prior_log_abs,
        epsilon_sign,
        epsilon_log_abs,
        log_gain,
    ) = _prior_curvatures(tuple(all_priors), inflation)

    if covariate_draws is None:
        generator = np.random.default_rng() if rng is None else rng
        covariates = generator.uniform(-1.0, 1.0, size=(repetitions, patients, covariate_count))
    else:
        shape = np.shape(covariate_draws)
        if shape != (repetitions, patients, covariate_count):
            raise ValueError(
                "covariate_draws must match (replicates, max_patients, non-intercept covariates)"
            )
        raw = np.asarray(covariate_draws)
        if np.iscomplexobj(raw):
            raise ValueError("covariate_draws must be real")
        covariates = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(covariates).all() or np.any(np.abs(covariates) > 1.0):
            raise ValueError("covariate_draws must be finite values in [-1,1]")
        covariates = covariates.copy()

    log_sum_paths = np.full((patients + 1, parameter_count), -np.inf, dtype=float)
    log_sum_paths[0] = -np.inf
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for replicate in range(repetitions):
            design = np.column_stack((np.ones(patients, dtype=float), covariates[replicate]))
            if model == "logistic":
                eta = design @ coefficient_means
                if not np.isfinite(eta).all():
                    raise ArithmeticError("logistic predictor at prior means is not finite")
                log_variance = -np.logaddexp(0.0, eta) - np.logaddexp(0.0, -eta)
                log_increment = log_variance[:, None] + 2.0 * np.log(np.abs(design))
            else:
                assert precision_prior is not None
                log_increment = np.empty((patients, parameter_count), dtype=float)
                log_increment[:, :-1] = np.log(mean_precision) + 2.0 * np.log(np.abs(design))
                log_increment[:, -1] = np.log(0.5) - 2.0 * np.log(mean_precision)
            if np.isnan(log_increment).any() or np.isposinf(log_increment).any():
                raise ArithmeticError("per-patient information is not representable")
            log_cumulative = np.logaddexp.accumulate(log_increment, axis=0)
            log_path = np.empty((patients + 1, parameter_count), dtype=float)
            log_path[0] = -np.inf
            log_path[1:] = log_cumulative
            log_sum_paths = np.logaddexp(log_sum_paths, log_path)

    log_mean_paths = log_sum_paths - np.log(repetitions)
    names = tuple(f"coefficient_{i}" for i in range(len(priors)))
    if model == "normal":
        names += ("precision",)
    return RegressionESSSimulation(
        model,
        names,
        _frozen(log_mean_paths),
        _frozen(prior_sign),
        _frozen(prior_log_abs),
        _frozen(epsilon_sign),
        _frozen(epsilon_log_abs),
        _frozen(log_gain),
        _frozen(covariates),
        repetitions,
        patients,
        inflation,
    )
