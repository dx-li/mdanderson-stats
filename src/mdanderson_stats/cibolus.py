"""CiBolus time-to-response, response-dependent toxicity and prediction model."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from ._cdflib import _freeze
from ._validation import FloatArray

_MAX_GRID_CELLS = 200_000
_MAX_RECORDS = 200


def _real(value: ArrayLike, name: str, maximum: int = _MAX_GRID_CELLS) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.size > maximum:
        raise ValueError(f"{name} must be bounded real numeric data")
    result = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _scalar(value: object, name: str) -> float:
    answer = _real(value, name, 1)
    if answer.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    return float(answer)


def _log_softplus(value: float) -> float:
    """log(log(1+exp(value))) without losing the far lower tail."""
    return value if value < -36 else float(np.log(np.logaddexp(0.0, value)))


def _log_expm1(value: float) -> float:
    if value <= 0:
        raise ValueError("log(expm1(x)) requires x > 0")
    if value < 1e-5:
        return float(np.log(value) + np.log1p(value / 2 + value * value / 6))
    return float(value + np.log1p(-np.exp(-value)))


def _log1mexp(value: float) -> float:
    """log(1-exp(value)) for value <= 0."""
    if value > 0:
        raise ValueError("log1mexp requires a nonpositive argument")
    if value == 0:
        return -np.inf
    return float(np.log1p(-np.exp(value)) if value < -np.log(2) else np.log(-np.expm1(value)))


def cibolus_parameter_names() -> tuple[str, ...]:
    """Coordinates are log(alpha0..alpha5,beta0..beta4), in paper order."""
    return tuple(
        f"log_{name}{index}"
        for name, count in (("alpha", 6), ("beta", 5))
        for index in range(count)
    )


def _physical(log_parameters: ArrayLike) -> tuple[FloatArray, FloatArray]:
    values = _real(log_parameters, "log_parameters", 11)
    if values.shape != (11,):
        raise ValueError("log_parameters must contain eleven coordinates")
    with np.errstate(over="ignore", under="ignore"):
        physical = np.exp(values)
    if np.any(~np.isfinite(physical)) or np.any(physical == 0):
        raise ArithmeticError("positive model parameters are outside floating-point range")
    return physical[:6], physical[6:]


@dataclass(frozen=True)
class CiBolusPrior:
    """Independent normal priors on the eleven log-parameter coordinates."""

    mean: FloatArray
    sd: FloatArray

    def __init__(self, mean: ArrayLike, sd: ArrayLike) -> None:
        m, s = _real(mean, "prior mean", 11), _real(sd, "prior sd", 11)
        if m.shape != (11,) or s.shape != (11,) or np.any(s < 0):
            raise ValueError("prior mean and nonnegative SD must each have length eleven")
        object.__setattr__(self, "mean", _freeze(m))
        object.__setattr__(self, "sd", _freeze(s))


def cibolus_published_prior() -> CiBolusPrior:
    """Study-specific prior reported in the paper's illustration (SD 9 each)."""
    return CiBolusPrior(
        [-1.04, -1.60, -7.25, -4.74, -2.85, 2.37, -6.10, -3.79, -7.05, -5.42, -7.88],
        [9.0] * 11,
    )


@dataclass(frozen=True)
class CiBolusObservation:
    """One patient's response interval/type and binary toxicity outcome.

    ``kind`` is ``bolus``, ``exact``, ``interval`` or ``failure``. Bolus and
    failure records do not take a time; exact records use ``time``; interval
    records use ``lower`` and ``upper`` for (lower, upper].
    """

    concentration: float
    bolus_fraction: float
    kind: str
    toxicity: bool
    time: float | None = None
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        concentration = _scalar(self.concentration, "concentration")
        bolus = _scalar(self.bolus_fraction, "bolus_fraction")
        if concentration <= 0 or not 0 <= bolus <= 1:
            raise ValueError("concentration must be positive and bolus_fraction in [0,1]")
        if not isinstance(self.toxicity, (bool, np.bool_)):
            raise ValueError("toxicity must be boolean")
        if self.kind == "bolus" or self.kind == "failure":
            if self.time is not None or self.lower is not None or self.upper is not None:
                raise ValueError("bolus and failure records do not take response times")
        elif self.kind == "exact":
            time = _scalar(self.time, "time")
            if not 0 < time <= 1 or self.lower is not None or self.upper is not None:
                raise ValueError("exact response time must lie in (0,1]")
            object.__setattr__(self, "time", time)
        elif self.kind == "interval":
            lower, upper = _scalar(self.lower, "lower"), _scalar(self.upper, "upper")
            if not 0 <= lower < upper <= 1 or self.time is not None:
                raise ValueError("response interval must satisfy 0 <= lower < upper <= 1")
            object.__setattr__(self, "lower", lower)
            object.__setattr__(self, "upper", upper)
        else:
            raise ValueError("kind must be bolus, exact, interval or failure")
        object.__setattr__(self, "concentration", concentration)
        object.__setattr__(self, "bolus_fraction", bolus)
        object.__setattr__(self, "toxicity", bool(self.toxicity))


def _bolus_logmass(alpha: FloatArray, concentration: float, bolus: float) -> tuple[float, float]:
    if bolus == 0:
        return -np.inf, 0.0
    log_mass_hazard = np.log(alpha[0]) + alpha[1] * np.log(concentration) + alpha[2] * np.log(bolus)
    if log_mass_hazard > np.log(np.finfo(float).max):
        return 0.0, np.inf
    mass_hazard = float(np.exp(log_mass_hazard))
    if mass_hazard < 1e-8:
        log_mass = log_mass_hazard + np.log1p(-mass_hazard / 2 + mass_hazard**2 / 6)
    else:
        log_mass = _log1mexp(-mass_hazard)
    return log_mass, mass_hazard


def _continuous_hazard_log(
    alpha: FloatArray, concentration: float, bolus: float, time: float
) -> float:
    if time < 0:
        return -np.inf
    logc = np.log(concentration)
    logx = alpha[1] * logc
    if bolus == 1:
        logd = logx
    elif bolus == 0:
        if time == 0:
            logd = -np.inf
        else:
            logd = logx + np.log(time)
    else:
        logv = alpha[2] * np.log(bolus)
        log1mv = float(np.log(-np.expm1(logv)))
        logw = float(np.logaddexp(logv, log1mv + np.log(time))) if time > 0 else logv
        logd = logx + logw
    if not np.isfinite(logd) and logd != -np.inf:
        raise ArithmeticError("dose concentration predictor is not representable")
    if bolus == 0 and time == 0:
        if alpha[5] > 1:
            extra = -np.inf
        elif alpha[5] == 1:
            extra = float(np.log(alpha[4]))
        else:
            extra = np.inf
    else:
        z = np.log(alpha[4]) + alpha[5] * logd
        if z >= -36:
            extra = float(np.log(alpha[5]) - logd - np.logaddexp(0.0, -z))
        else:
            extra = float(np.log(alpha[4]) + np.log(alpha[5]) + (alpha[5] - 1) * logd)
    return float(np.logaddexp(np.log(alpha[3]), extra))


def _log_continuous_increment(
    alpha: FloatArray, concentration: float, bolus: float, lower: float, upper: float
) -> float:
    """Log integrated continuous hazard on ``(lower, upper]``."""
    duration = upper - lower
    if duration <= 0:
        return -np.inf
    if bolus == 1:
        return float(np.log(duration) + _continuous_hazard_log(alpha, concentration, bolus, 0.0))

    logx = alpha[1] * np.log(concentration)
    if bolus == 0:
        logv, log1mv = -np.inf, 0.0
    else:
        logv = alpha[2] * np.log(bolus)
        log1mv = float(np.log(-np.expm1(logv)))
    logwl = logv if lower == 0 else float(np.logaddexp(logv, log1mv + np.log(lower)))
    logwu = float(np.logaddexp(logv, log1mv + np.log(upper)))
    z_upper = np.log(alpha[4]) + alpha[5] * (logx + logwu)
    logk = logx + log1mv

    if np.isneginf(logwl):
        log_extra = _log_softplus(float(z_upper)) - logk
        if z_upper < -36:
            log_extra = np.log(alpha[4]) + (alpha[5] - 1) * logx + alpha[5] * logwu - log1mv
    else:
        log_ratio_arg = log1mv + np.log(duration) - logwl
        log_dz = np.log(alpha[5]) + _log_softplus(log_ratio_arg)
        if z_upper < -36:
            log_correction = log_dz if log_dz < -36 else _log1mexp(-float(np.exp(log_dz)))
            log_extra = (
                np.log(alpha[4])
                + (alpha[5] - 1) * logx
                + alpha[5] * logwu
                + log_correction
                - log1mv
            )
        elif log_dz < np.log(50.0):
            z_lower = np.log(alpha[4]) + alpha[5] * (logx + logwl)
            log_expm1_dz = log_dz if log_dz < -36 else _log_expm1(float(np.exp(log_dz)))
            log_term = -float(np.logaddexp(0.0, -z_lower)) + log_expm1_dz
            log_extra = _log_softplus(log_term) - logk
        else:
            if z_upper < -36:
                log_extra = (
                    np.log(alpha[4]) + (alpha[5] - 1) * logx + alpha[5] * logwu + log_dz - log1mv
                )
            elif z_lower > 36:
                log_extra = float(log_dz - logk)
            else:
                dz = float(np.exp(log_dz))
                z_lower = np.log(alpha[4]) + alpha[5] * (logx + logwl)
                z_upper = z_lower + dz
                difference = float(np.logaddexp(0.0, z_upper) - np.logaddexp(0.0, z_lower))
                if difference <= 0:
                    raise ArithmeticError(
                        "positive interval lost precision in its hazard increment"
                    )
                log_extra = float(np.log(difference) - logk)
    return float(np.logaddexp(np.log(alpha[3]) + np.log(duration), log_extra))


def _continuous_cumulative(
    alpha: FloatArray, concentration: float, bolus: float, time: float
) -> float:
    if time <= 0:
        return 0.0
    logvalue = _log_continuous_increment(alpha, concentration, bolus, 0.0, time)
    if logvalue > np.log(np.finfo(float).max):
        return np.inf
    with np.errstate(under="ignore"):
        return float(np.exp(logvalue))


def _response_terms(
    alpha: FloatArray, concentration: float, bolus: float, time: float
) -> tuple[float, float, float, float]:
    log_bolus, bolus_hazard = _bolus_logmass(alpha, concentration, bolus)
    continuous = _continuous_cumulative(alpha, concentration, bolus, time)
    log_survival = -bolus_hazard - continuous
    log_density = log_survival + _continuous_hazard_log(alpha, concentration, bolus, time)
    if time == 0:
        log_density = -np.inf
    if log_survival > 0 or np.isnan(log_survival):
        raise ArithmeticError("response survival probability is invalid")
    with np.errstate(under="ignore"):
        cdf = float(-np.expm1(log_survival))
    return log_bolus, log_survival, log_density, cdf


@dataclass(frozen=True)
class CiBolusResponse:
    bolus_probability: float
    cdf: FloatArray
    continuous_hazard: FloatArray
    cumulative_hazard: FloatArray
    log_survival: FloatArray


def cibolus_response(
    times: ArrayLike, concentration: float, bolus_fraction: float, log_parameters: ArrayLike
) -> CiBolusResponse:
    """Evaluate bolus response mass and continuous response-time quantities."""
    t = _real(times, "times", 10_000)
    concentration_value, bolus = (
        _scalar(concentration, "concentration"),
        _scalar(bolus_fraction, "bolus_fraction"),
    )
    if np.any((t < 0) | (t > 1)) or concentration_value <= 0 or not 0 <= bolus <= 1:
        raise ValueError("times must lie in [0,1], concentration positive and bolus in [0,1]")
    alpha, _ = _physical(log_parameters)
    bolus_log, bolus_hazard = _bolus_logmass(alpha, concentration_value, bolus)
    bolus_probability = float(np.exp(bolus_log)) if np.isfinite(bolus_log) else 0.0
    cdf = np.empty(t.shape)
    hazard = np.empty(t.shape)
    cumulative = np.empty(t.shape)
    survival = np.empty(t.shape)
    for index in np.ndindex(t.shape):
        time = float(t[index])
        cumulative[index] = _continuous_cumulative(alpha, concentration_value, bolus, time)
        survival[index] = -bolus_hazard - cumulative[index]
        cdf[index] = -np.expm1(survival[index]) if np.isfinite(survival[index]) else 1.0
        loghazard = _continuous_hazard_log(alpha, concentration_value, bolus, time)
        hazard[index] = 0.0 if loghazard < np.log(np.nextafter(0.0, 1.0)) else np.exp(loghazard)
    return CiBolusResponse(
        bolus_probability,
        _freeze(cdf),
        _freeze(hazard),
        _freeze(cumulative),
        _freeze(survival),
    )


def _toxicity_log_probability(
    beta: FloatArray,
    concentration: float,
    bolus: float,
    time: float,
    failure: bool,
    toxic: bool,
) -> float:
    logc = np.log(concentration)
    terms = [np.log(beta[0])]
    if bolus > 0:
        terms.append(np.log(beta[2]) + beta[1] * logc + np.log(bolus))
    if bolus < 1 and time > 0:
        terms.append(np.log(beta[3]) + beta[1] * logc + np.log1p(-bolus) + np.log(time))
    if failure:
        terms.append(np.log(beta[4]))
    log_exponent = float(logsumexp(terms))
    if log_exponent > np.log(np.finfo(float).max):
        return 0.0 if toxic else -np.inf
    exponent = float(np.exp(log_exponent))
    if toxic:
        if exponent < 1e-8:
            return float(log_exponent + np.log1p(-exponent / 2 + exponent**2 / 6))
        return _log1mexp(-exponent)
    return -exponent


def cibolus_toxicity(
    times: ArrayLike,
    concentration: float,
    bolus_fraction: float,
    log_parameters: ArrayLike,
    *,
    failure: bool = False,
) -> FloatArray:
    """Evaluate binary toxicity risk, optionally including the no-response effect."""
    t = _real(times, "times", 10_000)
    concentration_value, bolus = (
        _scalar(concentration, "concentration"),
        _scalar(bolus_fraction, "bolus_fraction"),
    )
    if np.any((t < 0) | (t > 1)) or concentration_value <= 0 or not 0 <= bolus <= 1:
        raise ValueError("invalid time, concentration or bolus fraction")
    if not isinstance(failure, (bool, np.bool_)):
        raise ValueError("failure must be boolean")
    if failure and np.any(t != 1):
        raise ValueError("failure toxicity is defined at the infusion endpoint time 1")
    _, beta = _physical(log_parameters)
    output = np.empty(t.shape)
    for index in np.ndindex(t.shape):
        log_not = _toxicity_log_probability(
            beta, concentration_value, bolus, float(t[index]), failure, False
        )
        output[index] = -np.expm1(log_not)
    return _freeze(output)


@dataclass(frozen=True)
class CiBolusPrediction:
    """Dose-grid predictions; joint axes are concentration, bolus, category, toxicity."""

    joint: FloatArray
    response_probability: FloatArray
    expected_utility: FloatArray
    response_at_one: FloatArray
    toxicity_at_one_response: FloatArray
    toxicity_at_one_failure: FloatArray
    marginal_toxicity: FloatArray


def cibolus_predict(
    log_parameters: ArrayLike,
    concentrations: ArrayLike,
    bolus_fractions: ArrayLike,
    endpoints: ArrayLike,
    *,
    utility: ArrayLike,
) -> CiBolusPrediction:
    """Return response-category/toxicity probabilities and utility on a regimen grid."""
    alpha, beta = _physical(log_parameters)
    concentrations_value = _real(concentrations, "concentrations", 20)
    bolus_value = _real(bolus_fractions, "bolus_fractions", 20)
    endpoint_value = _real(endpoints, "endpoints", 20)
    utility_value = _real(utility, "utility", 100)
    if (
        concentrations_value.ndim != 1
        or not 1 <= concentrations_value.size <= 20
        or np.any(concentrations_value <= 0)
        or np.any(np.diff(concentrations_value) <= 0)
        or bolus_value.ndim != 1
        or not 1 <= bolus_value.size <= 20
        or np.any((bolus_value < 0) | (bolus_value > 1))
        or np.any(np.diff(bolus_value) <= 0)
        or endpoint_value.ndim != 1
        or not 1 <= endpoint_value.size <= 20
        or np.any((endpoint_value <= 0) | (endpoint_value > 1))
        or np.any(np.diff(endpoint_value) <= 0)
        or endpoint_value[-1] != 1
    ):
        raise ValueError("invalid concentration, bolus or endpoint grids")
    if (
        np.prod((concentrations_value.size, bolus_value.size, endpoint_value.size + 2, 2))
        > _MAX_GRID_CELLS
    ):
        raise ValueError("prediction grid exceeds 200,000 category cells")
    categories = endpoint_value.size + 2  # bolus, endpoint intervals, failure
    if utility_value.shape != (categories, 2):
        raise ValueError("utility must have one row per response category and two toxicity columns")
    joint = np.empty((concentrations_value.size, bolus_value.size, categories, 2))
    response = np.empty((concentrations_value.size, bolus_value.size, categories))
    utility_result = np.empty((concentrations_value.size, bolus_value.size))
    risk_one = np.empty_like(utility_result)
    toxicity_response = np.empty_like(utility_result)
    toxicity_failure = np.empty_like(utility_result)
    marginal_tox = np.empty_like(utility_result)
    for ci, concentration in enumerate(concentrations_value):
        for qi, bolus in enumerate(bolus_value):
            log_bolus, _, _, _ = _response_terms(alpha, float(concentration), float(bolus), 0.0)
            bolus_mass = 0.0 if log_bolus == -np.inf else float(np.exp(log_bolus))
            response[ci, qi, 0] = bolus_mass
            joint[ci, qi, 0, 1] = (
                bolus_mass * cibolus_toxicity([0.0], concentration, bolus, log_parameters)[0]
            )
            joint[ci, qi, 0, 0] = bolus_mass - joint[ci, qi, 0, 1]
            lower = 0.0
            for ei, upper in enumerate(endpoint_value):
                _, log_low, _, _ = _response_terms(alpha, float(concentration), float(bolus), lower)
                log_increment = _log_continuous_increment(
                    alpha, float(concentration), float(bolus), lower, float(upper)
                )
                if log_low == -np.inf:
                    log_mass = -np.inf
                elif log_increment == -np.inf:
                    log_mass = -np.inf
                elif log_increment > np.log(np.finfo(float).max):
                    log_mass = log_low
                else:
                    increment = float(np.exp(log_increment))
                    log_mass = log_low + _log1mexp(-increment)
                mass = 0.0 if log_mass == -np.inf else float(np.exp(log_mass))
                category = ei + 1
                response[ci, qi, category] = mass
                toxicity = float(cibolus_toxicity([upper], concentration, bolus, log_parameters)[0])
                joint[ci, qi, category, 1] = mass * toxicity
                joint[ci, qi, category, 0] = mass - joint[ci, qi, category, 1]
                lower = float(upper)
            _, log_failure, _, risk = _response_terms(
                alpha, float(concentration), float(bolus), 1.0
            )
            failure_mass = 0.0 if not np.isfinite(log_failure) else float(np.exp(log_failure))
            response[ci, qi, -1] = failure_mass
            tox_fail = float(
                cibolus_toxicity([1.0], concentration, bolus, log_parameters, failure=True)[0]
            )
            joint[ci, qi, -1, 1] = failure_mass * tox_fail
            joint[ci, qi, -1, 0] = failure_mass - joint[ci, qi, -1, 1]
            risk_one[ci, qi] = risk
            toxicity_response[ci, qi] = cibolus_toxicity(
                [1.0], concentration, bolus, log_parameters
            )[0]
            toxicity_failure[ci, qi] = tox_fail
            marginal_tox[ci, qi] = np.sum(joint[ci, qi, :, 1])
            utility_result[ci, qi] = np.sum(joint[ci, qi] * utility_value)
    if (
        np.any(~np.isfinite(joint))
        or np.any(joint < 0)
        or np.any(np.abs(joint.sum(axis=(-2, -1)) - 1) > 1e-10)
    ):
        raise ArithmeticError("predicted joint response/toxicity probabilities are invalid")
    if not np.all(np.isfinite(utility_result)):
        raise ArithmeticError("expected utility exceeds floating-point range")
    return CiBolusPrediction(
        *(
            _freeze(x)
            for x in (
                joint,
                response,
                utility_result,
                risk_one,
                toxicity_response,
                toxicity_failure,
                marginal_tox,
            )
        )
    )


def cibolus_loglikelihood(
    observations: tuple[CiBolusObservation, ...] | list[CiBolusObservation],
    log_parameters: ArrayLike,
) -> float:
    """Evaluate individual interval/exact/bolus/failure and toxicity likelihoods."""
    if not isinstance(observations, (tuple, list)) or len(observations) > _MAX_RECORDS:
        raise ValueError(f"observations must be at most {_MAX_RECORDS} records")
    if any(not isinstance(row, CiBolusObservation) for row in observations):
        raise ValueError("every observation must be a CiBolusObservation")
    alpha, beta = _physical(log_parameters)
    likelihood = 0.0
    for row in observations:
        if row.kind == "bolus":
            log_response = _bolus_logmass(alpha, row.concentration, row.bolus_fraction)[0]
            tox_time, failure = 0.0, False
        elif row.kind == "exact":
            assert row.time is not None
            _, log_response, log_density, _ = _response_terms(
                alpha, row.concentration, row.bolus_fraction, row.time
            )
            log_response = log_density
            tox_time, failure = row.time, False
        elif row.kind == "interval":
            assert row.lower is not None and row.upper is not None
            _, low, _, _ = _response_terms(alpha, row.concentration, row.bolus_fraction, row.lower)
            _, high, _, _ = _response_terms(alpha, row.concentration, row.bolus_fraction, row.upper)
            delta = high - low
            log_response = low + _log1mexp(delta)
            tox_time, failure = row.upper, False
        else:
            _, log_response, _, _ = _response_terms(
                alpha, row.concentration, row.bolus_fraction, 1.0
            )
            tox_time, failure = 1.0, True
        if log_response == -np.inf:
            return -np.inf
        toxicity_ll = _toxicity_log_probability(
            beta, row.concentration, row.bolus_fraction, tox_time, failure, row.toxicity
        )
        likelihood += log_response + toxicity_ll
        if not np.isfinite(likelihood):
            if likelihood == -np.inf:
                return -np.inf
            raise ArithmeticError("CiBolus log likelihood is not representable")
    return float(likelihood)
