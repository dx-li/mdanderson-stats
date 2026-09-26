"""Forward power calculations for STPLAN censored-survival procedures."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri
from scipy.stats import poisson

from ._validation import FloatArray, scalar
from .stplan_continuous import (
    _alpha,
    _inputs,
    _power,
    stplan_exponential_one_sample_power,
    stplan_exponential_two_sample_power,
)
from .survival_sample_size import exponential_event_probability

_MAX_SUPPORT_TERMS = 200_000


def stplan_censored_exponential_one_sample_power(
    null_mean: ArrayLike,
    alternative_mean: ArrayLike,
    accrual_rate: ArrayLike,
    accrual_duration: ArrayLike,
    followup_duration: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
    tail_tolerance: float = 1e-10,
) -> FloatArray:
    """Poisson-event mixture approximation for one-sample censored survival.

    The event count has mean ``accrual_rate * accrual_duration`` times the
    alternative event probability under uniform accrual and fixed follow-up.
    Zero events are nonrejections. The upper Poisson tail omitted from the sum
    is at most ``tail_tolerance``; this is an event-count approximation, not an
    exact censored-trial operating characteristic.
    """
    _alpha(alpha, sides)
    tail = scalar(tail_tolerance, "tail_tolerance")
    if not 0 < tail < 0.01:
        raise ValueError("tail_tolerance must lie strictly between 0 and 0.01")
    null, alternative, accrual_rate, accrual, followup = _inputs(
        (null_mean, "null_mean"),
        (alternative_mean, "alternative_mean"),
        (accrual_rate, "accrual_rate"),
        (accrual_duration, "accrual_duration"),
        (followup_duration, "followup_duration"),
    )
    if (
        np.any(null <= 0)
        or np.any(alternative <= 0)
        or np.any(accrual_rate <= 0)
        or np.any(accrual <= 0)
        or np.any(followup < 0)
    ):
        raise ValueError("means and accrual values must be positive; followup must be nonnegative")
    probability = exponential_event_probability(1 / alternative, accrual, followup)
    with np.errstate(over="ignore"):
        event_mean = accrual_rate * accrual * probability
    if np.any(~np.isfinite(event_mean)):
        raise ArithmeticError("expected event count is not representable")
    upper = np.maximum(0.0, np.ceil(poisson.isf(tail, event_mean)))
    if np.any(~np.isfinite(upper) | (upper >= 2**53)):
        raise ValueError("Poisson event-count cutoff is not representable")
    while np.any(poisson.sf(upper, event_mean) > tail):
        needs_room = poisson.sf(upper, event_mean) > tail
        upper = np.where(needs_room, upper + 1, upper)
        if np.any(upper >= 2**53):
            raise ArithmeticError("could not bound the omitted Poisson tail")
    if np.any(upper + 1 > _MAX_SUPPORT_TERMS):
        raise ValueError("survival event mixture exceeds 200000 support terms")
    work = sum(int(value) for value in upper.flat)
    if work > _MAX_SUPPORT_TERMS:
        raise ValueError("survival event mixture exceeds 200000 total support terms")
    if event_mean.size == 0:
        return np.empty(event_mean.shape, dtype=float)

    # Counts start at one because the source exponential test has no useful
    # rejection region with zero observed deaths.
    flat_upper = upper.ravel().astype(np.int64)
    lengths = flat_upper.copy()
    case_index: NDArray[np.int64] = np.repeat(np.arange(lengths.size), lengths)
    starts: NDArray[np.int64] = np.repeat(np.cumsum(lengths, dtype=np.int64) - lengths, lengths)
    counts: NDArray[np.int64] = np.arange(work, dtype=np.int64) - starts + 1
    flat_mu = event_mean.ravel()
    flat_ratio = alternative.ravel() / null.ravel()
    if np.any(~np.isfinite(flat_ratio) | (flat_ratio <= 0)):
        raise ArithmeticError("alternative-to-null mean ratio is not representable")
    result = np.zeros(flat_mu.size, dtype=float)
    for start in range(0, work, 2048):
        stop = min(start + 2048, work)
        idx = case_index[start:stop]
        k = counts[start:stop]
        conditional = stplan_exponential_one_sample_power(
            flat_ratio[idx], k.astype(float), alpha=alpha, sides=sides
        )
        weights = poisson.pmf(k, flat_mu[idx])
        np.add.at(result, idx, weights * conditional)
    return _power(result.reshape(event_mean.shape))


def stplan_george_desu_survival_power(
    mean1: ArrayLike,
    mean2: ArrayLike,
    accrual_rate: ArrayLike,
    accrual_duration: ArrayLike,
    followup_duration: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """George–Desu F approximation using expected deaths under each arm."""
    _alpha(alpha, sides)
    m1, m2, rate, accrual, followup = _inputs(
        (mean1, "mean1"),
        (mean2, "mean2"),
        (accrual_rate, "accrual_rate"),
        (accrual_duration, "accrual_duration"),
        (followup_duration, "followup_duration"),
    )
    if np.any(m1 <= 0) or np.any(m2 <= 0) or np.any(rate <= 0) or np.any(accrual <= 0):
        raise ValueError("means and accrual values must be positive")
    if np.any(followup < 0):
        raise ValueError("followup_duration must be nonnegative")
    with np.errstate(over="ignore"):
        exposure = rate * accrual / 2
    deaths1 = exposure * exponential_event_probability(1 / m1, accrual, followup)
    deaths2 = exposure * exponential_event_probability(1 / m2, accrual, followup)
    scale = np.maximum(m1, m2)
    weight1, weight2 = m1 / scale, m2 / scale
    weight_sum = weight1 + weight2
    effective_size = (weight1 / weight_sum) * deaths1 + (weight2 / weight_sum) * deaths2
    if np.any(~np.isfinite(effective_size) | (effective_size <= 0)):
        raise ArithmeticError("effective death count is not representable")
    return stplan_exponential_two_sample_power(
        m1, m2, effective_size, effective_size, alpha=alpha, sides=sides
    )


def stplan_information_survival_power(
    mean1: ArrayLike,
    mean2: ArrayLike,
    accrual_rate: ArrayLike,
    accrual_duration: ArrayLike,
    followup_duration: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """Normal approximation using one quarter of the expected deaths as information."""
    a = _alpha(alpha, sides)
    m1, m2, rate, accrual, followup = _inputs(
        (mean1, "mean1"),
        (mean2, "mean2"),
        (accrual_rate, "accrual_rate"),
        (accrual_duration, "accrual_duration"),
        (followup_duration, "followup_duration"),
    )
    if np.any(m1 <= 0) or np.any(m2 <= 0) or np.any(rate <= 0) or np.any(accrual <= 0):
        raise ValueError("means and accrual values must be positive")
    if np.any(followup < 0):
        raise ValueError("followup_duration must be nonnegative")
    p1 = exponential_event_probability(1 / m1, accrual, followup)
    p2 = exponential_event_probability(1 / m2, accrual, followup)
    with np.errstate(over="ignore"):
        deaths = 0.5 * rate * accrual * (p1 + p2)
        log_ratio = np.log(m1) - np.log(m2)
        root_information = np.sqrt(deaths) / 2
    if np.any(~np.isfinite(root_information)):
        raise ArithmeticError("expected information is not representable")
    z = np.abs(log_ratio) * root_information + ndtri(a)
    return _power(ndtr(z))


def stplan_historical_survival_power(
    experimental_hazard: ArrayLike,
    control_hazard: ArrayLike,
    accrual_rate: ArrayLike,
    accrual_duration: ArrayLike,
    followup_duration: ArrayLike,
    historical_deaths: ArrayLike,
    historical_alive: ArrayLike,
    *,
    control_allocation: ArrayLike = 0.0,
    continued_followup: bool = True,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """Directional power against a fixed historical-control survival cohort.

    Lower experimental hazard is the favorable direction. Historical alive
    subjects contribute future control deaths only when continued_followup is
    true, matching STPLAN's historical-control model. STPLAN exposes only its
    one-sided rule; sides=2 is a Python extension that halves alpha while
    retaining the same favorable direction.
    """
    a = _alpha(alpha, sides)
    if not isinstance(continued_followup, (bool, np.bool_)):
        raise ValueError("continued_followup must be boolean")
    hre, hrc, rate, accrual, followup, old_deaths, old_alive, allocation = _inputs(
        (experimental_hazard, "experimental_hazard"),
        (control_hazard, "control_hazard"),
        (accrual_rate, "accrual_rate"),
        (accrual_duration, "accrual_duration"),
        (followup_duration, "followup_duration"),
        (historical_deaths, "historical_deaths"),
        (historical_alive, "historical_alive"),
        (control_allocation, "control_allocation"),
    )
    if (
        np.any(hre <= 0)
        or np.any(hrc <= 0)
        or np.any(rate <= 0)
        or np.any(accrual <= 0)
        or np.any(followup < 0)
        or np.any(old_deaths <= 0)
        or np.any(old_alive < 0)
        or np.any((allocation < 0) | (allocation >= 1))
    ):
        raise ValueError(
            "hazards/accrual/deaths must be positive; other counts and allocation invalid"
        )
    with np.errstate(over="ignore"):
        study_length = accrual + followup
        experimental_events = (
            (1 - allocation)
            * rate
            * accrual
            * exponential_event_probability(hre, accrual, followup)
        )
    if np.any(~np.isfinite(study_length) | (study_length <= 0)):
        raise ArithmeticError("study duration is not representable")
    if continued_followup:
        with np.errstate(over="ignore"):
            historical_future = old_alive * (-np.expm1(-hrc * study_length))
        new_control_events = (
            allocation * rate * accrual * exponential_event_probability(hrc, accrual, followup)
        )
        control_events = historical_future + new_control_events
    else:
        control_events = np.zeros_like(experimental_events)
    if np.any(
        ~np.isfinite(experimental_events)
        | (experimental_events <= 0)
        | ~np.isfinite(control_events)
        | (control_events < 0)
    ):
        raise ArithmeticError("expected study deaths are not representable")
    combined = old_deaths + control_events
    if np.any(~np.isfinite(combined) | (combined <= 0)):
        raise ArithmeticError("combined historical and future deaths are not representable")
    boundary_sd = np.sqrt(1 / combined + 1 / experimental_events)
    alternative_sd = np.sqrt((control_events / combined) / combined + 1 / experimental_events)
    effect = np.log(hrc) - np.log(hre)
    return _power(ndtr((effect + ndtri(a) * boundary_sd) / alternative_sd))


def _uniform_piecewise_event_probability(
    hazard_before: FloatArray,
    hazard_after: FloatArray,
    breakpoint: FloatArray,
    accrual: FloatArray,
    followup: FloatArray,
) -> FloatArray:
    """Average event probability for uniform entry over a piecewise hazard."""
    start = followup
    end = followup + accrual
    if np.any(~np.isfinite(end)):
        raise ArithmeticError("follow-up age interval is not representable")
    before_length = np.clip(breakpoint - start, 0, accrual)
    after_length = accrual - before_length

    def segment_event_area(
        length: FloatArray, hazard: FloatArray, cumulative: FloatArray
    ) -> FloatArray:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            x = hazard * length
            small = x < 1e-3
            xs = np.where(small, x, 0.0)
            increment = xs * (0.5 + xs * (-1 / 6 + xs * (1 / 24 + xs * (-1 / 120 + xs / 720))))
            safe_x = np.where(small, 1.0, x)
            increment = np.where(small, increment, 1 + np.expm1(-x) / safe_x)
            initial_event = -np.expm1(-cumulative)
            initial_survival = np.exp(-cumulative)
            mean_event = initial_event + initial_survival * increment
        return np.where(length > 0, mean_event, 0.0)

    cumulative_before = hazard_before * start
    cumulative_after = hazard_before * breakpoint + hazard_after * np.maximum(start - breakpoint, 0)
    before_average = segment_event_area(before_length, hazard_before, cumulative_before)
    after_average = segment_event_area(after_length, hazard_after, cumulative_after)
    return (before_length / accrual) * before_average + (after_length / accrual) * after_average


def stplan_piecewise_survival_power(
    hazard_before: ArrayLike,
    hazard_after: ArrayLike,
    change_time: ArrayLike,
    hazard_ratio: ArrayLike,
    accrual_rate: ArrayLike,
    accrual_duration: ArrayLike,
    followup_duration: ArrayLike,
    *,
    model_arm: str = "lower_hazard",
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """Piecewise-exponential information approximation for survival power.

    Uniform entry times range from zero to ``accrual_duration``; at analysis,
    their follow-up ages therefore range from ``followup_duration`` to
    ``followup_duration + accrual_duration``. ``model_arm`` states whether the
    supplied piecewise hazards belong to the lower- or higher-hazard arm; the
    comparison arm is obtained through ``hazard_ratio``. This integrates the
    correctly normalized survival curve, unlike the native routine's
    dimensionally inconsistent integral. If both arms have zero hazards, the
    model contains zero event information and this approximation returns its
    achieved one-sided significance level.
    """
    a = _alpha(alpha, sides)
    if model_arm not in ("lower_hazard", "higher_hazard"):
        raise ValueError("model_arm must be 'lower_hazard' or 'higher_hazard'")
    h1, h2, change, hr, rate, accrual, followup = _inputs(
        (hazard_before, "hazard_before"),
        (hazard_after, "hazard_after"),
        (change_time, "change_time"),
        (hazard_ratio, "hazard_ratio"),
        (accrual_rate, "accrual_rate"),
        (accrual_duration, "accrual_duration"),
        (followup_duration, "followup_duration"),
    )
    if (
        np.any(h1 < 0)
        or np.any(h2 < 0)
        or np.any(change < 0)
        or np.any(hr < 1)
        or np.any(rate <= 0)
        or np.any(accrual <= 0)
        or np.any(followup < 0)
    ):
        raise ValueError("hazards/change time must be nonnegative; ratio >= 1 and accrual positive")
    if model_arm == "lower_hazard":
        lower1, lower2 = h1, h2
        with np.errstate(over="ignore"):
            higher1, higher2 = h1 * hr, h2 * hr
    else:
        higher1, higher2 = h1, h2
        lower1, lower2 = h1 / hr, h2 / hr
    if np.any(~np.isfinite(higher1) | ~np.isfinite(higher2)):
        raise ArithmeticError("comparison hazard is not representable")
    lower_pd = _uniform_piecewise_event_probability(lower1, lower2, change, accrual, followup)
    higher_pd = _uniform_piecewise_event_probability(higher1, higher2, change, accrual, followup)
    deaths = 0.5 * rate * accrual * (lower_pd + higher_pd)
    if np.any(~np.isfinite(deaths) | (deaths < 0)):
        raise ArithmeticError("expected death information is not representable")
    z = np.log(hr) * np.sqrt(deaths / 4) + ndtri(a)
    return _power(ndtr(z))
