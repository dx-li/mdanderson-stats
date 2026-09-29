"""BOP2-DC randomized two-arm exponential survival monitoring."""

from dataclasses import dataclass
from math import exp, isfinite, log, prod

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.special import gammainc, gammaincinv, gammaln

from ._bop2_dc_randomized_rules import randomized_dual_decisions
from ._validation import FloatArray, count, finite, scalar
from .beta_comparison import _cdf_from_logs
from .boin import _owned
from .bop2_dc_survival_trial import _check_vector_shape, _real_array

_LOG2 = log(2.0)
_MAX_SUBJECTS = 1_000
_MAX_MONITOR_CELLS = 64
_MAX_QUADRATURE_EVALUATIONS = 1_700_000
_QUAD_EVALUATIONS_PER_INTEGRAL = 21 * (2 * 300 - 1)
_MAX_CALENDAR_WORK = 30_000_000


@dataclass(frozen=True)
class BOP2DCRandomizedSurvivalState:
    """As-of sufficient statistics and tail probabilities (arm axis: control, treatment)."""

    total_n: NDArray[np.int64]
    control_n: NDArray[np.int64]
    treatment_n: NDArray[np.int64]
    control_events: NDArray[np.int64]
    treatment_events: NDArray[np.int64]
    control_exposure: FloatArray
    treatment_exposure: FloatArray
    posterior_shape: FloatArray
    posterior_scale: FloatArray
    posterior_lrv: FloatArray
    posterior_cmv: FloatArray
    absolute_error_lrv: FloatArray
    absolute_error_cmv: FloatArray
    decision: NDArray[np.str_]


@dataclass(frozen=True)
class BOP2DCRandomizedSurvivalTrial:
    """Fixed-allocation calendar replay through its first terminal scheduled look."""

    arm_assignments_observed: NDArray[np.int64]
    enrollment_times_observed: FloatArray
    calendar_times: FloatArray
    states: tuple[BOP2DCRandomizedSurvivalState, ...]
    enrolled: int
    decision: str


@dataclass(frozen=True)
class BOP2DCRandomizedSurvivalDesign:
    """Independent inverse-gamma mean-survival priors and fixed randomized allocation."""

    max_subjects: int
    median_lrv: float
    median_cmv: float
    lambda_lrv: float
    lambda_cmv: float
    gamma_lrv: float
    gamma_cmv: float
    control_prior: tuple[float, float]
    treatment_prior: tuple[float, float]
    arm_assignments: NDArray[np.int64]
    looks: NDArray[np.int64]
    graduate_at_interim: bool
    comparison_tolerance: float

    def monitor(
        self,
        control_events: ArrayLike,
        control_exposure: ArrayLike,
        control_n: ArrayLike,
        treatment_events: ArrayLike,
        treatment_exposure: ArrayLike,
        treatment_n: ArrayLike,
    ) -> BOP2DCRandomizedSurvivalState:
        """Update both inverse-gamma posteriors from event counts and total exposure."""
        raw = (
            (control_events, "control_events"),
            (control_exposure, "control_exposure"),
            (control_n, "control_n"),
            (treatment_events, "treatment_events"),
            (treatment_exposure, "treatment_exposure"),
            (treatment_n, "treatment_n"),
        )
        shapes = []
        for value, name in raw:
            if isinstance(value, np.ndarray):
                shape = value.shape
            elif isinstance(value, (list, tuple)):
                if len(value) > _MAX_MONITOR_CELLS or any(not np.isscalar(item) for item in value):
                    raise ValueError(f"{name} must be a bounded scalar or array")
                shape = (len(value),)
            else:
                shape = np.shape(value)
            cells = 1
            for size in shape:
                cells *= int(size)
            if cells > _MAX_MONITOR_CELLS:
                raise ValueError("monitor batch exceeds its cell budget")
            shapes.append(shape)
            if np.iscomplexobj(value):
                raise ValueError(f"{name} must be real-valued")

        try:
            batch_shape = np.broadcast_shapes(*shapes)
        except ValueError as exc:
            raise ValueError("arm sufficient statistics must broadcast") from exc
        batch_cells = prod(batch_shape)
        if batch_cells > _MAX_MONITOR_CELLS:
            raise ValueError("monitor batch exceeds its cell budget")
        quadrature_margins = int(self.median_lrv != 0) + int(self.median_cmv != 0)
        if (
            batch_cells * quadrature_margins * _QUAD_EVALUATIONS_PER_INTEGRAL
            > _MAX_QUADRATURE_EVALUATIONS
        ):
            raise ValueError("inverse-gamma comparison work exceeds its bound")

        d_c = count(control_events, "control_events")
        e_c = finite(control_exposure, "control_exposure")
        n_c = count(control_n, "control_n")
        d_e = count(treatment_events, "treatment_events")
        e_e = finite(treatment_exposure, "treatment_exposure")
        n_e = count(treatment_n, "treatment_n")
        d_c, e_c, n_c, d_e, e_e, n_e = np.broadcast_arrays(d_c, e_c, n_c, d_e, e_e, n_e)
        total_n = n_c + n_e
        if np.any(
            (d_c > n_c)
            | (d_e > n_e)
            | (total_n > self.max_subjects)
            | (e_c < 0)
            | (e_e < 0)
            | ((d_c > 0) & (e_c <= 0))
            | ((d_e > 0) & (e_e <= 0))
            | ((n_c == 0) & (e_c != 0))
            | ((n_e == 0) & (e_e != 0))
        ):
            raise ValueError("inconsistent arm event counts, sample sizes, or exposure")
        prefix_c = np.r_[0, np.cumsum(self.arm_assignments == 0)]
        prefix_e = np.r_[0, np.cumsum(self.arm_assignments == 1)]
        total_indices = total_n.astype(np.int64)
        if np.any((n_c != prefix_c[total_indices]) | (n_e != prefix_e[total_indices])):
            raise ValueError("arm sample sizes do not match the fixed allocation prefix")

        shape_c = self.control_prior[0] + d_c
        scale_c = self.control_prior[1] + e_c
        shape_e = self.treatment_prior[0] + d_e
        scale_e = self.treatment_prior[1] + e_e
        if np.any(
            ~np.isfinite(shape_c)
            | ~np.isfinite(scale_c)
            | ~np.isfinite(shape_e)
            | ~np.isfinite(scale_e)
            | (shape_c <= 0)
            | (shape_e <= 0)
            | (scale_c <= 0)
            | (scale_e <= 0)
        ):
            raise ArithmeticError("posterior inverse-gamma parameters are not representable")

        p_lrv = np.empty(batch_shape, dtype=np.float64)
        p_cmv = np.empty(batch_shape, dtype=np.float64)
        err_lrv = np.empty(batch_shape, dtype=np.float64)
        err_cmv = np.empty(batch_shape, dtype=np.float64)
        for index in np.ndindex(batch_shape):
            args = (
                float(shape_e[index]),
                float(scale_e[index]),
                float(shape_c[index]),
                float(scale_c[index]),
            )
            p_lrv[index], err_lrv[index] = _median_difference_probability(
                *args, self.median_lrv, self.comparison_tolerance
            )
            p_cmv[index], err_cmv[index] = _median_difference_probability(
                *args, self.median_cmv, self.comparison_tolerance
            )
        decision = randomized_dual_decisions(
            total_indices,
            p_lrv,
            p_cmv,
            err_lrv,
            err_cmv,
            max_subjects=self.max_subjects,
            looks=self.looks,
            lambda_lrv=self.lambda_lrv,
            lambda_cmv=self.lambda_cmv,
            gamma_lrv=self.gamma_lrv,
            gamma_cmv=self.gamma_cmv,
            graduate_at_interim=self.graduate_at_interim,
        )
        return BOP2DCRandomizedSurvivalState(
            _owned(total_indices.astype(np.int64)),
            _owned(n_c.astype(np.int64)),
            _owned(n_e.astype(np.int64)),
            _owned(d_c.astype(np.int64)),
            _owned(d_e.astype(np.int64)),
            _owned(e_c),
            _owned(e_e),
            _owned(np.stack((shape_c, shape_e), axis=-1)),
            _owned(np.stack((scale_c, scale_e), axis=-1)),
            _owned(p_lrv),
            _owned(p_cmv),
            _owned(err_lrv),
            _owned(err_cmv),
            decision,
        )


def _log_gamma_quantile(shape: float, probability: float) -> float:
    if probability <= 0:
        return -np.inf
    if probability >= 1:
        return np.inf
    value = float(gammaincinv(shape, probability))
    if value > 0 and isfinite(value):
        return log(value)
    if value == 0:
        log_value = (log(probability) + float(gammaln(shape + 1))) / shape
        if isfinite(log_value):
            return log_value
    raise ArithmeticError("gamma quantile is outside the supported numerical range")


def _gamma_cdf_from_log(shape: float, log_value: float) -> float:
    if log_value == -np.inf:
        return 0.0
    if log_value == np.inf:
        return 1.0
    if log_value < -700:
        leading = shape * log_value - float(gammaln(shape + 1))
        return 0.0 if leading < -745 else exp(leading)
    if log_value > 709:
        return 1.0
    return float(gammainc(shape, exp(log_value)))


def _median_difference_probability(
    shape_e: float,
    scale_e: float,
    shape_c: float,
    scale_c: float,
    margin: float,
    tolerance: float,
) -> tuple[float, float]:
    """Return P(log(2)*(mean_E-mean_C)>margin) and quadrature error estimate."""
    if margin == 0:
        log_e_cdf = -float(np.logaddexp(0.0, log(scale_c) - log(scale_e)))
        log_c_cdf = -float(np.logaddexp(0.0, log(scale_e) - log(scale_c)))
        return _cdf_from_logs(shape_e, shape_c, log_e_cdf, log_c_cdf), 0.0

    if not all(isfinite(float(gammaln(shape))) for shape in (shape_e, shape_c)):
        raise ArithmeticError("inverse-gamma posterior shape exceeds the supported range")
    log_rate_ratio = log(scale_e) - log(scale_c)
    log_abs_scaled_margin = log(abs(margin)) - log(_LOG2) - log(scale_c)
    negative = margin < 0

    def integrand(probability: float) -> float:
        log_y = _log_gamma_quantile(shape_c, probability)
        log_product = log_abs_scaled_margin + log_y
        if negative and log_product >= 0:
            return 1.0
        if negative:
            log_denominator = log(-np.expm1(log_product))
        else:
            log_denominator = float(np.logaddexp(0.0, log_product))
        log_threshold = log_rate_ratio + log_y - log_denominator
        return _gamma_cdf_from_log(shape_e, log_threshold)

    intervals = [(0.0, 1.0)]
    tail_probability = 0.0
    if negative:
        log_critical_y = -log_abs_scaled_margin
        critical_probability = _gamma_cdf_from_log(shape_c, log_critical_y)
        critical_probability = min(1.0, max(0.0, critical_probability))
        intervals = [(0.0, critical_probability)]
        tail_probability = 1.0 - critical_probability

    value = tail_probability
    estimated_error = 0.0
    for left, right in intervals:
        if right <= left:
            continue
        result = quad(
            integrand,
            left,
            right,
            epsabs=tolerance / 4,
            epsrel=tolerance / 4,
            limit=300,
            full_output=1,
        )
        if (
            len(result) != 3
            or not isfinite(float(result[0]))
            or not isfinite(float(result[1]))
            or result[1] > tolerance
        ):
            raise ArithmeticError("inverse-gamma median comparison quadrature failed")
        value += float(result[0])
        estimated_error += float(result[1])
    if not 0 <= value <= 1 or estimated_error > tolerance:
        raise ArithmeticError("inverse-gamma median comparison exceeded its error tolerance")
    return value, estimated_error


def _prior_pair(value: ArrayLike, name: str) -> tuple[float, float]:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) != 2 or any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must contain positive inverse-gamma shape and scale")
        shape = (len(value),)
    else:
        shape = np.shape(value)
    if shape != (2,) or np.iscomplexobj(value):
        raise ValueError(f"{name} must contain positive inverse-gamma shape and scale")
    pair = finite(value, name)
    if np.any(pair <= 0):
        raise ValueError(f"{name} must contain positive inverse-gamma shape and scale")
    return float(pair[0]), float(pair[1])


def _fixed_arm_assignments(value: ArrayLike, max_subjects: int) -> NDArray[np.int64]:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) != max_subjects or any(not np.isscalar(item) for item in value):
            raise ValueError("arm_assignments must have one 0/1 value per planned subject")
        shape = (len(value),)
    else:
        shape = np.shape(value)
    if shape != (max_subjects,) or np.iscomplexobj(value):
        raise ValueError("arm_assignments must have one 0/1 value per planned subject")
    assignments = count(value, "arm_assignments")
    if np.any(assignments > 1) or not np.any(assignments == 0) or not np.any(assignments == 1):
        raise ValueError("allocation must use both arms, coded 0=control and 1=treatment")
    return _owned(assignments.astype(np.int64))


def _look_schedule(value: ArrayLike, max_subjects: int) -> NDArray[np.int64]:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) > max_subjects or any(not np.isscalar(item) for item in value):
            raise ValueError("looks must increase and end at max_subjects")
        shape = (len(value),)
    else:
        shape = np.shape(value)
    if len(shape) != 1 or shape[0] < 1 or shape[0] > max_subjects:
        raise ValueError("looks must be a bounded vector ending at max_subjects")
    if np.iscomplexobj(value):
        raise ValueError("looks must be real-valued")
    looks = count(value, "looks")
    if (
        np.any(looks < 1)
        or np.any(looks > max_subjects)
        or looks[-1] != max_subjects
        or np.any(np.diff(looks) <= 0)
    ):
        raise ValueError("looks must increase strictly and end at max_subjects")
    return _owned(looks.astype(np.int64))


def bop2_dc_randomized_survival_design(
    max_subjects: int,
    median_lrv: float,
    median_cmv: float,
    *,
    control_prior: ArrayLike,
    treatment_prior: ArrayLike,
    arm_assignments: ArrayLike,
    looks: ArrayLike,
    lambda_lrv: float = 0.9,
    lambda_cmv: float = 0.5,
    gamma_lrv: float = 0.5,
    gamma_cmv: float = 0.5,
    graduate_at_interim: bool = False,
    comparison_tolerance: float = 1e-9,
) -> BOP2DCRandomizedSurvivalDesign:
    """Create randomized BOP2-DC with explicit IG(shape, scale) priors on mean time.

    Margins compare treatment minus control *median survival times*. Priors and
    allocation are caller-specified; the source does not define a universal ratio.
    """
    n_value = scalar(max_subjects, "max_subjects")
    n = int(n_value)
    if n_value != n or not 2 <= n <= _MAX_SUBJECTS:
        raise ValueError(f"max_subjects must be an integer in [2,{_MAX_SUBJECTS}]")
    lrv, cmv = scalar(median_lrv, "median_lrv"), scalar(median_cmv, "median_cmv")
    if not lrv < cmv:
        raise ValueError("require median_lrv < median_cmv")
    ll, lc = scalar(lambda_lrv, "lambda_lrv"), scalar(lambda_cmv, "lambda_cmv")
    gl, gc = scalar(gamma_lrv, "gamma_lrv"), scalar(gamma_cmv, "gamma_cmv")
    if not 0 < ll < 1 or not 0 < lc < 1 or not 0 <= gl <= 1 or not 0 <= gc <= 1:
        raise ValueError("lambda values must lie in (0,1) and gamma values in [0,1]")
    if not isinstance(graduate_at_interim, (bool, np.bool_)):
        raise ValueError("graduate_at_interim must be boolean")
    tolerance = scalar(comparison_tolerance, "comparison_tolerance")
    if not 1e-12 <= tolerance <= 1e-3:
        raise ValueError("comparison_tolerance must lie in [1e-12,1e-3]")
    prior_c = _prior_pair(control_prior, "control_prior")
    prior_e = _prior_pair(treatment_prior, "treatment_prior")
    assignments = _fixed_arm_assignments(arm_assignments, n)
    schedule = _look_schedule(looks, n)
    if ll * (int(schedule[0]) / n) ** gl == 0 or lc * (int(schedule[0]) / n) ** gc == 0:
        raise ArithmeticError("interim futility cutoff underflows; increase lambda")
    return BOP2DCRandomizedSurvivalDesign(
        n,
        lrv,
        cmv,
        ll,
        lc,
        gl,
        gc,
        prior_c,
        prior_e,
        assignments,
        schedule,
        bool(graduate_at_interim),
        tolerance,
    )


def run_bop2_dc_randomized_survival_trial(
    design: BOP2DCRandomizedSurvivalDesign,
    enrollment_times: ArrayLike,
    event_times: ArrayLike,
    *,
    final_followup: float,
) -> BOP2DCRandomizedSurvivalTrial:
    """Replay event durations under administrative censoring at scheduled looks.

    Event times are durations from enrollment and `+inf` denotes no observed event.
    Interim analyses occur at the last enrollment of each look; the final analysis
    follows the last enrollment by `final_followup`. Tied enrollment times use input
    order. This is a fixed-calendar Python replay, not a native RNG/timing simulation.
    """
    if not isinstance(design, BOP2DCRandomizedSurvivalDesign):
        raise TypeError("design must be BOP2DCRandomizedSurvivalDesign")
    _check_vector_shape(enrollment_times, design.max_subjects, "enrollment_times")
    _check_vector_shape(event_times, design.max_subjects, "event_times")
    enrolled = _real_array(enrollment_times, "enrollment_times")
    durations = _real_array(event_times, "event_times")
    followup = scalar(final_followup, "final_followup")
    if np.any(enrolled < 0) or np.any(np.diff(enrolled) < 0):
        raise ValueError("enrollment_times must be ordered and nonnegative")
    if np.any(np.isnan(durations) | (durations < 0)):
        raise ValueError("event_times must be nonnegative or positive infinity")
    if followup < 0:
        raise ValueError("final_followup must be nonnegative")
    final_clock = float(enrolled[-1] + followup)
    if not isfinite(final_clock):
        raise ArithmeticError("final analysis calendar time overflows")
    if followup > 0 and final_clock <= enrolled[-1]:
        raise ArithmeticError("positive final_followup is lost at this calendar-time scale")
    if int(np.sum(design.looks)) > _MAX_CALENDAR_WORK:
        raise ValueError("calendar replay exceeds its repeated-look work bound")

    states: list[BOP2DCRandomizedSurvivalState] = []
    clocks: list[float] = []
    enrolled_count = 0
    for raw_n in design.looks:
        n = int(raw_n)
        final = n == design.max_subjects
        last = float(enrolled[n - 1])
        clock = float(last + (followup if final else 0.0))
        elapsed = last - enrolled[:n]
        if final:
            elapsed = elapsed + followup
        if np.any(elapsed < 0) or np.any(~np.isfinite(elapsed)):
            raise ArithmeticError("as-of follow-up is not representable")
        observed = np.minimum(durations[:n], elapsed)
        event_observed = durations[:n] <= elapsed
        prefix = design.arm_assignments[:n]
        c_mask, e_mask = prefix == 0, prefix == 1
        state = design.monitor(
            int(np.count_nonzero(event_observed[c_mask])),
            float(np.sum(observed[c_mask], dtype=np.float64)),
            int(np.count_nonzero(c_mask)),
            int(np.count_nonzero(event_observed[e_mask])),
            float(np.sum(observed[e_mask], dtype=np.float64)),
            int(np.count_nonzero(e_mask)),
        )
        states.append(state)
        clocks.append(clock)
        enrolled_count = n
        if state.decision.item() != "continue":
            break
    return BOP2DCRandomizedSurvivalTrial(
        _owned(design.arm_assignments[:enrolled_count]),
        _owned(enrolled[:enrolled_count]),
        _owned(np.asarray(clocks, dtype=np.float64)),
        tuple(states),
        enrolled_count,
        str(states[-1].decision.item()),
    )
