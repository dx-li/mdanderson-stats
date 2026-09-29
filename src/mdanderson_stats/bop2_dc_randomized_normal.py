"""BOP2-DC randomized Normal monitoring with independent arm-specific NIG priors.

The randomized extension follows §2.4 of the BOP2-DC paper: fit each arm
independently and evaluate the posterior probability that the mean difference
exceeds each clinical margin. Allocation is a caller-supplied fixed tape; the
paper does not prescribe a universal randomization ratio or prior.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.stats import t as student_t

from ._bop2_dc_randomized_rules import randomized_dual_decisions
from ._validation import FloatArray, count, finite, scalar
from .normal_updating import NormalInverseGamma, NormalSample

IntArray = NDArray[np.int64]
_MAX_SUBJECTS = 1_000
_MAX_LOOKS = 100
_MAX_QUADRATURE_LIMIT = 200
_MAX_COMPARISON_WORK = 10_000_000
_MAX_REPLAY_WORK = 1_000_000
_DECISIONS = ("continue", "stop_no_go", "graduate", "final_go", "final_consider", "final_no_go")


def _freeze_int(value: ArrayLike) -> IntArray:
    result = np.array(value, dtype=np.int64, copy=True)
    result.flags.writeable = False
    return result


def _freeze_float(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _real_scalar(value: float, name: str) -> float:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    return scalar(value, name)


def _arm_vector(value: ArrayLike, name: str, maximum: int) -> FloatArray:
    """Validate a bounded real arm vector, allowing an empty arm at a look."""
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) > maximum or any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must be a bounded one-dimensional vector")
        if any(np.iscomplexobj(item) for item in value):
            raise ValueError(f"{name} must be real")
        shape = (len(value),)
    else:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if len(shape) != 1 or int(shape[0]) > maximum:
        raise ValueError(f"{name} must be a bounded one-dimensional vector")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    values = finite(value, name)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must contain finite real values")
    return np.asarray(values, dtype=np.float64)


def _integer_tape(value: ArrayLike, n: int, name: str) -> IntArray:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) != n or any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must have one value per planned patient")
        shape = (len(value),)
    else:
        raise ValueError(f"{name} must be a one-dimensional tape")
    if shape != (n,) or np.iscomplexobj(value):
        raise ValueError(f"{name} must have one real value per planned patient")
    return _freeze_int(count(value, name))


def _prior(value: ArrayLike, name: str) -> tuple[float, float, float, float]:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (tuple, list)):
        if len(value) != 4 or any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must be (location, mean_precision, shape, scale)")
        shape = (len(value),)
    else:
        raise ValueError(f"{name} must be (location, mean_precision, shape, scale)")
    if shape != (4,) or np.iscomplexobj(value):
        raise ValueError(f"{name} must contain four real scalar values")
    entries = finite(value, name)
    location, precision, shape, scale = (float(x) for x in entries)
    if precision <= 0 or shape <= 0 or scale <= 0:
        raise ValueError(f"{name} requires positive mean_precision, shape, and scale")
    # Construct once here to apply the same representability checks as updating.
    NormalInverseGamma(location, precision, shape, scale)
    return location, precision, shape, scale


def _look_tape(value: ArrayLike, n: int) -> IntArray:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) > min(n, _MAX_LOOKS) or any(not np.isscalar(item) for item in value):
            raise ValueError(f"looks must contain 1 to {_MAX_LOOKS} total-N analyses")
        shape = (len(value),)
    else:
        raise ValueError("looks must be a bounded one-dimensional schedule")
    if len(shape) != 1 or not 1 <= int(shape[0]) <= min(n, _MAX_LOOKS):
        raise ValueError(f"looks must contain 1 to {_MAX_LOOKS} total-N analyses")
    looks = _integer_tape(value, int(shape[0]), "looks")
    if not 1 <= looks.size <= min(n, _MAX_LOOKS):
        raise ValueError(f"looks must contain 1 to {_MAX_LOOKS} total-N analyses")
    if np.any(looks < 1) or np.any(looks > n) or looks[-1] != n or np.any(np.diff(looks) <= 0):
        raise ValueError("looks must increase strictly and end at max_subjects")
    return _freeze_int(looks)


def _difference_tail(
    control_location: float,
    control_scale: float,
    control_df: float,
    treatment_location: float,
    treatment_scale: float,
    treatment_df: float,
    margin: float,
    tolerance: float,
    limit: int,
) -> tuple[float, float]:
    """Integrate P(Treatment - Control > margin) with an absolute error estimate."""
    tail = tolerance / 16
    center_difference = treatment_location - control_location
    if not np.isfinite(center_difference):
        raise ArithmeticError("centered posterior location difference is not representable")
    if margin == center_difference:
        # The difference of independent symmetric Student-t variables is
        # symmetric around this exact location, even when their dfs differ.
        return 0.5, 0.0

    def integrand(u: float) -> float:
        control_quantile = float(student_t.ppf(u, control_df))
        if not np.isfinite(control_quantile):
            raise ArithmeticError("Student-t convolution quantile is not representable")
        with np.errstate(over="ignore", invalid="ignore"):
            standardized = (
                margin - center_difference + control_scale * control_quantile
            ) / treatment_scale
        if np.isnan(standardized):
            raise ArithmeticError("Student-t convolution argument is not representable")
        return float(student_t.sf(standardized, treatment_df))

    result = quad(
        integrand,
        tail,
        1 - tail,
        epsabs=tolerance / 4,
        epsrel=tolerance / 4,
        limit=limit,
        full_output=1,
    )
    probability, error = float(result[0]), float(result[1])
    if (
        len(result) != 3
        or not np.isfinite(probability)
        or not np.isfinite(error)
        or error > tolerance / 2
        or not 0 <= probability <= 1
    ):
        raise ArithmeticError("Student-t difference quadrature failed its error tolerance")
    total_error = error + 2 * tail
    return probability, total_error


@dataclass(frozen=True)
class BOP2DCRandomizedNormalState:
    """One total-N analysis; arm locations share `location_offset`."""

    total_n: int
    control_n: int
    treatment_n: int
    control_location_centered: float
    treatment_location_centered: float
    location_offset: float
    control_df: float
    control_scale: float
    treatment_df: float
    treatment_scale: float
    difference_location: float
    posterior_lrv: float
    posterior_cmv: float
    absolute_error_lrv: float
    absolute_error_cmv: float
    decision: str

    @property
    def control_location(self) -> float:
        value = self.control_location_centered + self.location_offset
        if not np.isfinite(value):
            raise ArithmeticError("absolute control posterior location is not representable")
        return value

    @property
    def treatment_location(self) -> float:
        value = self.treatment_location_centered + self.location_offset
        if not np.isfinite(value):
            raise ArithmeticError("absolute treatment posterior location is not representable")
        return value


@dataclass(frozen=True)
class BOP2DCRandomizedNormalReplay:
    """Fixed-allocation randomized Normal replay through a terminal scheduled look."""

    arm_assignments_observed: IntArray
    outcomes_observed: FloatArray
    states: tuple[BOP2DCRandomizedNormalState, ...]
    terminal_decision: str


@dataclass(frozen=True)
class BOP2DCRandomizedNormalDesign:
    """Independent arm-specific NIG posteriors and fixed randomized allocation."""

    max_subjects: int
    theta_lrv: float
    theta_cmv: float
    lambda_lrv: float
    lambda_cmv: float
    gamma_lrv: float
    gamma_cmv: float
    control_prior: tuple[float, float, float, float]
    treatment_prior: tuple[float, float, float, float]
    arm_assignments: IntArray
    looks: IntArray
    graduate_at_interim: bool
    comparison_tolerance: float
    quadrature_limit: int

    def _posterior(
        self, centered_values: FloatArray, prior: tuple[float, float, float, float], offset: float
    ) -> NormalInverseGamma:
        location, precision, shape, scale = prior
        centered_location = location - offset
        if not np.isfinite(centered_location):
            raise ArithmeticError("centered Normal prior location is not representable")
        posterior = NormalInverseGamma(centered_location, precision, shape, scale)
        if centered_values.size:
            posterior = posterior.update(NormalSample.from_data(centered_values))
        return posterior

    def monitor(
        self, control_observations: ArrayLike, treatment_observations: ArrayLike
    ) -> BOP2DCRandomizedNormalState:
        control = _arm_vector(control_observations, "control_observations", self.max_subjects)
        treatment = _arm_vector(treatment_observations, "treatment_observations", self.max_subjects)
        total = int(control.size + treatment.size)
        if total > self.max_subjects:
            raise ValueError("arm observation counts exceed max_subjects")
        if total:
            prefix = self.arm_assignments[:total]
            expected_control = int(np.count_nonzero(prefix == 0))
            expected_treatment = total - expected_control
            if (control.size, treatment.size) != (expected_control, expected_treatment):
                raise ValueError("arm observations do not match the fixed allocation prefix")
        offset = (
            float(control[0]) if control.size else float(treatment[0]) if treatment.size else 0.0
        )
        with np.errstate(over="ignore", invalid="ignore"):
            centered_control = control - offset
            centered_treatment = treatment - offset
        if np.any(~np.isfinite(centered_control)) or np.any(~np.isfinite(centered_treatment)):
            raise ArithmeticError("centered Normal observations are not representable")
        cpost = self._posterior(centered_control, self.control_prior, offset)
        tpost = self._posterior(centered_treatment, self.treatment_prior, offset)
        c_location = float(cpost.location)
        t_location = float(tpost.location)
        c_scale = float(cpost.mean_scale)
        t_scale = float(tpost.mean_scale)
        c_df = float(2 * cpost.shape)
        t_df = float(2 * tpost.shape)
        difference_location = t_location - c_location
        if not np.isfinite(difference_location):
            raise ArithmeticError("posterior mean difference is not representable")
        p_lrv, error_lrv = _difference_tail(
            c_location,
            c_scale,
            c_df,
            t_location,
            t_scale,
            t_df,
            self.theta_lrv,
            self.comparison_tolerance,
            self.quadrature_limit,
        )
        p_cmv, error_cmv = _difference_tail(
            c_location,
            c_scale,
            c_df,
            t_location,
            t_scale,
            t_df,
            self.theta_cmv,
            self.comparison_tolerance,
            self.quadrature_limit,
        )
        decision = randomized_dual_decisions(
            np.asarray([total], dtype=np.int64),
            np.asarray([p_lrv]),
            np.asarray([p_cmv]),
            np.asarray([error_lrv]),
            np.asarray([error_cmv]),
            max_subjects=self.max_subjects,
            looks=self.looks,
            lambda_lrv=self.lambda_lrv,
            lambda_cmv=self.lambda_cmv,
            gamma_lrv=self.gamma_lrv,
            gamma_cmv=self.gamma_cmv,
            graduate_at_interim=self.graduate_at_interim,
        )
        return BOP2DCRandomizedNormalState(
            total,
            int(control.size),
            int(treatment.size),
            c_location,
            t_location,
            offset,
            c_df,
            c_scale,
            t_df,
            t_scale,
            difference_location,
            p_lrv,
            p_cmv,
            error_lrv,
            error_cmv,
            str(decision[0]),
        )

    def replay(self, outcomes: ArrayLike) -> BOP2DCRandomizedNormalReplay:
        tape = _arm_vector(outcomes, "outcomes", self.max_subjects)
        if tape.size != self.max_subjects:
            raise ValueError("outcomes must have one value per planned patient")
        work = self.max_subjects * self.looks.size
        if work > _MAX_REPLAY_WORK:
            raise ValueError("randomized Normal replay exceeds its repeated-look work budget")
        states: list[BOP2DCRandomizedNormalState] = []
        terminal = "continue"
        used = 0
        next_look = 0
        for index in range(self.max_subjects):
            used = index + 1
            if used != int(self.looks[next_look]):
                continue
            arms = self.arm_assignments[:used]
            response = tape[:used]
            state = self.monitor(response[arms == 0], response[arms == 1])
            states.append(state)
            terminal = state.decision
            if terminal != "continue":
                break
            next_look += 1
        return BOP2DCRandomizedNormalReplay(
            _freeze_int(self.arm_assignments[:used]),
            _freeze_float(tape[:used]),
            tuple(states),
            terminal,
        )


def bop2_dc_randomized_normal_design(
    max_subjects: int,
    theta_lrv: float,
    theta_cmv: float,
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
    comparison_tolerance: float = 1e-8,
    quadrature_limit: int = 200,
) -> BOP2DCRandomizedNormalDesign:
    """Construct a fixed-allocation randomized Normal BOP2-DC design.

    Each prior is `(location, mean_precision, inverse_gamma_shape,
    inverse_gamma_scale)` for `NormalInverseGamma`. ``arm_assignments`` uses 0
    for control and 1 for treatment. ``looks`` are total enrolled sample sizes
    and must end at ``max_subjects``. Arm allocation probabilities and prior
    defaults are not inferred from the paper.
    """
    n_value = _real_scalar(max_subjects, "max_subjects")
    n = int(n_value)
    if n_value != n or not 2 <= n <= _MAX_SUBJECTS:
        raise ValueError(f"max_subjects must be an integer in [2,{_MAX_SUBJECTS}]")
    lrv, cmv = scalar(theta_lrv, "theta_lrv"), scalar(theta_cmv, "theta_cmv")
    if not lrv < cmv:
        raise ValueError("require theta_lrv < theta_cmv")
    ll, lc = scalar(lambda_lrv, "lambda_lrv"), scalar(lambda_cmv, "lambda_cmv")
    gl, gc = scalar(gamma_lrv, "gamma_lrv"), scalar(gamma_cmv, "gamma_cmv")
    if not 0 < ll < 1 or not 0 < lc < 1 or not 0 <= gl <= 1 or not 0 <= gc <= 1:
        raise ValueError("lambda values must lie in (0,1) and gamma values in [0,1]")
    if not isinstance(graduate_at_interim, (bool, np.bool_)):
        raise ValueError("graduate_at_interim must be boolean")
    tolerance = _real_scalar(comparison_tolerance, "comparison_tolerance")
    limit_value = _real_scalar(quadrature_limit, "quadrature_limit")
    limit = int(limit_value)
    if limit_value != limit or not 1 <= limit <= _MAX_QUADRATURE_LIMIT:
        raise ValueError(f"quadrature_limit must be an integer in [1,{_MAX_QUADRATURE_LIMIT}]")
    if not 1e-12 <= tolerance <= 1e-3:
        raise ValueError("comparison_tolerance must lie in [1e-12,1e-3]")
    control = _prior(control_prior, "control_prior")
    treatment = _prior(treatment_prior, "treatment_prior")
    assignments = _integer_tape(arm_assignments, n, "arm_assignments")
    if np.any(assignments > 1) or not np.any(assignments == 0) or not np.any(assignments == 1):
        raise ValueError("arm_assignments must use 0/1 and include both arms")
    schedule = _look_tape(looks, n)
    work = int(schedule.size) * 2 * 21 * (2 * limit - 1)
    if work > _MAX_COMPARISON_WORK:
        raise ValueError("Student-t comparison exceeds its deterministic quadrature work budget")
    for gamma, level, label in ((gl, ll, "LRV"), (gc, lc, "CMV")):
        if gamma and level * (int(schedule[0]) / n) ** gamma == 0:
            raise ArithmeticError(f"interim {label} cutoff underflows")
    return BOP2DCRandomizedNormalDesign(
        n,
        lrv,
        cmv,
        ll,
        lc,
        gl,
        gc,
        control,
        treatment,
        _freeze_int(assignments),
        schedule,
        bool(graduate_at_interim),
        tolerance,
        limit,
    )
