"""BOP2-DC monitoring and exact OC recursion for randomized binary trials."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erf, erfinv

from ._validation import FloatArray, count, finite, scalar
from .beta_binomial import BetaBinomialPosterior
from .beta_comparison import compare_beta_difference

IntArray = NDArray[np.int64]

_MAX_SUBJECTS = 1_000
_MAX_MONITOR_CELLS = 1_000
_MAX_OC_SCENARIOS = 1_000
_MAX_COMPARISON_CELLS = 5_000
_MAX_RECURSION_WORK = 5_000_000
_MAX_OC_RESULT_CELLS = 100_000


def _owned(value: ArrayLike, *, dtype: np.dtype | type = np.float64) -> NDArray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _prior_pair(value: ArrayLike, name: str) -> tuple[float, float]:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) != 2 or any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must contain two positive Beta shapes")
        shape = (len(value),)
    else:
        shape = np.shape(value)
    if shape != (2,) or np.iscomplexobj(value):
        raise ValueError(f"{name} must contain two positive Beta shapes")
    alpha, beta = finite(value, name)
    if np.any(np.asarray([alpha, beta]) <= 0) or not np.isfinite(alpha + beta):
        raise ValueError(f"{name} must contain two positive finite Beta shapes")
    return float(alpha), float(beta)


def _arm_schedule(value: ArrayLike, n: int) -> IntArray:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) != n or any(not np.isscalar(item) for item in value):
            raise ValueError("arm_assignments must have one 0/1 value per planned subject")
        shape = (len(value),)
    else:
        shape = np.shape(value)
    if shape != (n,) or np.iscomplexobj(value):
        raise ValueError("arm_assignments must have one 0/1 value per planned subject")
    assignment = count(value, "arm_assignments")
    if np.any(assignment > 1):
        raise ValueError("arm_assignments use 0 for control and 1 for experimental")
    if not np.any(assignment == 0) or not np.any(assignment == 1):
        raise ValueError("the fixed allocation must assign at least one subject to each arm")
    return _owned(assignment, dtype=np.int64)


def _look_schedule(value: ArrayLike, n: int) -> IntArray:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) > n or any(not np.isscalar(item) for item in value):
            raise ValueError("looks must be a nonempty increasing schedule ending at max_subjects")
        shape = (len(value),)
    else:
        shape = np.shape(value)
    if len(shape) != 1 or not 0 < shape[0] <= n or np.iscomplexobj(value):
        raise ValueError("looks must be a nonempty increasing schedule ending at max_subjects")
    looks = count(value, "looks")
    if np.any(looks < 1) or np.any(looks > n) or looks[-1] != n or np.any(np.diff(looks) <= 0):
        raise ValueError("looks must increase strictly and end at max_subjects")
    return _owned(looks, dtype=np.int64)


def _binary_tape(value: ArrayLike, n: int, name: str) -> IntArray:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) != n or any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must contain one binary value per planned subject")
        shape = (len(value),)
    else:
        shape = np.shape(value)
    if shape != (n,) or np.iscomplexobj(value):
        raise ValueError(f"{name} must contain one binary value per planned subject")
    result = count(value, name)
    if np.any(result > 1):
        raise ValueError(f"{name} values must be zero or one")
    return _owned(result, dtype=np.int64)


@dataclass(frozen=True)
class BOP2DCRandomizedBinaryState:
    total_n: IntArray
    control_n: IntArray
    treatment_n: IntArray
    control_responses: IntArray
    treatment_responses: IntArray
    posterior_lrv: FloatArray
    posterior_cmv: FloatArray
    absolute_error_lrv: FloatArray
    absolute_error_cmv: FloatArray
    decision: NDArray[np.str_]


@dataclass(frozen=True)
class BOP2DCRandomizedBinaryReplay:
    """Fixed-allocation replay through the first terminal scheduled look."""

    arm_assignments_observed: IntArray
    responses_observed: IntArray
    states: tuple[BOP2DCRandomizedBinaryState, ...]
    terminal_decision: str


@dataclass(frozen=True)
class BOP2DCRandomizedBinaryOperatingCharacteristics:
    control_probability: FloatArray
    treatment_probability: FloatArray
    looks: IntArray
    stop_no_go: FloatArray
    graduate: FloatArray
    final_go: FloatArray
    final_consider: FloatArray
    final_no_go: FloatArray
    sample_size_probability: FloatArray
    expected_sample_size: FloatArray
    maximum_comparison_error: FloatArray

    @property
    def no_go_probability(self) -> FloatArray:
        return self.stop_no_go.sum(axis=-1) + self.final_no_go


@dataclass(frozen=True)
class BOP2DCRandomizedBinaryDesign:
    """Fixed-allocation independent-Beta BOP2-DC design for two binary arms."""

    max_subjects: int
    theta_lrv: float
    theta_cmv: float
    lambda_lrv: float
    lambda_cmv: float
    gamma_lrv: float
    gamma_cmv: float
    control_prior: tuple[float, float]
    treatment_prior: tuple[float, float]
    arm_assignments: IntArray
    looks: IntArray
    graduate_at_interim: bool
    comparison_tolerance: float

    def _posterior_tables(
        self,
    ) -> tuple[tuple[FloatArray, FloatArray, FloatArray, FloatArray], ...]:
        control_counts = np.r_[0, np.cumsum(self.arm_assignments == 0)].astype(np.int64)
        treatment_counts = np.r_[0, np.cumsum(self.arm_assignments == 1)].astype(np.int64)
        comparison_cells = sum(
            2 * (int(control_counts[int(look)]) + 1) * (int(treatment_counts[int(look)]) + 1)
            for look in self.looks
        )
        if comparison_cells > _MAX_COMPARISON_CELLS:
            raise ValueError("posterior count-state comparison table exceeds its work bound")
        table = []
        for raw_look in self.looks:
            look = int(raw_look)
            n_control = int(control_counts[look])
            n_treatment = int(treatment_counts[look])
            c_responses = np.arange(n_control + 1, dtype=np.float64)[:, None]
            t_responses = np.arange(n_treatment + 1, dtype=np.float64)[None, :]
            control = BetaBinomialPosterior(
                self.control_prior[0] + c_responses,
                self.control_prior[1] + n_control - c_responses,
            )
            treatment = BetaBinomialPosterior(
                self.treatment_prior[0] + t_responses,
                self.treatment_prior[1] + n_treatment - t_responses,
            )
            lrv = compare_beta_difference(
                control,
                treatment,
                self.theta_lrv,
                absolute_tolerance=self.comparison_tolerance,
            )
            cmv = compare_beta_difference(
                control,
                treatment,
                self.theta_cmv,
                absolute_tolerance=self.comparison_tolerance,
            )
            table.append(
                (
                    lrv.above_margin,
                    cmv.above_margin,
                    lrv.absolute_error,
                    cmv.absolute_error,
                )
            )
        return tuple(table)

    def _gradient_cutoffs(self, look: int) -> tuple[float, float]:
        fraction = look / self.max_subjects
        return (
            float(erf(erfinv(self.lambda_lrv) / np.sqrt(fraction))),
            float(erf(erfinv(self.lambda_cmv) / np.sqrt(fraction))),
        )

    def _decision_from_tails(
        self, total_n: np.ndarray, posterior_lrv: np.ndarray, posterior_cmv: np.ndarray
    ) -> NDArray[np.str_]:
        decision = np.full(total_n.shape, "continue", dtype="U16")
        for raw_look in self.looks[:-1]:
            look = int(raw_look)
            at_look = total_n == look
            no_go = (
                at_look
                & (posterior_lrv < self.lambda_lrv * (look / self.max_subjects) ** self.gamma_lrv)
                & (posterior_cmv < self.lambda_cmv * (look / self.max_subjects) ** self.gamma_cmv)
            )
            decision[no_go] = "stop_no_go"
            if self.graduate_at_interim:
                grad_lrv, grad_cmv = self._gradient_cutoffs(look)
                graduate = at_look & (posterior_lrv > grad_lrv) & (posterior_cmv > grad_cmv)
                if np.any(no_go & graduate):
                    raise ArithmeticError("interim graduation and no-go rules overlap")
                decision[graduate] = "graduate"
        final = total_n == self.max_subjects
        go = final & (posterior_lrv > self.lambda_lrv) & (posterior_cmv > self.lambda_cmv)
        no_go = final & (posterior_lrv < self.lambda_lrv) & (posterior_cmv < self.lambda_cmv)
        decision[go] = "final_go"
        decision[no_go] = "final_no_go"
        decision[final & ~(go | no_go)] = "final_consider"
        decision.flags.writeable = False
        return decision

    def _decision_from_error_intervals(
        self,
        total_n: np.ndarray,
        posterior_lrv: FloatArray,
        posterior_cmv: FloatArray,
        error_lrv: FloatArray,
        error_cmv: FloatArray,
    ) -> NDArray[np.str_]:
        """Classify only when reported quadrature errors cannot change the action."""
        low_lrv = np.maximum(0.0, posterior_lrv - error_lrv)
        high_lrv = np.minimum(1.0, posterior_lrv + error_lrv)
        low_cmv = np.maximum(0.0, posterior_cmv - error_cmv)
        high_cmv = np.minimum(1.0, posterior_cmv + error_cmv)
        choices = (
            self._decision_from_tails(total_n, low_lrv, low_cmv),
            self._decision_from_tails(total_n, low_lrv, high_cmv),
            self._decision_from_tails(total_n, high_lrv, low_cmv),
            self._decision_from_tails(total_n, high_lrv, high_cmv),
        )
        reference = choices[0]
        if any(np.any(choice != reference) for choice in choices[1:]):
            raise ArithmeticError(
                "reported beta-comparison quadrature error could change a strict trial decision"
            )
        return reference

    def monitor(
        self,
        control_responses: ArrayLike,
        control_sample_size: ArrayLike,
        treatment_responses: ArrayLike,
        treatment_sample_size: ArrayLike,
    ) -> BOP2DCRandomizedBinaryState:
        raw_inputs = (
            (control_responses, "control_responses"),
            (control_sample_size, "control_sample_size"),
            (treatment_responses, "treatment_responses"),
            (treatment_sample_size, "treatment_sample_size"),
        )
        for value, name in raw_inputs:
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
        values = [
            count(control_responses, "control_responses"),
            count(control_sample_size, "control_sample_size"),
            count(treatment_responses, "treatment_responses"),
            count(treatment_sample_size, "treatment_sample_size"),
        ]
        cr, cn, tr, tn = np.broadcast_arrays(*values)
        if cr.size > _MAX_MONITOR_CELLS:
            raise ValueError("monitor batch exceeds its cell budget")
        total_n = cn + tn
        if np.any((cr > cn) | (tr > tn) | (total_n > self.max_subjects)):
            raise ValueError("response counts and arm sample sizes are inconsistent")
        control_prefix = np.r_[0, np.cumsum(self.arm_assignments == 0)]
        treatment_prefix = np.r_[0, np.cumsum(self.arm_assignments == 1)]
        n_control = control_prefix[total_n.astype(np.int64)]
        n_treatment = treatment_prefix[total_n.astype(np.int64)]
        if np.any((cn != n_control) | (tn != n_treatment)):
            raise ValueError("arm sample sizes do not match the fixed allocation prefix")
        control_posterior = BetaBinomialPosterior(
            self.control_prior[0] + cr, self.control_prior[1] + cn - cr
        )
        treatment_posterior = BetaBinomialPosterior(
            self.treatment_prior[0] + tr, self.treatment_prior[1] + tn - tr
        )
        lrv = compare_beta_difference(
            control_posterior,
            treatment_posterior,
            self.theta_lrv,
            absolute_tolerance=self.comparison_tolerance,
        )
        cmv = compare_beta_difference(
            control_posterior,
            treatment_posterior,
            self.theta_cmv,
            absolute_tolerance=self.comparison_tolerance,
        )
        decision = self._decision_from_error_intervals(
            total_n,
            lrv.above_margin,
            cmv.above_margin,
            lrv.absolute_error,
            cmv.absolute_error,
        )
        return BOP2DCRandomizedBinaryState(
            _owned(total_n, dtype=np.int64),
            _owned(cn, dtype=np.int64),
            _owned(tn, dtype=np.int64),
            _owned(cr, dtype=np.int64),
            _owned(tr, dtype=np.int64),
            _owned(lrv.above_margin),
            _owned(cmv.above_margin),
            _owned(lrv.absolute_error),
            _owned(cmv.absolute_error),
            decision,
        )

    def replay(self, responses: ArrayLike) -> BOP2DCRandomizedBinaryReplay:
        tape = _binary_tape(responses, self.max_subjects, "responses")
        states = []
        control_n = control_responses = treatment_n = treatment_responses = 0
        terminal = "continue"
        used = 0
        next_look = 0
        for index, (arm, response) in enumerate(zip(self.arm_assignments, tape, strict=True)):
            used = index + 1
            if arm == 0:
                control_n += 1
                control_responses += int(response)
            else:
                treatment_n += 1
                treatment_responses += int(response)
            if used != int(self.looks[next_look]):
                continue
            state = self.monitor(control_responses, control_n, treatment_responses, treatment_n)
            states.append(state)
            terminal = str(state.decision.item())
            if terminal != "continue":
                break
            next_look += 1
        return BOP2DCRandomizedBinaryReplay(
            _owned(self.arm_assignments[:used], dtype=np.int64),
            _owned(tape[:used], dtype=np.int64),
            tuple(states),
            terminal,
        )

    def operating_characteristics(
        self, control_probability: ArrayLike, treatment_probability: ArrayLike
    ) -> BOP2DCRandomizedBinaryOperatingCharacteristics:
        control_rate = _probability_scenarios(control_probability, "control_probability")
        treatment_rate = _probability_scenarios(treatment_probability, "treatment_probability")
        control_rate, treatment_rate = np.broadcast_arrays(control_rate, treatment_rate)
        scenario_count = control_rate.size
        if scenario_count > _MAX_OC_SCENARIOS:
            raise ValueError("too many randomized-arm truth scenarios")

        control_prefix = np.r_[0, np.cumsum(self.arm_assignments == 0)]
        treatment_prefix = np.r_[0, np.cumsum(self.arm_assignments == 1)]
        comparison_work = sum(
            2 * (int(control_prefix[int(n)]) + 1) * (int(treatment_prefix[int(n)]) + 1)
            for n in self.looks
        )
        if comparison_work > _MAX_COMPARISON_CELLS:
            raise ValueError("posterior count-state comparison table exceeds its work bound")
        transition_work = sum(
            (int(control_prefix[n]) + 1) * (int(treatment_prefix[n]) + 1)
            for n in range(1, self.max_subjects + 1)
        )
        total_work = scenario_count * transition_work + comparison_work
        if total_work > _MAX_RECURSION_WORK:
            raise ValueError("exact randomized-arm recursion exceeds its work bound")
        result_cells = scenario_count * (8 * len(self.looks) + 8)
        if result_cells > _MAX_OC_RESULT_CELLS:
            raise ValueError("randomized-arm OC result exceeds its retained-cell bound")

        posterior_tables = self._posterior_tables()
        shape = control_rate.shape
        scenarios = list(zip(control_rate.ravel(), treatment_rate.ravel(), strict=True))
        stop = np.zeros((scenario_count, self.looks.size), dtype=np.float64)
        graduate = np.zeros_like(stop)
        final_go = np.zeros(scenario_count)
        final_consider = np.zeros(scenario_count)
        final_no_go = np.zeros(scenario_count)
        sample_size = np.zeros_like(stop)
        expected = np.zeros(scenario_count)
        comparison_error = np.zeros(self.looks.size)

        look_to_index = {int(n): i for i, n in enumerate(self.looks)}
        for look_index, (_pl, _pcmv, error_lrv, error_cmv) in enumerate(posterior_tables):
            comparison_error[look_index] = float(np.max(np.maximum(error_lrv, error_cmv)))

        for scenario_index, (pc, pt) in enumerate(scenarios):
            state = np.ones((1, 1), dtype=np.float64)
            look_index = 0
            for patient_index, arm in enumerate(self.arm_assignments, start=1):
                if arm == 0:
                    updated = np.zeros((state.shape[0] + 1, state.shape[1]))
                    updated[:-1] += state * (1 - pc)
                    updated[1:] += state * pc
                else:
                    updated = np.zeros((state.shape[0], state.shape[1] + 1))
                    updated[:, :-1] += state * (1 - pt)
                    updated[:, 1:] += state * pt
                state = updated
                if patient_index not in look_to_index:
                    continue
                look_index = look_to_index[patient_index]
                pl, pcmv, error_lrv, error_cmv = posterior_tables[look_index]
                total_n = np.full(pl.shape, patient_index, dtype=np.int64)
                decision = self._decision_from_error_intervals(
                    total_n, pl, pcmv, error_lrv, error_cmv
                )
                if patient_index == self.max_subjects:
                    go_mask = decision == "final_go"
                    no_mask = decision == "final_no_go"
                    final_go[scenario_index] = float(np.sum(state[go_mask]))
                    final_no_go[scenario_index] = float(np.sum(state[no_mask]))
                    final_consider[scenario_index] = float(np.sum(state[~(go_mask | no_mask)]))
                    sample_size[scenario_index, look_index] = float(np.sum(state))
                    break

                no_mask = decision == "stop_no_go"
                no_mass = float(np.sum(state[no_mask]))
                stop[scenario_index, look_index] = no_mass
                state[no_mask] = 0.0
                grad_mass = 0.0
                if self.graduate_at_interim:
                    grad_mask = decision == "graduate"
                    grad_mass = float(np.sum(state[grad_mask]))
                    graduate[scenario_index, look_index] = grad_mass
                    state[grad_mask] = 0.0
                sample_size[scenario_index, look_index] = no_mass + grad_mass
            final_mass = (
                final_go[scenario_index]
                + final_consider[scenario_index]
                + final_no_go[scenario_index]
            )
            sample_size[scenario_index, -1] = final_mass
            expected[scenario_index] = float(sample_size[scenario_index] @ self.looks)

        if np.any(np.abs(sample_size.sum(axis=1) - 1) > 1e-10):
            raise ArithmeticError("randomized-arm OC probabilities do not sum to one")
        return BOP2DCRandomizedBinaryOperatingCharacteristics(
            _owned(control_rate.reshape(shape)),
            _owned(treatment_rate.reshape(shape)),
            self.looks,
            _owned(stop.reshape(*shape, self.looks.size)),
            _owned(graduate.reshape(*shape, self.looks.size)),
            _owned(final_go.reshape(shape)),
            _owned(final_consider.reshape(shape)),
            _owned(final_no_go.reshape(shape)),
            _owned(sample_size.reshape(*shape, self.looks.size)),
            _owned(expected.reshape(shape)),
            _owned(comparison_error),
        )


def _probability_scenarios(value: ArrayLike, name: str) -> FloatArray:
    if isinstance(value, np.ndarray):
        if value.size > _MAX_OC_SCENARIOS:
            raise ValueError("too many randomized-arm truth scenarios")
    elif isinstance(value, (list, tuple)):
        if len(value) > _MAX_OC_SCENARIOS or any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must be a scalar or bounded 1D scenario array")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    result = finite(value, name)
    if np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} must lie in [0,1]")
    return result


def bop2_dc_randomized_binary_design(
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
    comparison_tolerance: float = 1e-9,
) -> BOP2DCRandomizedBinaryDesign:
    """Build a BOP2-DC randomized binary design with explicit fixed allocation.

    ``arm_assignments`` is a length-``max_subjects`` tape with 0 for control
    and 1 for experimental. The source does not prescribe one universal
    randomization ratio, so no allocation schedule is synthesized. Posterior
    arm priors and cumulative total-subject looks are explicit.
    """
    n_value = scalar(max_subjects, "max_subjects")
    n = int(n_value)
    if n_value != n or not 2 <= n <= _MAX_SUBJECTS:
        raise ValueError(f"max_subjects must be an integer in [2,{_MAX_SUBJECTS}]")
    lrv, cmv = scalar(theta_lrv, "theta_lrv"), scalar(theta_cmv, "theta_cmv")
    if not -1 <= lrv < cmv <= 1:
        raise ValueError("require -1 <= theta_lrv < theta_cmv <= 1")
    ll, lc = scalar(lambda_lrv, "lambda_lrv"), scalar(lambda_cmv, "lambda_cmv")
    gl, gc = scalar(gamma_lrv, "gamma_lrv"), scalar(gamma_cmv, "gamma_cmv")
    if not 0 < ll < 1 or not 0 < lc < 1 or not 0 <= gl <= 1 or not 0 <= gc <= 1:
        raise ValueError("lambda values must lie in (0,1) and gamma values in [0,1]")
    if not isinstance(graduate_at_interim, (bool, np.bool_)):
        raise ValueError("graduate_at_interim must be boolean")
    tolerance = scalar(comparison_tolerance, "comparison_tolerance")
    if not 1e-12 <= tolerance <= 1e-3:
        raise ValueError("comparison_tolerance must lie in [1e-12,1e-3]")
    control = _prior_pair(control_prior, "control_prior")
    treatment = _prior_pair(treatment_prior, "treatment_prior")
    assignment = _arm_schedule(arm_assignments, n)
    schedule = _look_schedule(looks, n)
    if gl and ll * (int(schedule[0]) / n) ** gl == 0:
        raise ArithmeticError("interim LRV cutoff underflows; increase lambda_lrv")
    if gc and lc * (int(schedule[0]) / n) ** gc == 0:
        raise ArithmeticError("interim CMV cutoff underflows; increase lambda_cmv")
    return BOP2DCRandomizedBinaryDesign(
        n,
        lrv,
        cmv,
        ll,
        lc,
        gl,
        gc,
        control,
        treatment,
        assignment,
        schedule,
        bool(graduate_at_interim),
        tolerance,
    )
