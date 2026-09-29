"""Randomized BOP2-DC monitoring for paired binary endpoints.

Each arm has a four-category Dirichlet model. Marginal endpoint risk
comparisons are combined using the multiple-endpoint OR or co-primary AND
rules; dependence between endpoints within a patient is retained by the
joint-category model.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._bop2_dc_randomized_rules import randomized_obf_cutoffs
from ._validation import FloatArray, count, finite, scalar
from .beta_binomial import BetaBinomialPosterior
from .beta_comparison import compare_beta_difference

IntArray = NDArray[np.int64]
_ACTIONS = ("continue", "stop_no_go", "graduate", "final_go", "final_consider", "final_no_go")
_MAX_SUBJECTS = 1_000
_MAX_LOOKS = 100
_MAX_TAIL_ROWS = 200
_MAX_REPLAY_WORK = 50_000_000
_MAX_PREFIX_WORK = 500_000
_MAX_TAIL_COMPARISON_WORK = 65_000_000
_QUAD_LIMIT = 300
_QUAD_NODES = 21
_QUAD_INTERVAL_FACTOR = 2 * _QUAD_LIMIT - 1


def _owned(value: ArrayLike, *, dtype: np.dtype | type = np.float64) -> NDArray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _bounded_vector(value: ArrayLike, name: str, length: int) -> FloatArray:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) != length or any(not np.isscalar(x) for x in value):
            raise ValueError(f"{name} must be a length-{length} vector")
        shape = (len(value),)
    else:
        raise ValueError(f"{name} must be a length-{length} vector")
    if shape != (length,) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a real length-{length} vector")
    result = np.asarray(finite(value, name), dtype=np.float64)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain finite real values")
    return result


def _category_vector(value: ArrayLike, name: str) -> IntArray:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) != 4 or any(not np.isscalar(x) for x in value):
            raise ValueError(f"{name} must contain four category counts")
        shape = (len(value),)
    else:
        raise ValueError(f"{name} must contain four category counts")
    if shape != (4,) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a real length-four count vector")
    values = count(value, name)
    if np.any(values > _MAX_SUBJECTS):
        raise ValueError(f"{name} counts exceed the subject bound")
    return np.asarray(values, dtype=np.int64)


def _dirichlet_prior(value: ArrayLike, name: str) -> tuple[float, float, float, float]:
    prior = _bounded_vector(value, name, 4)
    if np.any(prior <= 0) or not np.isfinite(np.sum(prior)):
        raise ValueError(f"{name} must contain four positive finite Dirichlet shapes")
    return tuple(float(x) for x in prior)  # type: ignore[return-value]


def _integer_tape(value: ArrayLike, n: int, name: str, maximum: int) -> IntArray:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) != n or any(not np.isscalar(x) for x in value):
            raise ValueError(f"{name} must have one value per planned subject")
        shape = (len(value),)
    else:
        raise ValueError(f"{name} must be a bounded one-dimensional tape")
    if shape != (n,) or np.iscomplexobj(value):
        raise ValueError(f"{name} must have one real value per planned subject")
    result = count(value, name)
    if np.any(result > maximum):
        raise ValueError(f"{name} values exceed their allowed range")
    return _owned(result, dtype=np.int64)


def _endpoint_values(value: ArrayLike, name: str, lower: float, upper: float) -> FloatArray:
    values = _bounded_vector(value, name, 2)
    if np.any((values < lower) | (values > upper)):
        raise ValueError(f"{name} values must lie in [{lower},{upper}]")
    return values.copy()


def _endpoint_actions(
    probability: FloatArray,
    total_n: int,
    maximum_n: int,
    lambda_lrv: float,
    lambda_cmv: float,
    gamma_lrv: float,
    gamma_cmv: float,
    graduate_at_interim: bool,
) -> NDArray[np.int8]:
    """Endpoint actions: 0 continue/consider, 1 no-go, 2 graduate/go."""
    action = np.zeros(probability.shape[0], dtype=np.int8)
    if total_n == maximum_n:
        no_go = (probability[:, 0] < lambda_lrv) & (probability[:, 1] < lambda_cmv)
        go = (probability[:, 0] > lambda_lrv) & (probability[:, 1] > lambda_cmv)
        action[no_go], action[go] = 1, 2
        return action
    fraction = total_n / maximum_n
    cutoff_lrv = lambda_lrv * fraction**gamma_lrv
    cutoff_cmv = lambda_cmv * fraction**gamma_cmv
    no_go = (probability[:, 0] < cutoff_lrv) & (probability[:, 1] < cutoff_cmv)
    action[no_go] = 1
    if graduate_at_interim:
        grad_lrv, grad_cmv = randomized_obf_cutoffs(
            look=total_n,
            max_subjects=maximum_n,
            lambda_lrv=lambda_lrv,
            lambda_cmv=lambda_cmv,
        )
        graduate = (probability[:, 0] > grad_lrv) & (probability[:, 1] > grad_cmv)
        action[graduate] = 2
    return action


def _combine_actions(
    endpoint_actions: NDArray[np.int8], endpoint: str, final: bool
) -> NDArray[np.int8]:
    """Vectorized combined action codes: interim 0/1/2, final 0/1/2."""
    first, second = endpoint_actions[:, 0], endpoint_actions[:, 1]
    result = np.zeros(first.shape, dtype=np.int8)
    if endpoint == "multiple_efficacy":
        if final:
            result[(first == 2) | (second == 2)] = 2
            result[(first == 1) & (second == 1)] = 1
        else:
            result[(first == 1) & (second == 1)] = 1
            result[(first == 2) | (second == 2)] = 2
    elif final:
        result[(first == 2) & (second == 2)] = 2
        result[(first == 1) | (second == 1)] = 1
    else:
        result[(first == 1) | (second == 1)] = 1
        result[(first == 2) & (second == 2)] = 2
    return result


def _action_labels(codes: NDArray[np.int8], final: bool) -> NDArray[np.str_]:
    mapping = (
        ("final_consider", "final_no_go", "final_go")
        if final
        else ("continue", "stop_no_go", "graduate")
    )
    return np.asarray(mapping, dtype="U16")[codes]


def _paired_decisions_from_tails(
    probability: ArrayLike,
    errors: ArrayLike,
    *,
    total_n: int,
    max_subjects: int,
    looks: IntArray,
    endpoint: str,
    lambda_lrv: FloatArray,
    lambda_cmv: FloatArray,
    gamma_lrv: FloatArray,
    gamma_cmv: FloatArray,
    graduate_at_interim: bool,
) -> NDArray[np.str_]:
    """Classify bounded rows only if the combined decision is stable at 16 corners."""
    p_shape = probability.shape if isinstance(probability, np.ndarray) else None
    e_shape = errors.shape if isinstance(errors, np.ndarray) else None
    if p_shape is None or e_shape is None:
        raise ValueError("probability and errors must be bounded NumPy arrays")
    if len(p_shape) != 3 or len(e_shape) != 3:
        raise ValueError("probability and errors must have shape (rows,2,2)")
    if p_shape[0] > _MAX_TAIL_ROWS:
        raise ValueError(f"decision batches are limited to {_MAX_TAIL_ROWS} rows")
    raw_p, raw_e = np.asarray(probability), np.asarray(errors)
    if raw_p.shape[1:] != (2, 2) or raw_e.shape != raw_p.shape:
        raise ValueError("probability and errors must have shape (rows,2 endpoints,2 criteria)")
    if np.iscomplexobj(raw_p) or np.iscomplexobj(raw_e):
        raise ValueError("probability and errors must be real")
    p = np.asarray(finite(raw_p, "probability"), dtype=np.float64)
    e = np.asarray(finite(raw_e, "errors"), dtype=np.float64)
    if np.any((p < 0) | (p > 1)) or np.any(e < 0):
        raise ValueError("posterior probabilities must lie in [0,1] and errors be nonnegative")
    final = total_n == max_subjects
    if not np.any(looks == total_n):
        return _owned(["continue"] * p.shape[0], dtype=np.dtype("U16"))
    lower = np.maximum(0.0, p - e)
    upper = np.minimum(1.0, p + e)
    reference: NDArray[np.int8] | None = None
    for bits in product((0, 1), repeat=4):
        choose_upper = np.asarray(bits, dtype=bool).reshape(1, 2, 2)
        corner = np.where(choose_upper, upper, lower)
        actions = _endpoint_actions(
            corner[:, 0, :],
            total_n,
            max_subjects,
            float(lambda_lrv[0]),
            float(lambda_cmv[0]),
            float(gamma_lrv[0]),
            float(gamma_cmv[0]),
            graduate_at_interim,
        )
        # The endpoint axis is independent at the action stage and joint only
        # when the OR/AND composition is applied.
        second_actions = _endpoint_actions(
            corner[:, 1, :],
            total_n,
            max_subjects,
            float(lambda_lrv[1]),
            float(lambda_cmv[1]),
            float(gamma_lrv[1]),
            float(gamma_cmv[1]),
            graduate_at_interim,
        )
        paired_actions = np.column_stack((actions, second_actions))
        current = _combine_actions(paired_actions, endpoint, final)
        if reference is None:
            reference = current
        elif np.any(current != reference):
            raise ArithmeticError(
                "paired endpoint quadrature error could change the combined strict decision"
            )
    assert reference is not None
    return _owned(_action_labels(reference, final), dtype=np.dtype("U16"))


@dataclass(frozen=True)
class BOP2DCRandomizedPairedState:
    """One look's joint-category counts and endpoint posterior summaries."""

    total_n: int
    control_n: int
    treatment_n: int
    control_counts: IntArray
    treatment_counts: IntArray
    posterior_probability: FloatArray
    absolute_error: FloatArray
    nominal_endpoint_decisions: tuple[str, str]
    decision: str

    @property
    def endpoint_decisions(self) -> tuple[str, str]:
        """Nominal midpoint endpoint actions; the composite decision is guarded."""
        return self.nominal_endpoint_decisions


@dataclass(frozen=True)
class BOP2DCRandomizedPairedReplay:
    """Fixed-allocation paired-category replay through its first terminal look."""

    arm_assignments_observed: IntArray
    outcomes_observed: IntArray
    states: tuple[BOP2DCRandomizedPairedState, ...]
    terminal_decision: str


@dataclass(frozen=True)
class BOP2DCRandomizedPairedDesign:
    """Joint Dirichlet model and fixed-allocation paired-endpoint design."""

    max_subjects: int
    endpoint: str
    lrv: FloatArray
    cmv: FloatArray
    control_prior: tuple[float, float, float, float]
    treatment_prior: tuple[float, float, float, float]
    lambda_lrv: FloatArray
    lambda_cmv: FloatArray
    gamma_lrv: FloatArray
    gamma_cmv: FloatArray
    arm_assignments: IntArray
    looks: IntArray
    graduate_at_interim: bool
    comparison_tolerance: float

    def _posterior_tails_from_success_counts(
        self,
        control_n: int,
        treatment_n: int,
        control_successes: ArrayLike,
        treatment_successes: ArrayLike,
    ) -> tuple[FloatArray, FloatArray]:
        """Vectorized tails for state rows in (endpoint, criterion) axis order."""
        if not isinstance(control_successes, np.ndarray) or not isinstance(
            treatment_successes, np.ndarray
        ):
            raise ValueError("success arrays must be bounded NumPy arrays")
        if control_successes.ndim != 2 or treatment_successes.ndim != 2:
            raise ValueError("success arrays must have shape (rows,2 endpoints)")
        if control_successes.shape[0] > _MAX_TAIL_ROWS:
            raise ValueError(f"tail batches are limited to {_MAX_TAIL_ROWS} rows")
        c, t = np.asarray(control_successes), np.asarray(treatment_successes)
        if c.ndim != 2 or c.shape[1:] != (2,) or t.shape != c.shape:
            raise ValueError("success arrays must have shape (rows,2 endpoints)")
        estimated_work = c.shape[0] * 4 * 2 * 3 * _QUAD_NODES * _QUAD_INTERVAL_FACTOR
        if estimated_work > _MAX_TAIL_COMPARISON_WORK:
            raise ValueError("tail batch exceeds the bounded comparison-work budget")
        if np.iscomplexobj(c) or np.iscomplexobj(t):
            raise ValueError("success arrays must be real")
        c, t = count(c, "control_successes"), count(t, "treatment_successes")
        if np.any(c > control_n) or np.any(t > treatment_n):
            raise ValueError("endpoint successes cannot exceed arm sample size")
        probabilities = np.empty((c.shape[0], 2, 2), dtype=np.float64)
        errors = np.empty_like(probabilities)
        for endpoint_index in range(2):
            if endpoint_index == 0:
                c_s, c_f = (
                    self.control_prior[0] + self.control_prior[1],
                    self.control_prior[2] + self.control_prior[3],
                )
                t_s, t_f = (
                    self.treatment_prior[0] + self.treatment_prior[1],
                    self.treatment_prior[2] + self.treatment_prior[3],
                )
            else:
                c_s, c_f = (
                    self.control_prior[0] + self.control_prior[2],
                    self.control_prior[1] + self.control_prior[3],
                )
                t_s, t_f = (
                    self.treatment_prior[0] + self.treatment_prior[2],
                    self.treatment_prior[1] + self.treatment_prior[3],
                )
            control = BetaBinomialPosterior(
                c_s + c[:, endpoint_index], c_f + control_n - c[:, endpoint_index]
            )
            treatment = BetaBinomialPosterior(
                t_s + t[:, endpoint_index], t_f + treatment_n - t[:, endpoint_index]
            )
            for criterion, margin in enumerate(
                (self.lrv[endpoint_index], self.cmv[endpoint_index])
            ):
                if self.endpoint == "efficacy_toxicity" and endpoint_index == 1:
                    comparison = compare_beta_difference(
                        treatment,
                        control,
                        -float(margin),
                        absolute_tolerance=self.comparison_tolerance,
                    )
                else:
                    comparison = compare_beta_difference(
                        control,
                        treatment,
                        float(margin),
                        absolute_tolerance=self.comparison_tolerance,
                    )
                probabilities[:, endpoint_index, criterion] = comparison.above_margin
                errors[:, endpoint_index, criterion] = comparison.absolute_error
        return _owned(probabilities), _owned(errors)

    def _paired_decisions_from_tails(
        self, probability: ArrayLike, errors: ArrayLike, *, total_n: int
    ) -> NDArray[np.str_]:
        """Classify cached ``(rows, endpoint, criterion)`` tail tables."""
        return _paired_decisions_from_tails(
            probability,
            errors,
            total_n=total_n,
            max_subjects=self.max_subjects,
            looks=self.looks,
            endpoint=self.endpoint,
            lambda_lrv=self.lambda_lrv,
            lambda_cmv=self.lambda_cmv,
            gamma_lrv=self.gamma_lrv,
            gamma_cmv=self.gamma_cmv,
            graduate_at_interim=self.graduate_at_interim,
        )

    def monitor(
        self, control_counts: ArrayLike, treatment_counts: ArrayLike
    ) -> BOP2DCRandomizedPairedState:
        c = _category_vector(control_counts, "control_counts")
        t = _category_vector(treatment_counts, "treatment_counts")
        control_n, treatment_n = int(np.sum(c)), int(np.sum(t))
        total_n = control_n + treatment_n
        if total_n > self.max_subjects:
            raise ValueError("category counts exceed max_subjects")
        expected_control = int(np.count_nonzero(self.arm_assignments[:total_n] == 0))
        if control_n != expected_control:
            raise ValueError("arm counts do not match the fixed allocation prefix")
        c_success = np.array([[c[0] + c[1], c[0] + c[2]]], dtype=np.int64)
        t_success = np.array([[t[0] + t[1], t[0] + t[2]]], dtype=np.int64)
        probabilities, errors = self._posterior_tails_from_success_counts(
            control_n, treatment_n, c_success, t_success
        )
        labels = self._paired_decisions_from_tails(probabilities, errors, total_n=total_n)
        if not np.any(self.looks == total_n):
            endpoint_decisions = ("continue", "continue")
        else:
            endpoint_decisions_list = []
            for endpoint_index in range(2):
                action = int(
                    _endpoint_actions(
                        probabilities[:, endpoint_index, :],
                        total_n,
                        self.max_subjects,
                        float(self.lambda_lrv[endpoint_index]),
                        float(self.lambda_cmv[endpoint_index]),
                        float(self.gamma_lrv[endpoint_index]),
                        float(self.gamma_cmv[endpoint_index]),
                        self.graduate_at_interim,
                    )[0]
                )
                endpoint_decisions_list.append(
                    (
                        "continue" if total_n < self.max_subjects else "consider",
                        "no-go",
                        "graduate" if total_n < self.max_subjects else "go",
                    )[action]
                )
            endpoint_decisions = (endpoint_decisions_list[0], endpoint_decisions_list[1])
        return BOP2DCRandomizedPairedState(
            total_n,
            control_n,
            treatment_n,
            _owned(c, dtype=np.int64),
            _owned(t, dtype=np.int64),
            _owned(probabilities[0]),
            _owned(errors[0]),
            endpoint_decisions,  # type: ignore[arg-type]
            str(labels[0]),
        )

    def replay(self, outcomes: ArrayLike) -> BOP2DCRandomizedPairedReplay:
        tape = _integer_tape(outcomes, self.max_subjects, "outcomes", 3)
        prefix_work = int(np.sum(self.looks, dtype=np.int64))
        comparison_work = int(self.looks.size) * 4 * 2 * 3 * _QUAD_NODES * _QUAD_INTERVAL_FACTOR
        if prefix_work > _MAX_PREFIX_WORK or prefix_work + comparison_work > _MAX_REPLAY_WORK:
            raise ValueError("paired randomized replay exceeds its prefix/comparison work budget")
        counts = np.zeros((2, 4), dtype=np.int64)
        states: list[BOP2DCRandomizedPairedState] = []
        terminal = "continue"
        used = next_look = 0
        for index, (arm, category) in enumerate(zip(self.arm_assignments, tape, strict=True)):
            used = index + 1
            counts[int(arm), int(category)] += 1
            if used != int(self.looks[next_look]):
                continue
            state = self.monitor(counts[0], counts[1])
            states.append(state)
            terminal = state.decision
            if terminal != "continue":
                break
            next_look += 1
        return BOP2DCRandomizedPairedReplay(
            _owned(self.arm_assignments[:used], dtype=np.int64),
            _owned(tape[:used], dtype=np.int64),
            tuple(states),
            terminal,
        )


def bop2_dc_randomized_paired_design(
    max_subjects: int,
    endpoint: str,
    lrv: ArrayLike,
    cmv: ArrayLike,
    *,
    control_prior: ArrayLike,
    treatment_prior: ArrayLike,
    arm_assignments: ArrayLike,
    looks: ArrayLike,
    lambda_lrv: ArrayLike = (0.9, 0.9),
    lambda_cmv: ArrayLike = (0.5, 0.5),
    gamma_lrv: ArrayLike = (0.5, 0.5),
    gamma_cmv: ArrayLike = (0.5, 0.5),
    graduate_at_interim: bool = False,
    comparison_tolerance: float = 1e-8,
) -> BOP2DCRandomizedPairedDesign:
    """Construct a paired binary-endpoint design with caller-specified priors.

    Category order is both, endpoint-1-only, endpoint-2-only, neither. For
    efficacy/toxicity, endpoint 1 favors higher risks and endpoint 2 favors
    lower treatment-minus-control toxicity. Optional paired graduation follows
    the Python composition of the source OBF rule with the documented OR/AND
    endpoint action rules; the native app's paired-graduation behavior was not
    established by the available source.
    """
    raw_n = scalar(max_subjects, "max_subjects")
    n = int(raw_n)
    if raw_n != n or not 2 <= n <= _MAX_SUBJECTS:
        raise ValueError(f"max_subjects must be an integer in [2,{_MAX_SUBJECTS}]")
    if endpoint not in ("multiple_efficacy", "efficacy_toxicity"):
        raise ValueError("endpoint must be 'multiple_efficacy' or 'efficacy_toxicity'")
    lower, upper = _bounded_vector(lrv, "lrv", 2), _bounded_vector(cmv, "cmv", 2)
    if np.any((lower < -1) | (lower > 1) | (upper < -1) | (upper > 1)):
        raise ValueError("risk-difference margins must lie in [-1,1]")
    if endpoint == "multiple_efficacy" and np.any(upper <= lower):
        raise ValueError("efficacy endpoints require cmv > lrv")
    if endpoint == "efficacy_toxicity" and (upper[0] <= lower[0] or upper[1] >= lower[1]):
        raise ValueError("require efficacy cmv > lrv and toxicity cmv < lrv")
    cprior = _dirichlet_prior(control_prior, "control_prior")
    tprior = _dirichlet_prior(treatment_prior, "treatment_prior")
    assignments = _integer_tape(arm_assignments, n, "arm_assignments", 1)
    if not np.any(assignments == 0) or not np.any(assignments == 1):
        raise ValueError("arm_assignments must include control and treatment")
    if isinstance(looks, np.ndarray):
        shape = looks.shape
    elif isinstance(looks, (list, tuple)):
        if not 1 <= len(looks) <= min(_MAX_LOOKS, n) or any(not np.isscalar(x) for x in looks):
            raise ValueError("looks must be a bounded increasing schedule")
        shape = (len(looks),)
    else:
        raise ValueError("looks must be a bounded one-dimensional schedule")
    if len(shape) != 1 or not 1 <= int(shape[0]) <= min(_MAX_LOOKS, n):
        raise ValueError("looks must be a bounded increasing schedule")
    schedule = _integer_tape(looks, int(shape[0]), "looks", n)
    if np.any(schedule < 1) or schedule[-1] != n or np.any(np.diff(schedule) <= 0):
        raise ValueError("looks must increase strictly and end at max_subjects")
    ll = _endpoint_values(lambda_lrv, "lambda_lrv", 0, 1)
    lc = _endpoint_values(lambda_cmv, "lambda_cmv", 0, 1)
    gl = _endpoint_values(gamma_lrv, "gamma_lrv", 0, 1)
    gc = _endpoint_values(gamma_cmv, "gamma_cmv", 0, 1)
    if np.any((ll <= 0) | (ll >= 1) | (lc <= 0) | (lc >= 1)):
        raise ValueError("lambda cutoffs must lie in (0,1)")
    tol = scalar(comparison_tolerance, "comparison_tolerance")
    if not 1e-12 <= tol <= 1e-3:
        raise ValueError("comparison_tolerance must lie in [1e-12,1e-3]")
    if not isinstance(graduate_at_interim, (bool, np.bool_)):
        raise ValueError("graduate_at_interim must be boolean")
    fraction = int(schedule[0]) / n
    with np.errstate(under="ignore"):
        if np.any(ll * fraction**gl == 0) or np.any(lc * fraction**gc == 0):
            raise ArithmeticError("an interim cutoff underflows")
    prefix_work = int(np.sum(schedule, dtype=np.int64))
    comparison_work = int(schedule.size) * 4 * 2 * 3 * _QUAD_NODES * _QUAD_INTERVAL_FACTOR
    if prefix_work > _MAX_PREFIX_WORK or prefix_work + comparison_work > _MAX_REPLAY_WORK:
        raise ValueError("paired randomized replay exceeds its prefix/comparison work budget")
    return BOP2DCRandomizedPairedDesign(
        n,
        endpoint,
        _owned(lower),
        _owned(upper),
        cprior,
        tprior,
        _owned(ll),
        _owned(lc),
        _owned(gl),
        _owned(gc),
        assignments,
        schedule,
        bool(graduate_at_interim),
        tol,
    )
