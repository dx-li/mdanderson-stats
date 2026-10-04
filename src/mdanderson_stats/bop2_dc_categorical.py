"""BOP2-DC monitoring for joint categorical endpoints with binary indicators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betainc, betaincc

from ._bop2_dc_randomized_rules import randomized_obf_cutoffs
from ._validation import FloatArray, count, finite, scalar
from .beta_binomial import BetaBinomialPosterior
from .beta_comparison import compare_beta_difference

type IntArray = NDArray[np.int64]
_MAX_SUBJECTS = 1_000
_MAX_CATEGORIES = 256
_MAX_ENDPOINTS = 16
_MAX_LOOKS = 100
_MAX_TAIL_WORK = 100_000_000
_MAX_REPLAY_WORK = 50_000_000
_ACTIONS = ("continue", "stop_no_go", "graduate", "final_go", "final_consider", "final_no_go")


def _freeze(value: ArrayLike, dtype: np.dtype | type = np.float64) -> NDArray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _positive_int(value: int, name: str, maximum: int) -> int:
    if not np.isscalar(value) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be an integer scalar")
    raw = scalar(value, name)
    integer = int(raw)
    if raw != integer or not 1 <= integer <= maximum:
        raise ValueError(f"{name} must be an integer in [1,{maximum}]")
    return integer


def _indicator_matrix(value: ArrayLike) -> NDArray[np.bool_]:
    if isinstance(value, np.ndarray):
        shape = value.shape
        if value.size > 4_096 or np.iscomplexobj(value):
            raise ValueError("indicator matrix is too large or complex")
    elif isinstance(value, (list, tuple)):
        if not value or len(value) > _MAX_ENDPOINTS:
            raise ValueError(f"indicators must have 1..{_MAX_ENDPOINTS} rows")
        if not isinstance(value[0], (list, tuple, np.ndarray)):
            raise ValueError("indicators must be a two-dimensional binary matrix")
        shape = (len(value), len(value[0]))
        if not 2 <= shape[1] <= _MAX_CATEGORIES or len(value) * shape[1] > 4_096:
            raise ValueError("indicator matrix is too large")
        for row in value:
            if not isinstance(row, (list, tuple, np.ndarray)) or len(row) != shape[1]:
                raise ValueError("indicator rows must have the same bounded category count")
            if isinstance(row, np.ndarray) and (row.shape != (shape[1],) or np.iscomplexobj(row)):
                raise ValueError("indicators must be real")
            if not isinstance(row, np.ndarray) and any(
                not np.isscalar(item) or np.iscomplexobj(item) for item in row
            ):
                raise ValueError("indicator rows must contain scalar real values")
    else:
        raise ValueError("indicators must be a two-dimensional binary matrix")
    if (
        len(shape) != 2
        or not 1 <= shape[0] <= _MAX_ENDPOINTS
        or not 2 <= shape[1] <= _MAX_CATEGORIES
    ):
        raise ValueError("indicators must have shape (1..16 endpoints, 2..256 categories)")
    if int(shape[0]) * int(shape[1]) > 4_096 or np.iscomplexobj(value):
        raise ValueError("indicator matrix is too large or complex")
    raw = finite(value, "indicators")
    if raw.shape != shape or np.any((raw != 0) & (raw != 1)):
        raise ValueError("indicators must contain only zero and one")
    result = np.asarray(raw, dtype=bool)
    if np.any(np.all(result, axis=1)) or np.any(~np.any(result, axis=1)):
        raise ValueError("each endpoint indicator must select some but not all categories")
    return _freeze(result, dtype=np.bool_)


def _prior(value: ArrayLike, name: str, categories: int) -> tuple[float, ...]:
    if isinstance(value, np.ndarray):
        if value.shape != (categories,) or np.iscomplexobj(value):
            raise ValueError(f"{name} must contain {categories} positive Dirichlet shapes")
    elif isinstance(value, (list, tuple)):
        if len(value) != categories or any(not np.isscalar(x) or np.iscomplexobj(x) for x in value):
            raise ValueError(f"{name} must contain {categories} positive Dirichlet shapes")
    else:
        raise ValueError(f"{name} must contain {categories} positive Dirichlet shapes")
    raw = finite(value, name)
    if raw.shape != (categories,) or np.any(raw <= 0):
        raise ValueError(f"{name} must contain {categories} positive Dirichlet shapes")
    total = fsum(float(x) for x in raw)
    if not np.isfinite(total):
        raise ValueError(f"{name} total must be finite")
    return tuple(float(x) for x in raw)


def _endpoint_vector(
    value: ArrayLike | float, name: str, m: int, low: float, high: float
) -> FloatArray:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value) or value.size > m:
            raise ValueError(f"{name} must be a scalar or length-{m} real input")
    elif isinstance(value, (list, tuple)):
        if len(value) > m or any(not np.isscalar(x) or np.iscomplexobj(x) for x in value):
            raise ValueError(f"{name} must be a scalar or length-{m} real input")
    elif np.isscalar(value) and not np.iscomplexobj(value):
        pass
    else:
        raise ValueError(f"{name} must be a scalar or length-{m} real input")
    raw = finite(value, name)
    if raw.ndim == 0:
        result = np.full(m, float(raw))
    elif raw.shape == (m,):
        result = np.array(raw, dtype=np.float64, copy=True)
    else:
        raise ValueError(f"{name} must be a scalar or length-{m} vector")
    if np.any((result < low) | (result > high)):
        raise ValueError(f"{name} values must lie in [{low},{high}]")
    return result


def _settings(value: ArrayLike | float, name: str, m: int, *, cutoff: bool) -> FloatArray:
    low, high = (0.0, 1.0) if cutoff else (0.0, 1.0)
    result = _endpoint_vector(value, name, m, low, high)
    if cutoff and np.any((result <= 0) | (result >= 1)):
        raise ValueError(f"{name} must lie strictly between zero and one")
    return _freeze(result)


def _looks(value: ArrayLike | None, n: int) -> IntArray:
    if value is None:
        return _freeze([n], dtype=np.int64)
    if isinstance(value, np.ndarray):
        if value.size > min(n, _MAX_LOOKS) or value.ndim != 1 or np.iscomplexobj(value):
            raise ValueError("looks must be a bounded one-dimensional schedule")
    elif isinstance(value, (list, tuple)):
        if not 1 <= len(value) <= min(n, _MAX_LOOKS) or any(
            not np.isscalar(x) or np.iscomplexobj(x) for x in value
        ):
            raise ValueError("looks must be a bounded one-dimensional schedule")
    else:
        raise ValueError("looks must be a bounded one-dimensional schedule")
    raw = count(value, "looks")
    if raw.ndim != 1 or not 1 <= raw.size <= min(n, _MAX_LOOKS):
        raise ValueError("looks must be a bounded one-dimensional schedule")
    result = raw.astype(np.int64)
    if np.any(result < 1) or np.any(result > n) or result[-1] != n or np.any(np.diff(result) <= 0):
        raise ValueError("looks must increase strictly and end at max_subjects")
    return _freeze(result, dtype=np.int64)


def _category_tape(value: ArrayLike, n: int, categories: int, name: str) -> IntArray:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) != n or any(not np.isscalar(x) for x in value):
            raise ValueError(f"{name} must have one category per planned subject")
        shape = (len(value),)
    else:
        raise ValueError(f"{name} must be a bounded one-dimensional tape")
    if shape != (n,) or np.iscomplexobj(value):
        raise ValueError(f"{name} must have one real category per planned subject")
    result = count(value, name)
    if np.any(result >= categories):
        raise ValueError(f"{name} categories must be in [0,{categories - 1}]")
    return _freeze(result, dtype=np.int64)


def _count_matrix(value: ArrayLike, categories: int, randomized: bool, maximum: int) -> IntArray:
    expected = (2, categories) if randomized else (categories,)
    if isinstance(value, np.ndarray):
        if value.shape != expected or np.iscomplexobj(value):
            raise ValueError(f"category_counts must have shape {expected}")
    elif isinstance(value, (list, tuple)):
        if randomized:
            if len(value) != 2 or any(
                not isinstance(row, (list, tuple, np.ndarray)) or len(row) != categories
                for row in value
            ):
                raise ValueError(f"category_counts must have shape {expected}")
            if any(
                (
                    isinstance(row, np.ndarray)
                    and (row.shape != (categories,) or np.iscomplexobj(row))
                )
                or (
                    not isinstance(row, np.ndarray)
                    and any(not np.isscalar(item) or np.iscomplexobj(item) for item in row)
                )
                for row in value
            ):
                raise ValueError("category_counts must be real")
        elif len(value) != categories or any(
            not np.isscalar(item) or np.iscomplexobj(item) for item in value
        ):
            raise ValueError(f"category_counts must have shape {expected}")
    else:
        raise ValueError(f"category_counts must have shape {expected}")
    raw = count(value, "category_counts")
    if raw.shape != expected or np.any(raw > maximum):
        raise ValueError(f"category_counts must have shape {expected} with counts <= {maximum}")
    return np.asarray(raw, dtype=np.int64)


def _endpoint_actions(
    probabilities: FloatArray,
    *,
    total_n: int,
    design: BOP2DCCategoricalDesign,
) -> NDArray[np.int8]:
    """Map each endpoint's two favorable posterior tails to no-go/neutral/go."""
    final = total_n == design.max_subjects
    actions: NDArray[np.int8] = np.zeros(design.n_endpoints, dtype=np.int8)
    if final:
        no_go = (probabilities[:, 0] < design.lambda_lrv) & (
            probabilities[:, 1] < design.lambda_cmv
        )
        go = (probabilities[:, 0] > design.lambda_lrv) & (probabilities[:, 1] > design.lambda_cmv)
    else:
        fraction = total_n / design.max_subjects
        no_go = (probabilities[:, 0] < design.lambda_lrv * fraction**design.gamma_lrv) & (
            probabilities[:, 1] < design.lambda_cmv * fraction**design.gamma_cmv
        )
        if design.graduate_at_interim:
            if design.randomized:
                cutoffs = [
                    randomized_obf_cutoffs(
                        look=total_n,
                        max_subjects=design.max_subjects,
                        lambda_lrv=float(design.lambda_lrv[j]),
                        lambda_cmv=float(design.lambda_cmv[j]),
                    )
                    for j in range(design.n_endpoints)
                ]
                grad_lrv = np.array([item[0] for item in cutoffs])
                grad_cmv = np.array([item[1] for item in cutoffs])
            else:
                # Single-arm graduation uses the fixed final cutoffs (§2.2).
                grad_lrv, grad_cmv = design.lambda_lrv, design.lambda_cmv
            go = (probabilities[:, 0] > grad_lrv) & (probabilities[:, 1] > grad_cmv)
            if np.any(no_go & go):
                raise ValueError("interim graduation and no-go cutoffs overlap")
        else:
            go = np.zeros(design.n_endpoints, dtype=bool)
    actions[no_go], actions[go] = 1, 2
    return actions


def _combine_actions(actions: NDArray[np.int8], combination: str, final: bool) -> int:
    no_go, go = actions == 1, actions == 2
    if combination == "any":
        if np.any(go):
            return 2
        if np.all(no_go):
            return 1
    else:
        if np.any(no_go):
            return 1
        if np.all(go):
            return 2
    return 0


def _label(code: int, final: bool, graduate: bool) -> str:
    if final:
        return ("final_consider", "final_no_go", "final_go")[code]
    return ("continue", "stop_no_go", "graduate" if graduate else "continue")[code]


@dataclass(frozen=True)
class BOP2DCCategoricalState:
    """Posterior margins and combined action at one scheduled total-N look."""

    total_n: int
    category_counts: IntArray
    posterior_probability: FloatArray
    absolute_error: FloatArray
    endpoint_decisions: tuple[str, ...]
    decision: str


@dataclass(frozen=True)
class BOP2DCCategoricalReplay:
    """Observed category/arm prefixes and analyses through the first terminal look."""

    outcomes_observed: IntArray
    arm_assignments_observed: IntArray | None
    states: tuple[BOP2DCCategoricalState, ...]
    terminal_decision: str


@dataclass(frozen=True)
class BOP2DCCategoricalDesign:
    """Single-arm or fixed-allocation randomized categorical BOP2-DC design."""

    max_subjects: int
    indicators: NDArray[np.bool_]
    combination: str
    directions: tuple[str, ...]
    lrv: FloatArray
    cmv: FloatArray
    prior: tuple[float, ...]
    control_prior: tuple[float, ...] | None
    arm_assignments: IntArray | None
    looks: IntArray
    lambda_lrv: FloatArray
    lambda_cmv: FloatArray
    gamma_lrv: FloatArray
    gamma_cmv: FloatArray
    graduate_at_interim: bool
    comparison_tolerance: float

    @property
    def randomized(self) -> bool:
        return self.control_prior is not None

    @property
    def n_endpoints(self) -> int:
        return int(self.indicators.shape[0])

    @property
    def n_categories(self) -> int:
        return int(self.indicators.shape[1])

    def _marginal(
        self, counts: NDArray[np.int64], prior: tuple[float, ...], endpoint: int
    ) -> tuple[float, float]:
        selected = self.indicators[endpoint]
        a = fsum(prior[i] for i in range(self.n_categories) if selected[i])
        b = fsum(prior[i] for i in range(self.n_categories) if not selected[i])
        success = int(np.dot(counts, selected.astype(np.int64)))
        n = int(np.sum(counts, dtype=np.int64))
        return a + success, b + n - success

    def _posterior(
        self,
        counts: NDArray[np.int64],
        comparison_cache: dict[tuple[float | int, ...], tuple[float, float, float, float]]
        | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        probabilities: FloatArray = np.empty((self.n_endpoints, 2), dtype=np.float64)
        errors = np.zeros_like(probabilities)
        if not self.randomized:
            for j in range(self.n_endpoints):
                a, b = self._marginal(counts, self.prior, j)
                if not np.isfinite(a + b):
                    raise ArithmeticError("single-arm beta posterior shapes overflow")
                for criterion, margin in enumerate((self.lrv[j], self.cmv[j])):
                    if a == b and margin == 0.5:
                        # Preserve the exact symmetry at the strict decision boundary.
                        probabilities[j, criterion] = 0.5
                    else:
                        cdf = float(betainc(a, b, margin))
                        probabilities[j, criterion] = (
                            cdf if self.directions[j] == "less" else float(betaincc(a, b, margin))
                        )
            if not np.all(np.isfinite(probabilities)):
                raise ArithmeticError("single-arm beta posterior probability is not finite")
            return probabilities, errors
        assert self.control_prior is not None
        for j in range(self.n_endpoints):
            ca, cb = self._marginal(counts[0], self.control_prior, j)
            ta, tb = self._marginal(counts[1], self.prior, j)
            if not all(np.isfinite(x) for x in (ca + cb, ta + tb)):
                raise ArithmeticError("randomized beta posterior shapes overflow")
            control, treatment = BetaBinomialPosterior(ca, cb), BetaBinomialPosterior(ta, tb)
            cache_key: tuple[float | int, ...] = (j, ca, cb, ta, tb)
            cached = None if comparison_cache is None else comparison_cache.get(cache_key)
            if cached is not None:
                probabilities[j] = (cached[0], cached[2])
                errors[j] = (cached[1], cached[3])
                continue
            endpoint_values: list[float] = []
            for criterion, margin in enumerate((self.lrv[j], self.cmv[j])):
                result = compare_beta_difference(
                    control, treatment, float(margin), absolute_tolerance=self.comparison_tolerance
                )
                favorable = (
                    result.below_margin if self.directions[j] == "less" else result.above_margin
                )
                probabilities[j, criterion] = float(favorable)
                errors[j, criterion] = float(result.absolute_error)
                endpoint_values.extend((float(favorable), float(result.absolute_error)))
            if comparison_cache is not None:
                comparison_cache[cache_key] = (
                    endpoint_values[0],
                    endpoint_values[1],
                    endpoint_values[2],
                    endpoint_values[3],
                )
        if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(errors)):
            raise ArithmeticError("categorical BOP2-DC posterior comparison is not finite")
        return probabilities, errors

    def _combined_code(self, probability: FloatArray, total_n: int) -> int:
        actions = _endpoint_actions(probability, total_n=total_n, design=self)
        return _combine_actions(actions, self.combination, total_n == self.max_subjects)

    def _guarded_decision(self, probabilities: FloatArray, errors: FloatArray, total_n: int) -> str:
        if not np.any(self.looks == total_n):
            return "continue"
        if np.any(errors < 0) or np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ArithmeticError("posterior probabilities/errors are outside their valid range")
        lower = np.maximum(0, probabilities - errors)
        upper = np.minimum(1, probabilities + errors)
        low_action = self._combined_code(lower, total_n)
        high_action = self._combined_code(upper, total_n)
        if low_action != high_action:
            raise ArithmeticError("posterior numerical error could change the combined decision")
        return _label(low_action, total_n == self.max_subjects, self.graduate_at_interim)

    def monitor(self, category_counts: ArrayLike) -> BOP2DCCategoricalState:
        return self._monitor(category_counts, None)

    def _monitor(
        self,
        category_counts: ArrayLike,
        comparison_cache: dict[tuple[float | int, ...], tuple[float, float, float, float]] | None,
    ) -> BOP2DCCategoricalState:
        counts_array = _count_matrix(
            category_counts, self.n_categories, self.randomized, self.max_subjects
        )
        total_n = int(np.sum(counts_array, dtype=np.int64))
        if total_n > self.max_subjects:
            raise ValueError("category counts exceed max_subjects")
        if self.randomized:
            assert self.arm_assignments is not None
            expected_control = int(np.count_nonzero(self.arm_assignments[:total_n] == 0))
            if int(np.sum(counts_array[0])) != expected_control:
                raise ValueError("arm category counts do not match the fixed allocation prefix")
        probabilities, errors = self._posterior(counts_array, comparison_cache)
        decision = self._guarded_decision(probabilities, errors, total_n)
        if not np.any(self.looks == total_n):
            endpoint_actions: NDArray[np.int8] = np.zeros(self.n_endpoints, dtype=np.int8)
        else:
            endpoint_actions = _endpoint_actions(probabilities, total_n=total_n, design=self)
        final = total_n == self.max_subjects
        endpoint_labels = tuple(
            ("continue", "no_go", "go" if final else "graduate")[int(action)]
            if action
            else ("consider" if final else "continue")
            for action in endpoint_actions
        )
        return BOP2DCCategoricalState(
            total_n,
            _freeze(counts_array, dtype=np.int64),
            _freeze(probabilities),
            _freeze(errors),
            endpoint_labels,
            decision,
        )

    def replay(self, category_outcomes: ArrayLike) -> BOP2DCCategoricalReplay:
        tape = _category_tape(
            category_outcomes, self.max_subjects, self.n_categories, "category_outcomes"
        )
        prefix_work = int(np.sum(self.looks, dtype=np.int64)) * self.n_categories
        tail_work = (
            int(self.looks.size) * self.n_endpoints * 2 * 4 * 3 * 21 * 599 if self.randomized else 0
        )
        if prefix_work > _MAX_REPLAY_WORK or prefix_work + tail_work > _MAX_REPLAY_WORK:
            raise ValueError("categorical replay exceeds its bounded work budget")
        counts: NDArray[np.int64] = (
            np.zeros((2, self.n_categories), dtype=np.int64)
            if self.randomized
            else np.zeros(self.n_categories, dtype=np.int64)
        )
        states: list[BOP2DCCategoricalState] = []
        assignments = self.arm_assignments
        look_index = 0
        terminal = "continue"
        used = 0
        for i, category in enumerate(tape):
            used = i + 1
            if self.randomized:
                assert assignments is not None
                counts[int(assignments[i]), int(category)] += 1
            else:
                counts[int(category)] += 1
            if used != int(self.looks[look_index]):
                continue
            state = self.monitor(counts)
            states.append(state)
            terminal = state.decision
            if terminal != "continue":
                break
            look_index += 1
        return BOP2DCCategoricalReplay(
            _freeze(tape[:used], dtype=np.int64),
            None if assignments is None else _freeze(assignments[:used], dtype=np.int64),
            tuple(states),
            terminal,
        )


def bop2_dc_categorical_design(
    max_subjects: int,
    indicators: ArrayLike,
    *,
    combination: str,
    directions: Sequence[str] | str,
    lrv: ArrayLike | float,
    cmv: ArrayLike | float,
    prior: ArrayLike,
    control_prior: ArrayLike | None = None,
    arm_assignments: ArrayLike | None = None,
    looks: ArrayLike | None = None,
    lambda_lrv: ArrayLike | float = 0.9,
    lambda_cmv: ArrayLike | float = 0.5,
    gamma_lrv: ArrayLike | float = 0.5,
    gamma_cmv: ArrayLike | float = 0.5,
    graduate_at_interim: bool = False,
    comparison_tolerance: float = 1e-8,
) -> BOP2DCCategoricalDesign:
    """Construct a BOP2-DC design for arbitrary binary endpoint indicators.

    ``indicators`` has one 0/1 row per endpoint and one column per joint
    category. ``prior`` is the single-arm or treatment-arm Dirichlet shape
    vector; supplying ``control_prior`` selects fixed-allocation RCT mode.
    """
    n = _positive_int(max_subjects, "max_subjects", _MAX_SUBJECTS)
    matrix = _indicator_matrix(indicators)
    m, k = matrix.shape
    if combination not in ("any", "all"):
        raise ValueError("combination must be 'any' or 'all'")
    if isinstance(directions, str):
        direction_values = (directions,) * m
    elif isinstance(directions, Sequence) and len(directions) == m:
        direction_values = tuple(directions)
    elif isinstance(directions, np.ndarray) and directions.shape == (m,) and directions.size <= m:
        direction_values = tuple(directions.tolist())
    else:
        raise ValueError("directions must be 'greater'/'less' per endpoint")
    if len(direction_values) != m or any(x not in ("greater", "less") for x in direction_values):
        raise ValueError("directions must be 'greater'/'less' per endpoint")
    max_margin = 1.0 if control_prior is None else 1.0
    lower = _endpoint_vector(
        lrv, "lrv", m, -max_margin if control_prior is not None else 0, max_margin
    )
    upper = _endpoint_vector(
        cmv, "cmv", m, -max_margin if control_prior is not None else 0, max_margin
    )
    for j, direction in enumerate(direction_values):
        if (direction == "greater" and upper[j] <= lower[j]) or (
            direction == "less" and upper[j] >= lower[j]
        ):
            raise ValueError("CMV must be more favorable than LRV in each endpoint direction")
    tprior = _prior(prior, "prior", k)
    cprior = None if control_prior is None else _prior(control_prior, "control_prior", k)
    if (cprior is None) != (arm_assignments is None):
        raise ValueError("control_prior and arm_assignments must be supplied together")
    randomized = cprior is not None
    if randomized:
        assert arm_assignments is not None
        if isinstance(arm_assignments, np.ndarray):
            shape = arm_assignments.shape
            if arm_assignments.size > n or np.iscomplexobj(arm_assignments):
                raise ValueError("arm_assignments must be a one-dimensional 0/1 tape")
        elif isinstance(arm_assignments, (list, tuple)):
            if len(arm_assignments) != n or any(
                not np.isscalar(x) or np.iscomplexobj(x) for x in arm_assignments
            ):
                raise ValueError("arm_assignments must be a one-dimensional 0/1 tape")
            shape = (len(arm_assignments),)
        else:
            raise ValueError("arm_assignments must be a one-dimensional 0/1 tape")
        if shape != (n,):
            raise ValueError("arm_assignments must have one 0/1 value per planned subject")
        assignments_raw = count(arm_assignments, "arm_assignments")
        if np.any(assignments_raw > 1):
            raise ValueError("arm_assignments must contain only zero (control) and one (treatment)")
        assignments = _freeze(assignments_raw, dtype=np.int64)
        if not np.any(assignments == 0) or not np.any(assignments == 1):
            raise ValueError("arm_assignments must include both arms")
    else:
        assignments = None
    schedule = _looks(looks, n)
    ll = _settings(lambda_lrv, "lambda_lrv", m, cutoff=True)
    lc = _settings(lambda_cmv, "lambda_cmv", m, cutoff=True)
    gl = _settings(gamma_lrv, "gamma_lrv", m, cutoff=False)
    gc = _settings(gamma_cmv, "gamma_cmv", m, cutoff=False)
    if not isinstance(graduate_at_interim, (bool, np.bool_)):
        raise ValueError("graduate_at_interim must be boolean")
    tolerance = scalar(comparison_tolerance, "comparison_tolerance")
    if not 1e-12 <= tolerance <= 1e-3:
        raise ValueError("comparison_tolerance must lie in [1e-12,1e-3]")
    for look in schedule[:-1]:
        f = int(look) / n
        if np.any(ll * f**gl == 0) or np.any(lc * f**gc == 0):
            raise ArithmeticError("interim cutoff underflows")
        if graduate_at_interim and randomized:
            for j in range(m):
                randomized_obf_cutoffs(
                    look=int(look), max_subjects=n, lambda_lrv=float(ll[j]), lambda_cmv=float(lc[j])
                )
    estimated_tail_work = m * 2 * 2 * 3 * 21 * 599 if randomized else 0
    if estimated_tail_work > _MAX_TAIL_WORK:
        raise ValueError("design exceeds the posterior-comparison work budget")
    return BOP2DCCategoricalDesign(
        n,
        matrix,
        combination,
        tuple(direction_values),
        _freeze(lower),
        _freeze(upper),
        tprior,
        cprior,
        assignments,
        schedule,
        ll,
        lc,
        gl,
        gc,
        bool(graduate_at_interim),
        tolerance,
    )
