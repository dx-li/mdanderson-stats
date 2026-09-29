"""Finite-grid exact calibration for randomized paired-endpoint BOP2-DC."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite, scalar
from .bop2_dc_randomized_paired import (
    BOP2DCRandomizedPairedDesign,
    bop2_dc_randomized_paired_design,
)

_MAX_CANDIDATES = 2_000
_MAX_TOTAL_WORK = 100_000_000
_MAX_RETAINED_CELLS = 2_000_000
_MAX_STATE_CELLS = 1_000_000
_MAX_TRANSITION_WORK = 20_000_000
_MAX_TAIL_WORK = 80_000_000
_MAX_TAIL_CELLS = 1_500_000
_STATE_CHUNK = 200
_PROBABILITY_SUM_TOLERANCE = 1e-14
_FLOAT_TIE_FACTOR = 64.0
_DECISIONS = ("stop_no_go", "graduate", "final_go", "final_consider", "final_no_go")


class BOP2DCRandomizedPairedInfeasibleError(ValueError):
    """No candidate on the supplied finite grid satisfies the OC limits."""


@dataclass(frozen=True)
class BOP2DCRandomizedPairedCandidateEvidence:
    """Exact operating characteristics for both truths and all grid candidates."""

    parameters: FloatArray
    parameter_names: tuple[str, ...]
    decision_probability: FloatArray
    sample_size_probability: FloatArray
    expected_sample_size: FloatArray
    false_go_rate: FloatArray
    false_no_go_rate: FloatArray
    correct_go_rate: FloatArray
    false_consider_rate: FloatArray
    feasible: NDArray[np.bool_]


@dataclass(frozen=True)
class BOP2DCRandomizedPairedOperatingCharacteristics:
    """Exact fixed-allocation OCs for one or more joint endpoint truths."""

    category_probability: FloatArray
    looks: NDArray[np.int64]
    decision_labels: tuple[str, ...]
    decision_probability: FloatArray
    sample_size_probability: FloatArray
    expected_sample_size: FloatArray
    exact_work_units: int


@dataclass(frozen=True)
class BOP2DCRandomizedPairedOptimization:
    """Selected randomized paired design and complete exact-grid evidence."""

    design: BOP2DCRandomizedPairedDesign
    candidates: BOP2DCRandomizedPairedCandidateEvidence
    selected_index: int
    objective: str
    endpoint: str
    futile_joint_probabilities: FloatArray
    effective_joint_probabilities: FloatArray
    futile_marginal_probabilities: FloatArray
    effective_marginal_probabilities: FloatArray
    decision_labels: tuple[str, ...]
    false_go_limit: float
    false_no_go_limit: float
    false_consider_limit: float | None
    exact_work_units: int


def _owned_readonly(value: ArrayLike, *, dtype: np.dtype | type | None = None) -> NDArray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _freeze_owned(value: NDArray) -> NDArray:
    value.flags.writeable = False
    return value


def _positive_integer(value: int, name: str, maximum: int) -> int:
    number = scalar(value, name)
    integer = int(number)
    if number != integer or integer < 1 or integer > maximum:
        raise ValueError(f"{name} must be an integer in [1,{maximum}]")
    return integer


def _control_grid(value: ArrayLike, name: str) -> FloatArray:
    """Accept scalar-grid rows (broadcast to endpoints) or explicit two-value rows."""
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if not value or len(value) > _MAX_CANDIDATES:
            raise ValueError(f"{name} must contain 1..{_MAX_CANDIDATES} rows")
        nested = any(not np.isscalar(item) for item in value)
        if nested:
            if any(
                not isinstance(row, (list, tuple, np.ndarray)) or len(row) != 2 for row in value
            ):
                raise ValueError(f"{name} must be a scalar grid or contain two-value rows")
            shape = (len(value), 2)
        else:
            shape = (len(value),)
    else:
        raise ValueError(f"{name} must be a one-dimensional or (m,2) grid")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    if len(shape) == 1 and 0 < shape[0] <= _MAX_CANDIDATES:
        values = np.asarray(finite(value, name), dtype=np.float64)
        result = np.repeat(values[:, None], 2, axis=1)
    elif len(shape) == 2 and shape[1] == 2 and 0 < shape[0] <= _MAX_CANDIDATES:
        result = np.array(finite(value, name), dtype=np.float64, copy=True)
    else:
        raise ValueError(f"{name} must be a nonempty scalar grid or an (m,2) row grid")
    result.flags.writeable = False
    return result


def _truth(value: ArrayLike, name: str) -> FloatArray:
    """Validate (control, treatment) joint rows in both/first/second/neither order."""
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) != 2 or any(
            not isinstance(row, (list, tuple, np.ndarray)) or len(row) != 4 for row in value
        ):
            raise ValueError(f"{name} must have shape (2,4)")
        shape = (2, 4)
    else:
        raise ValueError(f"{name} must have shape (2,4)")
    if shape != (2, 4) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a real array with shape (2,4)")
    result = np.array(finite(value, name), dtype=np.float64, copy=True)
    if np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} entries must lie in [0,1]")
    totals = result.sum(axis=1)
    if np.any(~np.isfinite(totals)) or np.any(np.abs(totals - 1.0) > _PROBABILITY_SUM_TOLERANCE):
        raise ValueError(f"each arm row in {name} must sum to one")
    result /= totals[:, None]
    result.flags.writeable = False
    return result


def _marginals(joint: FloatArray) -> FloatArray:
    # Category order is (both, endpoint 1 only, endpoint 2 only, neither).
    return np.array([[row[0] + row[1], row[0] + row[2]] for row in joint])


def _roundoff_tied(left: float, right: float, max_subjects: int) -> bool:
    tolerance = _FLOAT_TIE_FACTOR * np.finfo(np.float64).eps * max_subjects
    scale = max(abs(left), abs(right))
    return bool(left == right or (scale > 0 and abs(left - right) <= tolerance * scale))


def _within_limit(value: float, limit: float, max_subjects: int) -> bool:
    return bool(value <= limit or _roundoff_tied(value, limit, max_subjects))


def _select_candidate(
    objective: str,
    feasible: NDArray[np.bool_],
    correct_go: FloatArray,
    expected_n_futile: FloatArray,
    max_subjects: int,
) -> int:
    indices = np.flatnonzero(feasible)
    if not indices.size:
        raise BOP2DCRandomizedPairedInfeasibleError(
            "no candidate satisfies the supplied randomized paired false-go/no-go limits"
        )

    def precedes(first: int, second: int) -> bool:
        if objective == "cgr":
            first_scores = (correct_go[first], -expected_n_futile[first])
            second_scores = (correct_go[second], -expected_n_futile[second])
            maximize = True
        else:
            first_scores = (expected_n_futile[first], -correct_go[first])
            second_scores = (expected_n_futile[second], -correct_go[second])
            maximize = False
        for a, b in zip(first_scores, second_scores, strict=True):
            if _roundoff_tied(float(a), float(b), max_subjects):
                continue
            return a > b if maximize else a < b
        return first < second

    best = int(indices[0])
    for raw in indices[1:]:
        candidate = int(raw)
        if precedes(candidate, best):
            best = candidate
    return best


def _success_state_rows(
    control_n: int, treatment_n: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return flattened 2D marginal count pairs in C-order for one arm-count pair."""
    control = np.repeat(np.arange(control_n + 1, dtype=np.int64), treatment_n + 1)
    treatment = np.tile(np.arange(treatment_n + 1, dtype=np.int64), control_n + 1)
    return control, treatment


def _cached_tails(
    design: BOP2DCRandomizedPairedDesign,
    control_n: int,
    treatment_n: int,
) -> tuple[FloatArray, FloatArray]:
    """Cache endpoint/criterion tails on one pair of arm success-count axes."""
    cells = (control_n + 1) * (treatment_n + 1)
    probability = np.empty((control_n + 1, treatment_n + 1, 2, 2), dtype=np.float64)
    error = np.empty_like(probability)
    control, treatment = _success_state_rows(control_n, treatment_n)
    for start in range(0, cells, 200):
        stop = min(start + 200, cells)
        c = control[start:stop]
        t = treatment[start:stop]
        control_successes = np.repeat(c[:, None], 2, axis=1)
        treatment_successes = np.repeat(t[:, None], 2, axis=1)
        tails, errors = design._posterior_tails_from_success_counts(
            control_n, treatment_n, control_successes, treatment_successes
        )
        if tails.shape != (stop - start, 2, 2) or errors.shape != tails.shape:
            raise ArithmeticError("paired core returned unexpected posterior-tail shape")
        probability.reshape(cells, 2, 2)[start:stop] = tails
        error.reshape(cells, 2, 2)[start:stop] = errors
    return probability, error


def _validate_truth_scenarios(value: ArrayLike, name: str) -> FloatArray:
    """Validate a bounded batch of arm-specific joint category probabilities."""
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) == 2 and all(
            isinstance(row, (list, tuple, np.ndarray)) and len(row) == 4 for row in value
        ):
            shape = (2, 4)
        elif 1 <= len(value) <= 100 and all(
            isinstance(scenario, (list, tuple, np.ndarray))
            and len(scenario) == 2
            and all(
                isinstance(row, (list, tuple, np.ndarray)) and len(row) == 4 for row in scenario
            )
            for scenario in value
        ):
            shape = (len(value), 2, 4)
        else:
            raise ValueError(f"{name} must have shape (2,4) or (scenarios,2,4)")
    else:
        raise ValueError(f"{name} must have shape (2,4) or (scenarios,2,4)")
    if shape == (2, 4):
        pass
    elif len(shape) == 3 and shape[1:] == (2, 4) and 1 <= shape[0] <= 100:
        pass
    else:
        raise ValueError(f"{name} must have shape (2,4) or (1..100,2,4)")
    if prod(int(dimension) for dimension in shape) > 800:
        raise ValueError(f"{name} exceeds the scenario input bound")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    result = np.array(finite(value, name), dtype=np.float64, copy=True)
    if len(shape) == 2:
        result = result[None, ...]
    if np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} entries must lie in [0,1]")
    totals = result.sum(axis=-1)
    if np.any(~np.isfinite(totals)) or np.any(np.abs(totals - 1.0) > _PROBABILITY_SUM_TOLERANCE):
        raise ValueError(f"each arm row in {name} must sum to one")
    result /= totals[..., None]
    result.flags.writeable = False
    return result


def _recursion_geometry(
    design: BOP2DCRandomizedPairedDesign,
    *,
    scenario_runs: int,
    retained_result_cells: int,
    work_limit: int = _MAX_TOTAL_WORK,
) -> tuple[tuple[tuple[FloatArray, FloatArray], ...], int]:
    """Preflight exact recursion and build reusable marginal posterior tables."""
    n = design.max_subjects
    control_prefix = np.r_[0, np.cumsum(design.arm_assignments == 0)]
    treatment_prefix = np.r_[0, np.cumsum(design.arm_assignments == 1)]
    state_sizes = [
        (int(control_prefix[k]) + 1) ** 2 * (int(treatment_prefix[k]) + 1) ** 2
        for k in range(1, n + 1)
    ]
    max_state = max(state_sizes)
    if max_state > _MAX_STATE_CELLS:
        raise ValueError("four-margin exact state lattice exceeds one million cells")
    transition_work = 4 * sum(state_sizes)
    if transition_work * scenario_runs > _MAX_TRANSITION_WORK:
        raise ValueError("exact joint-category recursion exceeds its transition-work limit")
    look_pairs = sum(
        (int(control_prefix[int(look)]) + 1) * (int(treatment_prefix[int(look)]) + 1)
        for look in design.looks
    )
    tail_work = look_pairs * 4 * 2 * 3 * 21 * (2 * 300 - 1)
    if tail_work > _MAX_TAIL_WORK:
        raise ValueError("paired posterior-tail cache exceeds its quadrature work bound")
    classification_work = (
        scenario_runs * 16 * sum(state_sizes[int(look) - 1] for look in design.looks)
    )
    exact_work = transition_work * scenario_runs + classification_work + tail_work
    if exact_work > work_limit:
        raise ValueError("exact paired recursion exceeds max_work")

    tail_cells = sum(
        (int(control_prefix[int(look)]) + 1) * (int(treatment_prefix[int(look)]) + 1) * 8
        for look in design.looks
    )
    max_pair_states = max(
        (int(control_prefix[int(look)]) + 1) * (int(treatment_prefix[int(look)]) + 1)
        for look in design.looks
    )
    chunk_cells = _STATE_CHUNK * (16 * 4 + 12)
    cache_scratch_cells = 2 * max_pair_states + 12 * min(200, max_pair_states)
    # Include old/new DP arrays and the multiply-by-category scratch lattice.
    workspace_cells = 3 * max_state + tail_cells + chunk_cells + cache_scratch_cells
    if (
        tail_cells > _MAX_TAIL_CELLS
        or workspace_cells + retained_result_cells > _MAX_RETAINED_CELLS
    ):
        raise ValueError("exact paired state/cache/output workspace exceeds two million cells")

    tail_cache = tuple(
        _cached_tails(
            design,
            int(control_prefix[int(look)]),
            int(treatment_prefix[int(look)]),
        )
        for look in design.looks
    )
    return tail_cache, exact_work


def _dp_scenario(
    design: BOP2DCRandomizedPairedDesign,
    truth: FloatArray,
    tail_cache: tuple[tuple[FloatArray, FloatArray], ...],
    *,
    state_chunk: int,
) -> tuple[FloatArray, FloatArray, float]:
    """Four-dimensional recursion, chunking positive states through decisions."""
    n = design.max_subjects
    assignments = design.arm_assignments
    look_index = {int(value): index for index, value in enumerate(design.looks)}
    stop_probability = np.zeros((len(design.looks), len(_DECISIONS)), dtype=np.float64)
    sample_size_probability = np.zeros(len(design.looks), dtype=np.float64)
    control_prefix = np.r_[0, np.cumsum(assignments == 0)]
    treatment_prefix = np.r_[0, np.cumsum(assignments == 1)]
    dp = np.ones((1, 1, 1, 1), dtype=np.float64)
    increments = ((1, 1), (1, 0), (0, 1), (0, 0))

    for patient_index, raw_arm in enumerate(assignments):
        arm = int(raw_arm)
        nc = int(control_prefix[patient_index + 1])
        nt = int(treatment_prefix[patient_index + 1])
        arriving = np.zeros((nc + 1, nc + 1, nt + 1, nt + 1), dtype=np.float64)
        if arm == 0:
            for probability, (first, second) in zip(truth[0], increments, strict=True):
                arriving[
                    first : first + dp.shape[0],
                    second : second + dp.shape[1],
                    :,
                    :,
                ] += dp * probability
        else:
            for probability, (first, second) in zip(truth[1], increments, strict=True):
                arriving[:, :, first : first + dp.shape[2], second : second + dp.shape[3]] += (
                    dp * probability
                )
        dp = arriving
        current_n = patient_index + 1
        index = look_index.get(current_n)
        if index is None or index == len(design.looks) - 1:
            continue
        tails, errors = tail_cache[index]
        flat_dp = dp.ravel()
        for start in range(0, flat_dp.size, state_chunk):
            local = np.flatnonzero(flat_dp[start : start + state_chunk])
            if local.size == 0:
                continue
            flat = local + start
            c1, c2, t1, t2 = np.unravel_index(flat, dp.shape)
            masses = flat_dp[flat]
            p_rows = np.stack((tails[c1, t1, 0, :], tails[c2, t2, 1, :]), axis=1)
            e_rows = np.stack((errors[c1, t1, 0, :], errors[c2, t2, 1, :]), axis=1)
            labels = design._paired_decisions_from_tails(
                p_rows,
                e_rows,
                total_n=current_n,
            )
            for action_index, action in enumerate(_DECISIONS):
                selected = labels == action
                if np.any(selected):
                    stop_probability[index, action_index] += float(np.sum(masses[selected]))
                    # Remove absorbing paths before the next patient.
                    dp[tuple(axis[selected] for axis in (c1, c2, t1, t2))] = 0.0
        sample_size_probability[index] = float(np.sum(stop_probability[index]))
        del flat_dp

    final_index = len(design.looks) - 1
    final_n = int(design.looks[final_index])
    if final_n != n:
        raise ArithmeticError("the last paired-design look must equal max_subjects")
    tails, errors = tail_cache[final_index]
    flat_dp = dp.ravel()
    for start in range(0, flat_dp.size, state_chunk):
        local = np.flatnonzero(flat_dp[start : start + state_chunk])
        if local.size == 0:
            continue
        flat = local + start
        c1, c2, t1, t2 = np.unravel_index(flat, dp.shape)
        masses = flat_dp[flat]
        p_rows = np.stack((tails[c1, t1, 0, :], tails[c2, t2, 1, :]), axis=1)
        e_rows = np.stack((errors[c1, t1, 0, :], errors[c2, t2, 1, :]), axis=1)
        labels = design._paired_decisions_from_tails(
            p_rows,
            e_rows,
            total_n=final_n,
        )
        for action_index in range(2, len(_DECISIONS)):
            action = _DECISIONS[action_index]
            stop_probability[final_index, action_index] += float(np.sum(masses[labels == action]))
    sample_size_probability[final_index] = float(np.sum(stop_probability[final_index]))
    total_probability = float(np.sum(stop_probability))
    if not np.isclose(total_probability, 1.0, rtol=0.0, atol=5e-11):
        raise ArithmeticError(f"paired exact recursion probability mass is {total_probability:g}")
    expected_n = float(np.dot(sample_size_probability, design.looks))
    return stop_probability, sample_size_probability, expected_n


def bop2_dc_randomized_paired_operating_characteristics(
    design: BOP2DCRandomizedPairedDesign,
    joint_probabilities: ArrayLike,
    *,
    max_work: int = _MAX_TOTAL_WORK,
) -> BOP2DCRandomizedPairedOperatingCharacteristics:
    """Compute exact OCs for a fixed randomized paired design and joint truths.

    ``joint_probabilities`` is either one ``(2,4)`` matrix or a bounded batch
    ``(scenarios,2,4)``. Rows are control then treatment; cells are both,
    endpoint 1 only, endpoint 2 only, neither. The returned arrays always keep
    the scenario axis. OCs are conditional on the design's fixed allocation
    tape and preserve the supplied endpoint association.
    """
    if not isinstance(design, BOP2DCRandomizedPairedDesign):
        raise TypeError("design must be a BOP2DCRandomizedPairedDesign")
    work_limit = _positive_integer(max_work, "max_work", _MAX_TOTAL_WORK)
    truths = _validate_truth_scenarios(joint_probabilities, "joint_probabilities")
    look_count = int(design.looks.size)
    scenarios = int(truths.shape[0])
    retained_cells = scenarios * (look_count * len(_DECISIONS) + look_count + 1) + 2 * truths.size
    tail_cache, exact_work = _recursion_geometry(
        design,
        scenario_runs=scenarios,
        retained_result_cells=retained_cells,
        work_limit=work_limit,
    )
    decision_probability = np.zeros((scenarios, look_count, len(_DECISIONS)), dtype=np.float64)
    sample_size_probability = np.zeros((scenarios, look_count), dtype=np.float64)
    expected_sample_size = np.zeros(scenarios, dtype=np.float64)
    for scenario, truth in enumerate(truths):
        decision, sample_size, expected_n = _dp_scenario(
            design, truth, tail_cache, state_chunk=_STATE_CHUNK
        )
        decision_probability[scenario] = decision
        sample_size_probability[scenario] = sample_size
        expected_sample_size[scenario] = expected_n
    return BOP2DCRandomizedPairedOperatingCharacteristics(
        _owned_readonly(truths),
        _owned_readonly(design.looks, dtype=np.int64),
        _DECISIONS,
        _freeze_owned(decision_probability),
        _freeze_owned(sample_size_probability),
        _freeze_owned(expected_sample_size),
        exact_work,
    )


def optimize_bop2_dc_randomized_paired(
    max_subjects: int,
    endpoint: str,
    lrv: ArrayLike,
    cmv: ArrayLike,
    futile_joint_probabilities: ArrayLike,
    effective_joint_probabilities: ArrayLike,
    *,
    arm_assignments: ArrayLike,
    lambda_lrv_grid: ArrayLike,
    lambda_cmv_grid: ArrayLike,
    gamma_lrv_grid: ArrayLike,
    gamma_cmv_grid: ArrayLike,
    control_prior: ArrayLike,
    treatment_prior: ArrayLike,
    looks: ArrayLike,
    false_go_limit: float = 0.1,
    false_no_go_limit: float = 0.1,
    false_consider_limit: float | None = None,
    objective: str = "cgr",
    graduate_at_interim: bool = False,
    comparison_tolerance: float = 1e-8,
    max_work: int = _MAX_TOTAL_WORK,
) -> BOP2DCRandomizedPairedOptimization:
    """Calibrate exact OCs over an explicit candidate grid and fixed arm tape.

    Truth matrices have rows control then treatment and columns both,
    endpoint-1 only, endpoint-2 only, neither. The recursion propagates those
    four joint categories, preserving the supplied association within each arm.
    Each cutoff/exponent grid accepts a scalar grid broadcast to both endpoints
    or an explicit (m,2) grid. Candidate ties retain product/input order.

    Correct-go includes optional interim graduation and final go. False-go is
    the same quantity under the caller-declared futile truth. False-no-go
    includes interim no-go and final no-go under the effective truth. The
    optional false-consider constraint is the larger final-consider probability
    across supplied truths. Results are exact for the supplied truths, fixed
    randomization tape, and finite candidate grid; no composite-null guarantee
    or Monte Carlo error is implied.

    The CGR objective maximizes effective-truth correct-go then minimizes
    futile-truth expected sample size. ESS-futile minimizes that expected size
    then maximizes correct-go. Remaining ties retain input/product order.
    """
    work_limit = _positive_integer(max_work, "max_work", _MAX_TOTAL_WORK)
    fg_limit = scalar(false_go_limit, "false_go_limit")
    fn_limit = scalar(false_no_go_limit, "false_no_go_limit")
    fc_limit = (
        None
        if false_consider_limit is None
        else scalar(false_consider_limit, "false_consider_limit")
    )
    if not 0 <= fg_limit <= 1 or not 0 <= fn_limit <= 1:
        raise ValueError("false_go_limit and false_no_go_limit must lie in [0,1]")
    if fc_limit is not None and not 0 <= fc_limit <= 1:
        raise ValueError("false_consider_limit must lie in [0,1]")
    if objective not in ("cgr", "ess_futile"):
        raise ValueError("objective must be 'cgr' or 'ess_futile'")

    grids = {
        "lambda_lrv": _control_grid(lambda_lrv_grid, "lambda_lrv_grid"),
        "lambda_cmv": _control_grid(lambda_cmv_grid, "lambda_cmv_grid"),
        "gamma_lrv": _control_grid(gamma_lrv_grid, "gamma_lrv_grid"),
        "gamma_cmv": _control_grid(gamma_cmv_grid, "gamma_cmv_grid"),
    }
    for name in ("lambda_lrv", "lambda_cmv"):
        if np.any((grids[name] <= 0) | (grids[name] >= 1)):
            raise ValueError(f"{name}_grid values must lie in (0,1)")
    for name in ("gamma_lrv", "gamma_cmv"):
        if np.any((grids[name] < 0) | (grids[name] > 1)):
            raise ValueError(f"{name}_grid values must lie in [0,1]")
    candidate_count = prod(int(grid.shape[0]) for grid in grids.values())
    if candidate_count > _MAX_CANDIDATES:
        raise ValueError(f"candidate grid exceeds {_MAX_CANDIDATES} combinations")

    futile = _truth(futile_joint_probabilities, "futile_joint_probabilities")
    effective = _truth(effective_joint_probabilities, "effective_joint_probabilities")
    futile_marginal = _marginals(futile)
    effective_marginal = _marginals(effective)
    base = bop2_dc_randomized_paired_design(
        max_subjects,
        endpoint,
        lrv,
        cmv,
        control_prior=control_prior,
        treatment_prior=treatment_prior,
        arm_assignments=arm_assignments,
        looks=looks,
        lambda_lrv=grids["lambda_lrv"][0],
        lambda_cmv=grids["lambda_cmv"][0],
        gamma_lrv=grids["gamma_lrv"][0],
        gamma_cmv=grids["gamma_cmv"][0],
        graduate_at_interim=graduate_at_interim,
        comparison_tolerance=comparison_tolerance,
    )
    differences = effective_marginal[1] - effective_marginal[0]
    if endpoint == "multiple_efficacy":
        effective_valid = bool(np.any(differences >= base.cmv))
    elif endpoint == "efficacy_toxicity":
        effective_valid = bool(differences[0] >= base.cmv[0] and differences[1] <= base.cmv[1])
    else:
        # Constructor owns canonical endpoint validation, but retain a local
        # guard so malformed alternate core objects cannot silently pass.
        raise ValueError("endpoint must be multiple_efficacy or efficacy_toxicity")
    if not effective_valid:
        raise ValueError("effective truth does not satisfy the endpoint clinical-go rule")

    n = base.max_subjects
    look_count = int(base.looks.size)
    if base.looks.size > 1:
        earliest_fraction = int(base.looks[0]) / n
        for lambda_name, gamma_name in (
            ("lambda_lrv", "gamma_lrv"),
            ("lambda_cmv", "gamma_cmv"),
        ):
            for column in range(2):
                with np.errstate(under="ignore"):
                    smallest_cutoff = float(np.min(grids[lambda_name][:, column])) * (
                        earliest_fraction ** float(np.max(grids[gamma_name][:, column]))
                    )
                if smallest_cutoff == 0:
                    raise ArithmeticError(
                        "an interim cutoff underflows for a candidate in the grid"
                    )

    result_cells = (
        2 * candidate_count * look_count * len(_DECISIONS)
        + 2 * candidate_count * look_count
        + 2 * candidate_count
        + 4 * candidate_count
        + 4 * candidate_count
        + candidate_count * 16
        + 48
    )
    tail_cache, exact_work = _recursion_geometry(
        base,
        scenario_runs=2 * candidate_count,
        retained_result_cells=result_cells,
        work_limit=work_limit,
    )
    candidates = np.asarray(
        [np.concatenate(rows) for rows in product(*(grids[name] for name in grids))],
        dtype=np.float64,
    )
    parameter_matrix = np.empty((candidate_count, 8), dtype=np.float64)
    candidate_decisions = np.zeros(
        (2, candidate_count, look_count, len(_DECISIONS)), dtype=np.float64
    )
    candidate_ss = np.zeros((2, candidate_count, look_count), dtype=np.float64)
    candidate_ess = np.zeros((2, candidate_count), dtype=np.float64)
    truths = (futile, effective)
    for index, values in enumerate(candidates):
        # Product grid rows are endpoint-pair blocks in the order stated above.
        parameter_matrix[index] = values
        candidate_design = bop2_dc_randomized_paired_design(
            n,
            endpoint,
            base.lrv,
            base.cmv,
            control_prior=base.control_prior,
            treatment_prior=base.treatment_prior,
            arm_assignments=base.arm_assignments,
            looks=base.looks,
            lambda_lrv=values[0:2],
            lambda_cmv=values[2:4],
            gamma_lrv=values[4:6],
            gamma_cmv=values[6:8],
            graduate_at_interim=graduate_at_interim,
            comparison_tolerance=comparison_tolerance,
        )
        for scenario_index, truth in enumerate(truths):
            decision, ss_probability, expected_n = _dp_scenario(
                candidate_design, truth, tail_cache, state_chunk=_STATE_CHUNK
            )
            candidate_decisions[scenario_index, index] = decision
            candidate_ss[scenario_index, index] = ss_probability
            candidate_ess[scenario_index, index] = expected_n

    false_go = candidate_decisions[0, :, :, 1].sum(axis=1) + candidate_decisions[0, :, -1, 2]
    false_no_go = candidate_decisions[1, :, :, 0].sum(axis=1) + candidate_decisions[1, :, -1, 4]
    correct_go = candidate_decisions[1, :, :, 1].sum(axis=1) + candidate_decisions[1, :, -1, 2]
    false_consider = np.maximum(candidate_decisions[0, :, -1, 3], candidate_decisions[1, :, -1, 3])
    feasible = np.array(
        [
            _within_limit(float(fg), fg_limit, n)
            and _within_limit(float(fn), fn_limit, n)
            and (fc_limit is None or _within_limit(float(fc), fc_limit, n))
            for fg, fn, fc in zip(false_go, false_no_go, false_consider, strict=True)
        ],
        dtype=np.bool_,
    )
    selected = _select_candidate(objective, feasible, correct_go, candidate_ess[0], n)
    selected_values = parameter_matrix[selected]
    selected_design = bop2_dc_randomized_paired_design(
        n,
        endpoint,
        base.lrv,
        base.cmv,
        control_prior=base.control_prior,
        treatment_prior=base.treatment_prior,
        arm_assignments=base.arm_assignments,
        looks=base.looks,
        lambda_lrv=selected_values[0:2],
        lambda_cmv=selected_values[2:4],
        gamma_lrv=selected_values[4:6],
        gamma_cmv=selected_values[6:8],
        graduate_at_interim=graduate_at_interim,
        comparison_tolerance=comparison_tolerance,
    )
    evidence = BOP2DCRandomizedPairedCandidateEvidence(
        _freeze_owned(parameter_matrix),
        (
            "lambda_lrv_1",
            "lambda_lrv_2",
            "lambda_cmv_1",
            "lambda_cmv_2",
            "gamma_lrv_1",
            "gamma_lrv_2",
            "gamma_cmv_1",
            "gamma_cmv_2",
        ),
        _freeze_owned(candidate_decisions),
        _freeze_owned(candidate_ss),
        _freeze_owned(candidate_ess),
        _freeze_owned(false_go),
        _freeze_owned(false_no_go),
        _freeze_owned(correct_go),
        _freeze_owned(false_consider),
        _freeze_owned(feasible),
    )
    return BOP2DCRandomizedPairedOptimization(
        selected_design,
        evidence,
        selected,
        objective,
        endpoint,
        _owned_readonly(futile),
        _owned_readonly(effective),
        _owned_readonly(futile_marginal),
        _owned_readonly(effective_marginal),
        _DECISIONS,
        fg_limit,
        fn_limit,
        fc_limit,
        exact_work,
    )
