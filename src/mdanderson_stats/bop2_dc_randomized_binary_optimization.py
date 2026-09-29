"""Finite-grid exact calibration for randomized binary BOP2-DC designs."""

from dataclasses import dataclass, replace
from itertools import product
from math import prod

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite, scalar
from .bop2_dc_randomized_binary import (
    _MAX_COMPARISON_CELLS,
    _MAX_RECURSION_WORK,
    BOP2DCRandomizedBinaryDesign,
    BOP2DCRandomizedBinaryOperatingCharacteristics,
    bop2_dc_randomized_binary_design,
)

_MAX_CANDIDATES = 10_000
_MAX_TOTAL_WORK = 50_000_000
_MAX_RETAINED_CELLS = 2_000_000
_MAX_WORK_CONTROL = 50_000_000
_SCENARIOS = ("futile", "effective")
_DECISIONS = ("stop_no_go", "graduate", "final_go", "final_consider", "final_no_go")


class BOP2DCRandomizedBinaryInfeasibleError(ValueError):
    """No candidate on the supplied finite grid satisfies the OC limits."""


@dataclass(frozen=True)
class BOP2DCRandomizedBinaryCandidateEvidence:
    """Complete exact-OC evidence for every candidate in product order."""

    parameters: FloatArray
    decision_probability: FloatArray
    sample_size_probability: FloatArray
    expected_sample_size: FloatArray
    false_go_rate: FloatArray
    false_no_go_rate: FloatArray
    correct_go_rate: FloatArray
    false_consider_rate: FloatArray
    feasible: NDArray[np.bool_]
    maximum_comparison_error: FloatArray


@dataclass(frozen=True)
class BOP2DCRandomizedBinaryOptimization:
    """Selected finite-grid design and its complete candidate evidence."""

    design: BOP2DCRandomizedBinaryDesign
    candidates: BOP2DCRandomizedBinaryCandidateEvidence
    selected_index: int
    objective: str
    scenarios: tuple[str, str]
    futile_truth: FloatArray
    effective_truth: FloatArray
    decision_labels: tuple[str, ...]
    false_go_limit: float
    false_no_go_limit: float
    false_consider_limit: float | None
    exact_work_units: int


def _grid(value: ArrayLike, name: str) -> FloatArray:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if not value or len(value) > _MAX_CANDIDATES:
            raise ValueError(f"{name} must be a nonempty grid of at most {_MAX_CANDIDATES} values")
        if any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must be one-dimensional")
        shape = (len(value),)
    else:
        raise ValueError(f"{name} must be a one-dimensional grid")
    if len(shape) != 1 or not 1 <= int(shape[0]) <= _MAX_CANDIDATES:
        raise ValueError(f"{name} must be a nonempty grid of at most {_MAX_CANDIDATES} values")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    result = np.asarray(finite(value, name), dtype=np.float64)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(result, dtype=np.float64, copy=True)


def _truth_pair(value: ArrayLike, name: str) -> FloatArray:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) != 2 or any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must be (control_probability, treatment_probability)")
        shape = (len(value),)
    else:
        raise ValueError(f"{name} must be (control_probability, treatment_probability)")
    if shape != (2,):
        raise ValueError(f"{name} must be (control_probability, treatment_probability)")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    result = np.array(finite(value, name), dtype=np.float64, copy=True)
    if np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} probabilities must lie in [0,1]")
    result.flags.writeable = False
    return result


def _positive_work(value: int, name: str) -> int:
    numeric = scalar(value, name)
    integer = int(numeric)
    if numeric != integer or not 1 <= integer <= _MAX_WORK_CONTROL:
        raise ValueError(f"{name} must be an integer in [1,{_MAX_WORK_CONTROL}]")
    return integer


def optimize_bop2_dc_randomized_binary(
    max_subjects: int,
    theta_lrv: float,
    theta_cmv: float,
    futile_truth: ArrayLike,
    effective_truth: ArrayLike,
    *,
    control_prior: ArrayLike,
    treatment_prior: ArrayLike,
    arm_assignments: ArrayLike,
    looks: ArrayLike,
    lambda_lrv_grid: ArrayLike,
    lambda_cmv_grid: ArrayLike,
    gamma_lrv_grid: ArrayLike,
    gamma_cmv_grid: ArrayLike,
    false_go_limit: float = 0.1,
    false_no_go_limit: float = 0.1,
    false_consider_limit: float | None = None,
    objective: str = "cgr",
    graduate_at_interim: bool = False,
    comparison_tolerance: float = 1e-9,
    max_work: int = _MAX_TOTAL_WORK,
) -> BOP2DCRandomizedBinaryOptimization:
    """Calibrate randomized binary BOP2-DC cutoffs over an explicit grid.

    Truth pairs are ordered ``(control, treatment)`` and independently supplied
    for the futile and effective scenarios. Exact operating characteristics are
    conditional on the fixed allocation tape. False-go includes early
    graduation and final go; false-no-go includes interim and final no-go.
    Correct-go is graduation plus final go under the effective truth. The
    optional false-consider bound uses the larger final-consider probability
    across the two truths. No Monte Carlo errors or holdout are involved.

    ``cgr`` maximizes effective-truth correct-go then minimizes futile expected
    sample size. ``ess_futile`` minimizes futile expected sample size then
    maximizes correct-go. Remaining ties retain candidate product/input order.
    This selects only among supplied candidates and does not claim a continuous
    optimum or average over random allocation schedules.
    """
    futile = _truth_pair(futile_truth, "futile_truth")
    effective = _truth_pair(effective_truth, "effective_truth")
    futile_difference = float(futile[1] - futile[0])
    effective_difference = float(effective[1] - effective[0])
    cmv = scalar(theta_cmv, "theta_cmv")
    lrv = scalar(theta_lrv, "theta_lrv")
    if not lrv < cmv:
        raise ValueError("require theta_lrv < theta_cmv")
    if not futile_difference < effective_difference or effective_difference < cmv:
        raise ValueError(
            "require futile treatment-control difference < effective difference and "
            "effective difference >= theta_cmv"
        )
    fg_limit = scalar(false_go_limit, "false_go_limit")
    fn_limit = scalar(false_no_go_limit, "false_no_go_limit")
    fc_limit = (
        None
        if false_consider_limit is None
        else scalar(false_consider_limit, "false_consider_limit")
    )
    if not 0 <= fg_limit <= 1 or not 0 <= fn_limit <= 1:
        raise ValueError("false-go and false-no-go limits must lie in [0,1]")
    if fc_limit is not None and not 0 <= fc_limit <= 1:
        raise ValueError("false_consider_limit must lie in [0,1]")
    if objective not in ("cgr", "ess_futile"):
        raise ValueError("objective must be 'cgr' or 'ess_futile'")

    grids = (
        _grid(lambda_lrv_grid, "lambda_lrv_grid"),
        _grid(lambda_cmv_grid, "lambda_cmv_grid"),
        _grid(gamma_lrv_grid, "gamma_lrv_grid"),
        _grid(gamma_cmv_grid, "gamma_cmv_grid"),
    )
    if np.any((grids[0] <= 0) | (grids[0] >= 1)) or np.any((grids[1] <= 0) | (grids[1] >= 1)):
        raise ValueError("lambda grids must lie in (0,1)")
    if any(np.any((grid < 0) | (grid > 1)) for grid in grids[2:]):
        raise ValueError("gamma grids must lie in [0,1]")
    candidate_count = prod(int(grid.size) for grid in grids)
    if candidate_count > _MAX_CANDIDATES:
        raise ValueError(f"candidate grid exceeds {_MAX_CANDIDATES} combinations")
    work_limit = _positive_work(max_work, "max_work")

    candidates = np.asarray(list(product(*grids)), dtype=np.float64)
    base = bop2_dc_randomized_binary_design(
        max_subjects,
        lrv,
        cmv,
        control_prior=control_prior,
        treatment_prior=treatment_prior,
        arm_assignments=arm_assignments,
        looks=looks,
        lambda_lrv=float(candidates[0, 0]),
        lambda_cmv=float(candidates[0, 1]),
        gamma_lrv=float(candidates[0, 2]),
        gamma_cmv=float(candidates[0, 3]),
        graduate_at_interim=graduate_at_interim,
        comparison_tolerance=comparison_tolerance,
    )

    control_prefix = np.r_[0, np.cumsum(base.arm_assignments == 0)]
    treatment_prefix = np.r_[0, np.cumsum(base.arm_assignments == 1)]
    state_cells = [
        (int(control_prefix[n]) + 1) * (int(treatment_prefix[n]) + 1)
        for n in range(1, base.max_subjects + 1)
    ]
    transition_work = sum(state_cells)
    decision_work = sum(
        (int(control_prefix[int(n)]) + 1) * (int(treatment_prefix[int(n)]) + 1) for n in base.looks
    )
    comparison_work = 2 * decision_work
    exact_work = comparison_work + 2 * candidate_count * (transition_work + decision_work)
    if comparison_work > _MAX_COMPARISON_CELLS:
        raise ValueError("posterior count-state comparison table exceeds its work bound")
    if 2 * transition_work + comparison_work > _MAX_RECURSION_WORK:
        raise ValueError("one exact randomized-arm OC exceeds the core recursion work bound")
    if exact_work > work_limit:
        raise ValueError("finite-grid exact OC work exceeds max_work")

    look_count = int(base.looks.size)
    decision_cells = 2 * candidate_count * look_count * len(_DECISIONS)
    result_cells = (
        decision_cells
        + 2 * candidate_count * look_count
        + 2 * candidate_count
        + 4 * candidate_count
        + 4 * candidate_count
        + look_count
        + candidates.size
    )
    table_cells = 4 * decision_work
    largest_state = max(state_cells)
    peak_cells = result_cells + table_cells + 16 * largest_state + 2 * (4 * look_count + 8)
    if result_cells > _MAX_RETAINED_CELLS or peak_cells > _MAX_RETAINED_CELLS:
        raise ValueError(
            "candidate evidence and posterior workspace exceed the two-million-cell bound"
        )

    # Posterior tails depend on priors, count states, margins and looks, but not
    # on lambda/gamma. Compute them once and reuse across the whole grid.
    posterior_tables = base._posterior_tables()
    decision_probability = np.zeros((2, candidate_count, look_count, len(_DECISIONS)))
    sample_size_probability = np.zeros((2, candidate_count, look_count))
    expected_sample_size = np.zeros((2, candidate_count))
    maximum_comparison_error = np.zeros(look_count)
    for look_index, (_lrv, _cmv, error_lrv, error_cmv) in enumerate(posterior_tables):
        maximum_comparison_error[look_index] = float(np.max(np.maximum(error_lrv, error_cmv)))

    control_truth = np.asarray([futile[0], effective[0]], dtype=np.float64)
    treatment_truth = np.asarray([futile[1], effective[1]], dtype=np.float64)
    for candidate_index, parameters in enumerate(candidates):
        design = replace(
            base,
            lambda_lrv=float(parameters[0]),
            lambda_cmv=float(parameters[1]),
            gamma_lrv=float(parameters[2]),
            gamma_cmv=float(parameters[3]),
        )
        oc: BOP2DCRandomizedBinaryOperatingCharacteristics = (
            design._operating_characteristics_with_tables(
                control_truth, treatment_truth, posterior_tables
            )
        )
        for scenario_index in range(2):
            decision_probability[scenario_index, candidate_index, :, 0] = oc.stop_no_go[
                scenario_index
            ]
            decision_probability[scenario_index, candidate_index, :, 1] = oc.graduate[
                scenario_index
            ]
            decision_probability[scenario_index, candidate_index, -1, 2] = oc.final_go[
                scenario_index
            ]
            decision_probability[scenario_index, candidate_index, -1, 3] = oc.final_consider[
                scenario_index
            ]
            decision_probability[scenario_index, candidate_index, -1, 4] = oc.final_no_go[
                scenario_index
            ]
        sample_size_probability[:, candidate_index] = oc.sample_size_probability
        expected_sample_size[:, candidate_index] = oc.expected_sample_size
        del oc, design

    false_go = decision_probability[0, :, :, 1].sum(axis=1) + decision_probability[0, :, -1, 2]
    false_no_go = decision_probability[1, :, :, 0].sum(axis=1) + decision_probability[1, :, -1, 4]
    correct_go = decision_probability[1, :, :, 1].sum(axis=1) + decision_probability[1, :, -1, 2]
    false_consider = np.maximum(
        decision_probability[0, :, -1, 3], decision_probability[1, :, -1, 3]
    )
    feasible = (false_go <= fg_limit) & (false_no_go <= fn_limit)
    if fc_limit is not None:
        feasible &= false_consider <= fc_limit

    selected_index: int | None = None
    selected_key: tuple[float, float] | None = None
    for index in range(candidate_count):
        if not feasible[index]:
            continue
        key = (
            (-correct_go[index], expected_sample_size[0, index])
            if objective == "cgr"
            else (expected_sample_size[0, index], -correct_go[index])
        )
        if selected_key is None or key < selected_key:
            selected_key, selected_index = key, index
    if selected_index is None:
        raise BOP2DCRandomizedBinaryInfeasibleError(
            "no finite-grid randomized binary candidate satisfies the exact OC constraints"
        )

    selected_values = candidates[selected_index]
    selected_design = replace(
        base,
        lambda_lrv=float(selected_values[0]),
        lambda_cmv=float(selected_values[1]),
        gamma_lrv=float(selected_values[2]),
        gamma_cmv=float(selected_values[3]),
    )
    for array in (
        candidates,
        decision_probability,
        sample_size_probability,
        expected_sample_size,
        false_go,
        false_no_go,
        correct_go,
        false_consider,
        feasible,
        maximum_comparison_error,
    ):
        array.flags.writeable = False
    return BOP2DCRandomizedBinaryOptimization(
        selected_design,
        BOP2DCRandomizedBinaryCandidateEvidence(
            candidates,
            decision_probability,
            sample_size_probability,
            expected_sample_size,
            false_go,
            false_no_go,
            correct_go,
            false_consider,
            feasible,
            maximum_comparison_error,
        ),
        selected_index,
        objective,
        _SCENARIOS,
        futile,
        effective,
        _DECISIONS,
        fg_limit,
        fn_limit,
        fc_limit,
        exact_work,
    )
