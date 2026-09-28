"""Finite-grid exact calibration for paired-endpoint BOP2-DC designs."""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from math import prod
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite, scalar
from .bop2_dc_paired import BOP2DCPairedDesign, bop2_dc_paired_design

_MAX_CANDIDATES = 10_000
_MAX_TOTAL_WORK = 100_000_000
_DEFAULT_MAX_WORK = 50_000_000
_MAX_RETAINED_CELLS = 2_000_000
_MAX_SINGLE_RECURSION_WORK = 5_000_000
_PROBABILITY_SUM_TOLERANCE = 1e-14


class BOP2DCPairedInfeasibleError(ValueError):
    """No supplied paired-design candidate satisfies the exact OC limits."""


@dataclass(frozen=True)
class BOP2DCPairedGridOC:
    """Exact OC summaries for one truth across the candidate-grid axis."""

    stop_no_go: FloatArray
    final_go: FloatArray
    final_consider: FloatArray
    final_no_go: FloatArray
    sample_size_probability: FloatArray
    expected_sample_size: FloatArray


@dataclass(frozen=True)
class BOP2DCPairedOptimization:
    """Finite-grid selection plus exact paired-truth operating characteristics."""

    design: BOP2DCPairedDesign
    endpoint: str
    objective: str
    candidate_count: int
    selected_index: int
    grid: Mapping[str, FloatArray]
    futile_joint_probability: FloatArray
    effective_joint_probability: FloatArray
    futile_marginal_probability: FloatArray
    effective_marginal_probability: FloatArray
    futile_oc: BOP2DCPairedGridOC
    effective_oc: BOP2DCPairedGridOC
    false_go_rate: FloatArray
    false_no_go_rate: FloatArray
    correct_go_rate: FloatArray
    false_consider_rate: FloatArray
    feasible: NDArray[np.bool_]
    false_go_limit: float
    false_no_go_limit: float
    false_consider_limit: float | None
    exact_work_units: int


def _readonly(value: ArrayLike, *, dtype: np.dtype | type | None = None) -> NDArray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _positive_integer(value: int, name: str, maximum: int) -> int:
    result = scalar(value, name)
    integer = int(result)
    if result != integer or integer < 1 or integer > maximum:
        raise ValueError(f"{name} must be an integer in [1,{maximum}]")
    return integer


def _control_grid(value: ArrayLike, name: str) -> FloatArray:
    """Read scalar grids as symmetric controls or row grids as endpoint-specific."""
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if not value or len(value) > _MAX_CANDIDATES:
            raise ValueError(f"{name} must have between 1 and {_MAX_CANDIDATES} rows")
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
        shape = np.shape(value)

    if len(shape) == 1 and 0 < shape[0] <= _MAX_CANDIDATES:
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real-valued")
        scalars = finite(value, name)
        result = np.repeat(scalars[:, None], 2, axis=1)
    elif len(shape) == 2 and shape[1] == 2 and 0 < shape[0] <= _MAX_CANDIDATES:
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real-valued")
        result = np.array(finite(value, name), dtype=np.float64, copy=True)
    else:
        raise ValueError(f"{name} must be a nonempty scalar grid or an (m,2) row grid")
    result.flags.writeable = False
    return result


def _joint_probability(value: ArrayLike, name: str) -> FloatArray:
    """Validate a single four-cell truth before any operating-characteristic work."""
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) != 4 or any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must be a length-four probability vector")
        shape = (len(value),)
    else:
        shape = np.shape(value)
    if shape != (4,):
        raise ValueError(f"{name} must be a length-four probability vector")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    result = np.array(finite(value, name), dtype=np.float64, copy=True)
    if np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} values must lie in [0,1]")
    total = float(np.sum(result))
    if not np.isfinite(total) or abs(total - 1.0) > _PROBABILITY_SUM_TOLERANCE:
        raise ValueError(f"{name} must sum to one within {_PROBABILITY_SUM_TOLERANCE:g}")
    result /= total
    result.flags.writeable = False
    return result


def _marginals(joint: FloatArray) -> FloatArray:
    # Both modes use the same cell layout for the two observed binary margins:
    # (both, first only, second only, neither).
    return np.array([joint[0] + joint[1], joint[0] + joint[2]], dtype=np.float64)


def optimize_bop2_dc_paired(
    max_subjects: int,
    endpoint: str,
    lrv: ArrayLike,
    cmv: ArrayLike,
    futile_probabilities: ArrayLike,
    effective_probabilities: ArrayLike,
    *,
    lambda_lrv_grid: ArrayLike,
    lambda_cmv_grid: ArrayLike,
    gamma_lrv_grid: ArrayLike,
    gamma_cmv_grid: ArrayLike,
    prior: ArrayLike | None = None,
    looks: ArrayLike | None = None,
    min_subjects: int = 10,
    cohort_size: int = 5,
    false_go_limit: float = 0.1,
    false_no_go_limit: float = 0.1,
    false_consider_limit: float | None = None,
    objective: str = "cgr",
    max_work: int = _DEFAULT_MAX_WORK,
) -> BOP2DCPairedOptimization:
    """Calibrate paired BOP2-DC cutoffs over the supplied finite control grid.

    Each joint truth is a four-cell vector in the count order
    ``(both, first only, second only, neither)``. For
    ``multiple_efficacy``, a 1D control grid broadcasts each scalar to both
    endpoints; an ``(m, 2)`` grid gives endpoint-specific control rows. The
    same convention applies to all four control grids.

    The futile joint truth is caller-declared and is not required to equal or
    lie below the LRVs. Effective truth must satisfy the source's clinical-go
    composition: at least one endpoint is at/above CMV for multiple efficacy;
    efficacy is at/above CMV and toxicity at/below CMV for efficacy/toxicity.
    Exact recursion preserves the supplied endpoint association. Constraints
    and objectives are exact for the supplied joint truths and finite grid.

    ``cgr`` maximizes effective-truth final-go probability, then minimizes
    futile-truth expected sample size. ``ess_futile`` minimizes futile-truth
    expected sample size, then maximizes CGR. Remaining ties use product/input
    grid order. No continuous-grid optimization is implied.
    """
    false_go_bound = scalar(false_go_limit, "false_go_limit")
    false_no_go_bound = scalar(false_no_go_limit, "false_no_go_limit")
    false_consider_bound = (
        None
        if false_consider_limit is None
        else scalar(false_consider_limit, "false_consider_limit")
    )
    if not 0 <= false_go_bound <= 1 or not 0 <= false_no_go_bound <= 1:
        raise ValueError("false_go_limit and false_no_go_limit must lie in [0,1]")
    if false_consider_bound is not None and not 0 <= false_consider_bound <= 1:
        raise ValueError("false_consider_limit must lie in [0,1]")
    if objective not in ("cgr", "ess_futile"):
        raise ValueError("objective must be 'cgr' or 'ess_futile'")
    work_limit = _positive_integer(max_work, "max_work", _MAX_TOTAL_WORK)

    grids = {
        "lambda_lrv": _control_grid(lambda_lrv_grid, "lambda_lrv_grid"),
        "lambda_cmv": _control_grid(lambda_cmv_grid, "lambda_cmv_grid"),
        "gamma_lrv": _control_grid(gamma_lrv_grid, "gamma_lrv_grid"),
        "gamma_cmv": _control_grid(gamma_cmv_grid, "gamma_cmv_grid"),
    }
    if np.any((grids["lambda_lrv"] <= 0) | (grids["lambda_lrv"] >= 1)):
        raise ValueError("lambda_lrv_grid values must lie in (0,1)")
    if np.any((grids["lambda_cmv"] <= 0) | (grids["lambda_cmv"] >= 1)):
        raise ValueError("lambda_cmv_grid values must lie in (0,1)")
    if np.any((grids["gamma_lrv"] < 0) | (grids["gamma_lrv"] > 1)):
        raise ValueError("gamma_lrv_grid values must lie in [0,1]")
    if np.any((grids["gamma_cmv"] < 0) | (grids["gamma_cmv"] > 1)):
        raise ValueError("gamma_cmv_grid values must lie in [0,1]")
    candidate_count = prod(len(values) for values in grids.values())
    if candidate_count > _MAX_CANDIDATES:
        raise ValueError(f"candidate grid contains more than {_MAX_CANDIDATES} combinations")

    futile_joint = _joint_probability(futile_probabilities, "futile_probabilities")
    effective_joint = _joint_probability(effective_probabilities, "effective_probabilities")
    futile_marginal = _marginals(futile_joint)
    effective_marginal = _marginals(effective_joint)

    base = bop2_dc_paired_design(
        max_subjects,
        endpoint,
        lrv,
        cmv,
        lambda_lrv=grids["lambda_lrv"][0],
        lambda_cmv=grids["lambda_cmv"][0],
        gamma_lrv=grids["gamma_lrv"][0],
        gamma_cmv=grids["gamma_cmv"][0],
        prior=prior,
        looks=looks,
        min_subjects=min_subjects,
        cohort_size=cohort_size,
    )
    if endpoint == "multiple_efficacy":
        effective_is_clinical = bool(np.any(effective_marginal >= base.cmv))
    else:
        effective_is_clinical = bool(
            effective_marginal[0] >= base.cmv[0] and effective_marginal[1] <= base.cmv[1]
        )
    if not effective_is_clinical:
        raise ValueError("effective_probabilities do not satisfy the endpoint clinical-go rule")

    scenarios = 2
    scenario_work = scenarios * (base.max_subjects + 1) ** 3
    if scenario_work > _MAX_SINGLE_RECURSION_WORK:
        raise ValueError("paired OC scenarios exceed the exact-recursion work budget")
    exact_work = candidate_count * scenario_work
    if exact_work > work_limit:
        raise ValueError("candidate grid exceeds max_work for exact paired recursion")

    look_count = int(base.looks.size)
    grid_cells = sum(values.size for values in grids.values())
    # Raw candidate summaries coexist with returned read-only copies. Include
    # one transient exact OC object while a candidate is being evaluated.
    retained_cells = candidate_count * (34 + 8 * look_count) + grid_cells + 48 + 8 * look_count
    if retained_cells > _MAX_RETAINED_CELLS:
        raise ValueError("candidate OC summaries exceed the retained-cell budget")

    futile_stop = np.empty((candidate_count, look_count), dtype=np.float64)
    effective_stop = np.empty_like(futile_stop)
    futile_go = np.empty(candidate_count, dtype=np.float64)
    effective_go = np.empty_like(futile_go)
    futile_consider = np.empty_like(futile_go)
    effective_consider = np.empty_like(futile_go)
    futile_no_go = np.empty_like(futile_go)
    effective_no_go = np.empty_like(futile_go)
    futile_ss_dist = np.empty((candidate_count, look_count), dtype=np.float64)
    effective_ss_dist = np.empty_like(futile_ss_dist)
    futile_expected_n = np.empty_like(futile_go)
    effective_expected_n = np.empty_like(futile_go)
    parameters = {name: np.empty((candidate_count, 2)) for name in grids}
    truth_grid = np.stack((futile_joint, effective_joint))

    candidates = product(*(values for values in grids.values()))
    for index, (lambda_lrv, lambda_cmv, gamma_lrv, gamma_cmv) in enumerate(candidates):
        design = bop2_dc_paired_design(
            base.max_subjects,
            endpoint,
            base.lrv,
            base.cmv,
            lambda_lrv=lambda_lrv,
            lambda_cmv=lambda_cmv,
            gamma_lrv=gamma_lrv,
            gamma_cmv=gamma_cmv,
            prior=base.prior,
            looks=base.looks,
        )
        oc = design.operating_characteristics(truth_grid)
        futile_stop[index] = oc.stop_no_go[0]
        effective_stop[index] = oc.stop_no_go[1]
        futile_go[index], effective_go[index] = oc.final_go
        futile_consider[index], effective_consider[index] = oc.final_consider
        futile_no_go[index], effective_no_go[index] = oc.final_no_go
        futile_ss_dist[index] = oc.sample_size_probability[0]
        effective_ss_dist[index] = oc.sample_size_probability[1]
        futile_expected_n[index], effective_expected_n[index] = oc.expected_sample_size
        for name, row in zip(grids, (lambda_lrv, lambda_cmv, gamma_lrv, gamma_cmv)):
            parameters[name][index] = row

    false_go = futile_go
    false_no_go = effective_stop.sum(axis=1) + effective_no_go
    correct_go = effective_go
    false_consider = np.maximum(futile_consider, effective_consider)
    feasible = (false_go <= false_go_bound) & (false_no_go <= false_no_go_bound)
    if false_consider_bound is not None:
        feasible &= false_consider <= false_consider_bound
    selected = _select_candidate(objective, feasible, correct_go, futile_expected_n)
    selected_design = bop2_dc_paired_design(
        base.max_subjects,
        endpoint,
        base.lrv,
        base.cmv,
        lambda_lrv=parameters["lambda_lrv"][selected],
        lambda_cmv=parameters["lambda_cmv"][selected],
        gamma_lrv=parameters["gamma_lrv"][selected],
        gamma_cmv=parameters["gamma_cmv"][selected],
        prior=base.prior,
        looks=base.looks,
    )

    def grid_oc(
        stop: FloatArray,
        go: FloatArray,
        consider: FloatArray,
        no_go: FloatArray,
        ss_dist: FloatArray,
        expected_n: FloatArray,
    ) -> BOP2DCPairedGridOC:
        return BOP2DCPairedGridOC(
            _readonly(stop),
            _readonly(go),
            _readonly(consider),
            _readonly(no_go),
            _readonly(ss_dist),
            _readonly(expected_n),
        )

    return BOP2DCPairedOptimization(
        selected_design,
        endpoint,
        objective,
        candidate_count,
        selected,
        MappingProxyType({key: _readonly(value) for key, value in grids.items()}),
        _readonly(futile_joint),
        _readonly(effective_joint),
        _readonly(futile_marginal),
        _readonly(effective_marginal),
        grid_oc(
            futile_stop, futile_go, futile_consider, futile_no_go, futile_ss_dist, futile_expected_n
        ),
        grid_oc(
            effective_stop,
            effective_go,
            effective_consider,
            effective_no_go,
            effective_ss_dist,
            effective_expected_n,
        ),
        _readonly(false_go),
        _readonly(false_no_go),
        _readonly(correct_go),
        _readonly(false_consider),
        _readonly(feasible, dtype=np.bool_),
        false_go_bound,
        false_no_go_bound,
        false_consider_bound,
        exact_work,
    )


def _select_candidate(
    objective: str,
    feasible: NDArray[np.bool_],
    correct_go: FloatArray,
    expected_n_futile: FloatArray,
) -> int:
    indices = np.flatnonzero(feasible)
    if not indices.size:
        raise BOP2DCPairedInfeasibleError(
            "no candidate satisfies the exact paired false-go/no-go constraints"
        )
    if objective == "cgr":
        return int(min(indices, key=lambda i: (-correct_go[i], expected_n_futile[i], int(i))))
    return int(min(indices, key=lambda i: (expected_n_futile[i], -correct_go[i], int(i))))
