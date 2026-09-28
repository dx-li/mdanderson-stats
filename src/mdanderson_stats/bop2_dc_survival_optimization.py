"""Finite-grid Monte Carlo calibration for BOP2-DC survival designs."""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from math import prod
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite, scalar
from .bop2_dc_survival import BOP2DCSurvivalDesign, bop2_dc_survival_design
from .bop2_dc_survival_trial import (
    _DECISIONS,
    _MAX_LOOK_WORK,
    _MAX_SIMULATION_CELLS,
    BOP2DCSurvivalSimulation,
    _replay_seed,
    _survival_paths,
    _trial_count,
    simulate_bop2_dc_survival,
)

_MAX_CANDIDATES = 10_000
_MAX_TOTAL_WORK = 100_000_000
_DEFAULT_MAX_WORK = 50_000_000
_MAX_PROBABILITY_CELLS = 2_000_000
_MAX_PATH_CHUNK_CELLS = 50_000
_MAX_CANDIDATE_BATCH_CELLS = 250_000
_MAX_RETAINED_CELLS = 2_000_000


class BOP2DCSurvivalInfeasibleError(ValueError):
    """No candidate on the supplied finite grid satisfies empirical constraints."""


@dataclass(frozen=True)
class BOP2DCSurvivalCalibrationOC:
    """Held-out decision probabilities for one truth scenario."""

    decision_labels: tuple[str, ...]
    decision_probability: FloatArray
    decision_mcse: FloatArray
    mean_sample_size: float
    sample_size_mcse: float
    trials: int


@dataclass(frozen=True)
class BOP2DCSurvivalOptimization:
    """Finite-grid selection evidence and independent held-out validation."""

    design: BOP2DCSurvivalDesign
    objective: str
    candidate_count: int
    selected_index: int
    grid: Mapping[str, FloatArray]
    futile_decision_probability: FloatArray
    futile_decision_mcse: FloatArray
    effective_decision_probability: FloatArray
    effective_decision_mcse: FloatArray
    false_go_rate: FloatArray
    false_no_go_rate: FloatArray
    correct_go_rate: FloatArray
    false_consider_rate: FloatArray
    expected_sample_size_futile: FloatArray
    sample_size_futile_mcse: FloatArray
    expected_sample_size_effective: FloatArray
    sample_size_effective_mcse: FloatArray
    feasible: NDArray[np.bool_]
    false_go_limit: float
    false_no_go_limit: float
    false_consider_limit: float | None
    futile_truth: float
    effective_truth: float
    calibration_trials: int
    validation_trials: int
    validation_futile: BOP2DCSurvivalCalibrationOC
    validation_effective: BOP2DCSurvivalCalibrationOC
    validation_false_go_rate: float
    validation_false_no_go_rate: float
    validation_correct_go_rate: float
    validation_false_consider_rate: float
    validation_feasible: bool
    rng_seed: int
    calibration_seed: int
    validation_seed: int
    work_units: int


def _grid(value: ArrayLike, name: str, *, maximum: int = _MAX_CANDIDATES) -> FloatArray:
    """Validate grid shape/size before converting caller data to float arrays."""
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) > maximum:
            raise ValueError(f"{name} contains more than {maximum} values")
        if any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must be one-dimensional")
        shape = (len(value),)
    else:
        shape = np.shape(value)
    if len(shape) != 1 or shape[0] == 0 or shape[0] > maximum:
        raise ValueError(f"{name} must be a nonempty 1D grid with at most {maximum} values")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    result = finite(value, name)
    result = np.array(result, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def optimize_bop2_dc_survival(
    max_subjects: int,
    lrv: float,
    cmv: float,
    theta_futile: float,
    theta_effective: float,
    *,
    lambda_lrv_grid: ArrayLike,
    lambda_cmv_grid: ArrayLike,
    gamma_lrv_grid: ArrayLike,
    gamma_cmv_grid: ArrayLike,
    prior_shape: float,
    prior_scale: float,
    looks: ArrayLike,
    accrual_rate: float,
    final_followup: float,
    false_go_limit: float,
    false_no_go_limit: float,
    false_consider_limit: float | None = None,
    objective: str = "cgr",
    n_trials: int = 10_000,
    n_validation: int = 10_000,
    arrival: str = "fixed",
    rng: np.random.Generator | int | None = None,
    max_work: int = _DEFAULT_MAX_WORK,
) -> BOP2DCSurvivalOptimization:
    """Select a finite-grid BOP2-DC survival design by common-path Monte Carlo.

    All four cutoff/power grids, inverse-gamma prior parameters, looks, truth
    medians, and calendar settings are caller supplied. Require the effective
    median to be at least CMV and the explicit futile median to be smaller.
    Calibration constraints apply to estimated rates only, not guaranteed true
    operating characteristics.
    The selected candidate is checked on independent held-out paths and is never
    reselected. Both truth scenarios use common random paths within each stage.

    ``objective='cgr'`` maximizes effective-truth final-go probability, breaking
    ties by lower futile-truth expected sample size and then product/input order.
    ``objective='ess_futile'`` minimizes futile-truth expected sample size, then
    maximizes CGR and uses product/input order. No continuous-grid or native
    optimizer/RNG parity is claimed.
    """
    futile = scalar(theta_futile, "theta_futile")
    effective = scalar(theta_effective, "theta_effective")
    lrv_value, cmv_value = scalar(lrv, "lrv"), scalar(cmv, "cmv")
    if not 0 < lrv_value < cmv_value:
        raise ValueError("require 0 < lrv < cmv")
    if not 0 < futile < effective or effective < cmv_value:
        raise ValueError("require 0 < theta_futile < theta_effective and theta_effective >= cmv")
    fg_limit, fn_limit = (
        scalar(false_go_limit, "false_go_limit"),
        scalar(false_no_go_limit, "false_no_go_limit"),
    )
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
    if arrival not in ("fixed", "poisson"):
        raise ValueError("arrival must be 'fixed' or 'poisson'")
    work_limit = _positive_integer(max_work, "max_work", _MAX_TOTAL_WORK)
    calibration_trials, validation_trials = _trial_count(n_trials), _trial_count(n_validation)

    grids = {
        "lambda_lrv": _grid(lambda_lrv_grid, "lambda_lrv_grid"),
        "lambda_cmv": _grid(lambda_cmv_grid, "lambda_cmv_grid"),
        "gamma_lrv": _grid(gamma_lrv_grid, "gamma_lrv_grid"),
        "gamma_cmv": _grid(gamma_cmv_grid, "gamma_cmv_grid"),
    }
    if np.any((grids["lambda_lrv"] <= 0) | (grids["lambda_lrv"] >= 1)):
        raise ValueError("lambda_lrv_grid values must lie in (0,1)")
    if np.any((grids["lambda_cmv"] <= 0) | (grids["lambda_cmv"] >= 1)):
        raise ValueError("lambda_cmv_grid values must lie in (0,1)")
    if np.any((grids["gamma_lrv"] < 0) | (grids["gamma_lrv"] > 1)):
        raise ValueError("gamma_lrv_grid values must lie in [0,1]")
    if np.any((grids["gamma_cmv"] < 0) | (grids["gamma_cmv"] > 1)):
        raise ValueError("gamma_cmv_grid values must lie in [0,1]")
    candidate_count = prod(values.size for values in grids.values())
    if candidate_count > _MAX_CANDIDATES:
        raise ValueError(f"candidate grid contains more than {_MAX_CANDIDATES} combinations")

    # Constructing the baseline validates the explicit prior and interim calendar.
    design = bop2_dc_survival_design(
        max_subjects,
        lrv_value,
        cmv_value,
        lambda_lrv=float(grids["lambda_lrv"][0]),
        lambda_cmv=float(grids["lambda_cmv"][0]),
        gamma_lrv=float(grids["gamma_lrv"][0]),
        gamma_cmv=float(grids["gamma_cmv"][0]),
        prior_shape=prior_shape,
        prior_scale=prior_scale,
        looks=looks,
    )
    rate = scalar(accrual_rate, "accrual_rate")
    followup = scalar(final_followup, "final_followup")
    if rate <= 0 or followup < 0:
        raise ValueError("accrual_rate must be positive and final_followup nonnegative")
    if arrival not in ("fixed", "poisson"):
        raise ValueError("arrival must be 'fixed' or 'poisson'")
    n = design.max_subjects
    look_count = int(design.looks.size)
    if not np.isfinite(futile / np.log(2.0)) or not np.isfinite(effective / np.log(2.0)):
        raise ArithmeticError("truth event-time scale is not representable")
    if not np.isfinite(1.0 / rate) or 1.0 / rate == 0:
        raise ArithmeticError("accrual interarrival scale is not representable")
    if min(futile / np.log(2.0), effective / np.log(2.0)) == 0:
        raise ArithmeticError("truth event-time scale underflows")
    if look_count > 1:
        earliest_fraction = float(design.looks[0]) / n
        for cutoff_grid, gamma_grid in (
            (grids["lambda_lrv"], grids["gamma_lrv"]),
            (grids["lambda_cmv"], grids["gamma_cmv"]),
        ):
            if float(np.min(cutoff_grid)) * earliest_fraction ** float(np.max(gamma_grid)) == 0:
                raise ArithmeticError("a candidate interim cutoff underflows")

    if validation_trials * n > _MAX_SIMULATION_CELLS:
        raise ValueError("validation exceeds the survival path cell budget")
    if validation_trials * int(np.sum(design.looks, dtype=np.int64)) > _MAX_LOOK_WORK:
        raise ValueError("validation exceeds the repeated-look work budget")
    probability_cells = 2 * look_count * calibration_trials
    if probability_cells > _MAX_PROBABILITY_CELLS:
        raise ValueError("calibration posterior-probability storage exceeds its cell budget")
    path_chunk_rows = max(1, min(5000, calibration_trials, _MAX_PATH_CHUNK_CELLS // n))
    path_chunk_cells = path_chunk_rows * n
    candidate_result_cells = 34 * candidate_count + 4 * look_count
    validation_cells = 12 * validation_trials
    fixed_retained_cells = (
        probability_cells + candidate_result_cells + validation_cells + 8 * path_chunk_cells
    )
    available_batch_cells = min(
        _MAX_CANDIDATE_BATCH_CELLS,
        (_MAX_RETAINED_CELLS - fixed_retained_cells) // 4,
    )
    if available_batch_cells < calibration_trials:
        raise ValueError("calibration exceeds the combined live/retained cell budget")
    batch_rows = max(1, min(candidate_count, available_batch_cells // calibration_trials))
    batch_cells = batch_rows * calibration_trials
    retained_cells = fixed_retained_cells + 4 * batch_cells
    if retained_cells > _MAX_RETAINED_CELLS:
        raise ValueError("calibration exceeds the combined live/retained cell budget")

    work_units = (
        2 * candidate_count * calibration_trials * look_count
        + 2 * validation_trials * look_count
        + 2 * int(np.sum(design.looks, dtype=np.int64)) * (calibration_trials + validation_trials)
    )
    if work_units > work_limit:
        raise ValueError("calibration exceeds max_work")

    candidates = tuple(product(*(grids[key] for key in grids)))
    ll, lc, gl, gc = (np.fromiter((row[i] for row in candidates), dtype=float) for i in range(4))
    master_seed = _replay_seed(rng)
    seed_sequence = np.random.SeedSequence(master_seed)
    calibration_seed, validation_seed = (
        int(child.generate_state(1, dtype=np.uint64)[0]) for child in seed_sequence.spawn(2)
    )
    futile_prob, futile_mcse, futile_en, futile_en_mcse = _evaluate_scenario(
        design,
        futile,
        calibration_trials,
        arrival,
        rate,
        followup,
        calibration_seed,
        ll,
        lc,
        gl,
        gc,
        batch_rows,
        path_chunk_rows,
    )
    effective_prob, effective_mcse, effective_en, effective_en_mcse = _evaluate_scenario(
        design,
        effective,
        calibration_trials,
        arrival,
        rate,
        followup,
        calibration_seed,
        ll,
        lc,
        gl,
        gc,
        batch_rows,
        path_chunk_rows,
    )
    decision_index = {name: index for index, name in enumerate(_DECISIONS)}
    fgr = futile_prob[:, decision_index["final_go"]]
    fnr = (
        effective_prob[:, decision_index["stop_no_go"]]
        + effective_prob[:, decision_index["final_no_go"]]
    )
    cgr = effective_prob[:, decision_index["final_go"]]
    fcr = np.maximum(
        futile_prob[:, decision_index["final_consider"]],
        effective_prob[:, decision_index["final_consider"]],
    )
    feasible = (fgr <= fg_limit) & (fnr <= fn_limit)
    if fc_limit is not None:
        feasible &= fcr <= fc_limit
    selected = _select_candidate(objective, feasible, cgr, futile_en)
    selected_design = bop2_dc_survival_design(
        n,
        lrv_value,
        cmv_value,
        lambda_lrv=float(ll[selected]),
        lambda_cmv=float(lc[selected]),
        gamma_lrv=float(gl[selected]),
        gamma_cmv=float(gc[selected]),
        prior_shape=prior_shape,
        prior_scale=prior_scale,
        looks=design.looks,
    )

    validation_futile = _validation_oc(
        selected_design,
        futile,
        rate,
        followup,
        validation_trials,
        arrival,
        validation_seed,
    )
    validation_effective = _validation_oc(
        selected_design,
        effective,
        rate,
        followup,
        validation_trials,
        arrival,
        validation_seed,
    )
    vfgr = float(validation_futile.decision_probability[decision_index["final_go"]])
    vfnr = float(
        validation_effective.decision_probability[decision_index["stop_no_go"]]
        + validation_effective.decision_probability[decision_index["final_no_go"]]
    )
    vcgr = float(validation_effective.decision_probability[decision_index["final_go"]])
    vfcr = float(
        max(
            validation_futile.decision_probability[decision_index["final_consider"]],
            validation_effective.decision_probability[decision_index["final_consider"]],
        )
    )
    validation_feasible = vfgr <= fg_limit and vfnr <= fn_limit
    if fc_limit is not None:
        validation_feasible &= vfcr <= fc_limit

    frozen_grid = MappingProxyType({key: _freeze(value) for key, value in grids.items()})
    return BOP2DCSurvivalOptimization(
        selected_design,
        objective,
        candidate_count,
        selected,
        frozen_grid,
        _freeze(futile_prob),
        _freeze(futile_mcse),
        _freeze(effective_prob),
        _freeze(effective_mcse),
        _freeze(fgr),
        _freeze(fnr),
        _freeze(cgr),
        _freeze(fcr),
        _freeze(futile_en),
        _freeze(futile_en_mcse),
        _freeze(effective_en),
        _freeze(effective_en_mcse),
        _freeze(feasible),
        fg_limit,
        fn_limit,
        fc_limit,
        futile,
        effective,
        calibration_trials,
        validation_trials,
        validation_futile,
        validation_effective,
        vfgr,
        vfnr,
        vcgr,
        vfcr,
        bool(validation_feasible),
        master_seed,
        calibration_seed,
        validation_seed,
        work_units,
    )


def _evaluate_scenario(
    design: BOP2DCSurvivalDesign,
    true_median: float,
    trials: int,
    arrival: str,
    accrual_rate: float,
    final_followup: float,
    seed: int,
    lambda_lrv: FloatArray,
    lambda_cmv: FloatArray,
    gamma_lrv: FloatArray,
    gamma_cmv: FloatArray,
    candidate_batch_rows: int,
    path_chunk_rows: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    p_lrv, p_cmv = _posterior_probabilities(
        design,
        true_median,
        trials,
        arrival,
        accrual_rate,
        final_followup,
        seed,
        path_chunk_rows,
    )
    count_candidates = lambda_lrv.size
    decision_probability = np.empty((count_candidates, len(_DECISIONS)))
    decision_mcse = np.empty_like(decision_probability)
    expected_n = np.empty(count_candidates)
    expected_n_mcse = np.empty(count_candidates)
    candidate_batch = max(1, min(count_candidates, candidate_batch_rows))
    decision_index = {name: index for index, name in enumerate(_DECISIONS)}
    n = design.max_subjects
    looks = design.looks

    for start in range(0, count_candidates, candidate_batch):
        stop = min(start + candidate_batch, count_candidates)
        batch_size = stop - start
        codes = np.full((batch_size, trials), -1, dtype=np.int8)
        sample_sizes = np.zeros((batch_size, trials), dtype=np.int16)
        active = np.ones((batch_size, trials), dtype=bool)
        ll = lambda_lrv[start:stop, None]
        lc = lambda_cmv[start:stop, None]
        gl = gamma_lrv[start:stop, None]
        gc = gamma_cmv[start:stop, None]
        for look_index, raw_n in enumerate(looks):
            look = int(raw_n)
            p0, p1 = p_lrv[look_index][None, :], p_cmv[look_index][None, :]
            if look < n:
                cutoff_l = ll * (look / n) ** gl
                cutoff_c = lc * (look / n) ** gc
                if np.any(cutoff_l == 0) or np.any(cutoff_c == 0):
                    raise ArithmeticError("interim cutoff underflows on the candidate grid")
                stop_no_go = active & (p0 < cutoff_l) & (p1 < cutoff_c)
                codes[stop_no_go] = decision_index["stop_no_go"]
                sample_sizes[stop_no_go] = look
                active[stop_no_go] = False
                continue

            final_go = active & (p0 > ll) & (p1 > lc)
            final_no_go = active & (p0 < ll) & (p1 < lc)
            final_consider = active & ~(final_go | final_no_go)
            codes[final_go] = decision_index["final_go"]
            codes[final_consider] = decision_index["final_consider"]
            codes[final_no_go] = decision_index["final_no_go"]
            sample_sizes[active] = n
            active[:] = False

        if np.any(codes < 0) or np.any(sample_sizes == 0):
            raise ArithmeticError("candidate evaluator left a trial without a terminal decision")
        counts = np.stack(
            [np.count_nonzero(codes == index, axis=1) for index in range(len(_DECISIONS))], axis=1
        )
        probabilities = counts.astype(np.float64) / trials
        decision_probability[start:stop] = probabilities
        decision_mcse[start:stop] = np.sqrt(probabilities * (1 - probabilities) / trials)
        expected_n[start:stop] = np.mean(sample_sizes, axis=1)
        expected_n_mcse[start:stop] = _row_mcse(sample_sizes.astype(np.float64))

    return decision_probability, decision_mcse, expected_n, expected_n_mcse


def _posterior_probabilities(
    design: BOP2DCSurvivalDesign,
    true_median: float,
    trials: int,
    arrival: str,
    accrual_rate: float,
    final_followup: float,
    seed: int,
    path_chunk_rows: int,
) -> tuple[FloatArray, FloatArray]:
    p_lrv = np.empty((design.looks.size, trials), dtype=np.float64)
    p_cmv = np.empty_like(p_lrv)
    generator = np.random.default_rng(seed)
    for start, enrolled, event_times, followup in _bounded_paths(
        design.max_subjects,
        true_median,
        accrual_rate,
        final_followup,
        trials,
        arrival,
        generator,
        path_chunk_rows,
    ):
        for look_index, raw_n in enumerate(design.looks):
            n = int(raw_n)
            last = enrolled[:, n - 1]
            elapsed = last[:, None] - enrolled[:, :n]
            if n == design.max_subjects:
                elapsed = elapsed + followup
            observed = np.minimum(event_times[:, :n], elapsed)
            events = np.count_nonzero(event_times[:, :n] <= elapsed, axis=-1)
            total_time = np.sum(observed, axis=-1, dtype=np.float64)
            if np.any(~np.isfinite(total_time)):
                raise ArithmeticError("simulated total observed time overflows")
            state = design.monitor(n, events, total_time)
            batch = enrolled.shape[0]
            p_lrv[look_index, start : start + batch] = state.posterior_lrv
            p_cmv[look_index, start : start + batch] = state.posterior_cmv
    return p_lrv, p_cmv


def _bounded_paths(
    max_subjects: int,
    true_median: float,
    accrual_rate: float,
    final_followup: float,
    trials: int,
    arrival: str,
    generator: np.random.Generator,
    chunk_rows: int,
):
    for global_start in range(0, trials, chunk_rows):
        current = min(chunk_rows, trials - global_start)
        for local_start, enrolled, times, followup in _survival_paths(
            max_subjects,
            true_median,
            accrual_rate,
            final_followup,
            current,
            arrival,
            generator,
        ):
            yield global_start + local_start, enrolled, times, followup


def _validation_oc(
    design: BOP2DCSurvivalDesign,
    median: float,
    accrual_rate: float,
    final_followup: float,
    trials: int,
    arrival: str,
    seed: int,
) -> BOP2DCSurvivalCalibrationOC:
    simulation: BOP2DCSurvivalSimulation = simulate_bop2_dc_survival(
        design,
        median,
        accrual_rate=accrual_rate,
        final_followup=final_followup,
        n_trials=trials,
        arrival=arrival,
        rng=seed,
    )
    return BOP2DCSurvivalCalibrationOC(
        simulation.decision_labels,
        _freeze(simulation.decision_probability),
        _freeze(simulation.decision_mcse),
        simulation.mean_enrollment,
        simulation.enrollment_mcse,
        trials,
    )


def _select_candidate(
    objective: str, feasible: NDArray[np.bool_], cgr: FloatArray, expected_n: FloatArray
) -> int:
    indices = np.flatnonzero(feasible)
    if not indices.size:
        raise BOP2DCSurvivalInfeasibleError(
            "no candidate on the supplied grid satisfies the empirical constraints"
        )
    if objective == "cgr":
        return int(min(indices, key=lambda i: (-cgr[i], expected_n[i], i)))
    return int(min(indices, key=lambda i: (expected_n[i], -cgr[i], i)))


def _row_mcse(values: FloatArray) -> FloatArray:
    if values.shape[1] < 2:
        return np.full(values.shape[0], np.nan)
    scale = np.max(np.abs(values), axis=1)
    scaled = np.divide(values, scale[:, None], out=np.zeros_like(values), where=scale[:, None] > 0)
    result = scale * np.std(scaled, axis=1, ddof=1) / np.sqrt(values.shape[1])
    if np.any(~np.isfinite(result)):
        raise ArithmeticError("sample-size Monte Carlo error is not representable")
    return result


def _positive_integer(value: int, name: str, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not 1 <= result <= maximum:
        raise ValueError(f"{name} must lie in [1,{maximum}]")
    return result


def _freeze(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result
