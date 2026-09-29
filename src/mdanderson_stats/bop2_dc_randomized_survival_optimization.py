"""Finite-grid calibration for randomized BOP2-DC survival designs."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import product
from math import prod
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erf, erfinv

from ._validation import FloatArray, finite, scalar
from .bop2_dc_randomized_survival import (
    _MAX_MONITOR_CELLS,
    _MAX_REPLAY_QUADRATURE_EVALUATIONS,
    _QUAD_EVALUATIONS_PER_INTEGRAL,
    BOP2DCRandomizedSurvivalDesign,
    _posterior_comparison_tails,
)
from .bop2_dc_randomized_survival_simulation import (
    _DECISIONS,
    BOP2DCRandomizedSurvivalSimulation,
    simulate_bop2_dc_randomized_survival,
)
from .bop2_dc_survival_trial import _freeze, _replay_seed, _trial_count
from .bop2_survival_trial import _survival_paths

_MAX_CANDIDATES = 10_000
_MAX_TOTAL_WORK = 100_000_000
_MAX_DEFAULT_WORK = 50_000_000
_MAX_PATH_CELLS = 1_000_000
_MAX_LOOK_WORK = 30_000_000
_MAX_PROBABILITY_CELLS = 2_000_000
_MAX_PATH_CHUNK_CELLS = 50_000
_MAX_CANDIDATE_BATCH_CELLS = 250_000
_MAX_RETAINED_CELLS = 2_000_000
_LOG2 = float(np.log(2.0))


class BOP2DCRandomizedSurvivalInfeasibleError(ValueError):
    """No candidate on the supplied cutoff/power grid meets all constraints."""


@dataclass(frozen=True)
class BOP2DCRandomizedSurvivalCalibrationOC:
    """Held-out outcomes for one explicit arm-specific truth scenario."""

    decision_labels: tuple[str, ...]
    decision_count: NDArray[np.int64]
    decision_probability: FloatArray
    decision_mcse: FloatArray
    mean_enrollment: float
    enrollment_mcse: float
    mean_events: FloatArray
    events_mcse: FloatArray
    mean_exposure: FloatArray
    exposure_mcse: FloatArray
    trials: int


@dataclass(frozen=True)
class BOP2DCRandomizedSurvivalOptimization:
    """Finite-grid selection evidence and independent holdout validation."""

    design: BOP2DCRandomizedSurvivalDesign
    objective: str
    candidate_count: int
    selected_index: int
    decision_labels: tuple[str, ...]
    grid: Mapping[str, FloatArray]
    futile_decision_count: NDArray[np.int64]
    futile_decision_probability: FloatArray
    futile_decision_mcse: FloatArray
    effective_decision_count: NDArray[np.int64]
    effective_decision_probability: FloatArray
    effective_decision_mcse: FloatArray
    false_go_rate: FloatArray
    false_go_mcse: FloatArray
    false_no_go_rate: FloatArray
    false_no_go_mcse: FloatArray
    correct_go_rate: FloatArray
    correct_go_mcse: FloatArray
    futile_final_consider_rate: FloatArray
    futile_final_consider_mcse: FloatArray
    effective_final_consider_rate: FloatArray
    effective_final_consider_mcse: FloatArray
    false_consider_rate: FloatArray
    expected_sample_size_futile: FloatArray
    sample_size_futile_mcse: FloatArray
    expected_sample_size_effective: FloatArray
    sample_size_effective_mcse: FloatArray
    feasible: NDArray[np.bool_]
    false_go_limit: float
    false_no_go_limit: float
    false_consider_limit: float | None
    futile_truth: tuple[float, float]
    effective_truth: tuple[float, float]
    calibration_trials: int
    validation_trials: int
    validation_futile: BOP2DCRandomizedSurvivalCalibrationOC
    validation_effective: BOP2DCRandomizedSurvivalCalibrationOC
    validation_false_go_rate: float
    validation_false_go_mcse: float
    validation_false_no_go_rate: float
    validation_false_no_go_mcse: float
    validation_correct_go_rate: float
    validation_correct_go_mcse: float
    validation_futile_final_consider_rate: float
    validation_futile_final_consider_mcse: float
    validation_effective_final_consider_rate: float
    validation_effective_final_consider_mcse: float
    validation_false_consider_rate: float
    validation_feasible: bool
    rng_seed: int
    calibration_seed: int
    validation_seed: int
    work_units: int


def optimize_bop2_dc_randomized_survival(
    design: BOP2DCRandomizedSurvivalDesign,
    futile_truth: ArrayLike,
    effective_truth: ArrayLike,
    *,
    lambda_lrv_grid: ArrayLike,
    lambda_cmv_grid: ArrayLike,
    gamma_lrv_grid: ArrayLike,
    gamma_cmv_grid: ArrayLike,
    accrual_rate: float,
    final_followup: float,
    false_go_limit: float,
    false_no_go_limit: float,
    false_consider_limit: float | None = None,
    objective: str = "cgr",
    n_trials: int = 100,
    n_validation: int = 100,
    arrival: str = "fixed",
    rng: np.random.Generator | int | None = None,
) -> BOP2DCRandomizedSurvivalOptimization:
    """Choose among candidate dual-cutoff rules using shared-path simulation.

    ``futile_truth`` and ``effective_truth`` are ``(control_median,
    treatment_median)`` pairs. Both medians must be positive; the futile signed
    treatment-control difference must be below the effective difference, and the
    effective difference must be at least the design CMV. The source does not
    require the caller-declared futile difference to be at or below LRV.

    The base design supplies margins, inverse-gamma priors, fixed arm assignment,
    looks, interim graduation behavior, and quadrature tolerance. Candidate grids
    vary only the four lambda/gamma values. Posterior probabilities and their
    estimated quadrature errors are computed once per truth, trial, and look, then
    reused across all candidates. Candidate decisions are accepted only when all
    four corners of each quadrature-error interval agree. These errors are
    estimates, not rigorous bounds.

    Calibration paths are common across candidates and the two truth scenarios.
    An independent holdout checks the selected design without reselection. The
    finite-grid empirical constraints do not guarantee true operating
    characteristics. ``objective='cgr'`` maximizes effective-truth go probability
    (interim graduation or final go), breaking ties by lower futile-truth mean
    enrollment and then product/input order. ``'ess_futile'`` minimizes futile
    mean enrollment, then maximizes correct-go probability and uses input order.
    Accrual and NumPy random streams are Python conventions; native RNG/calendar
    parity is not claimed. The default 100 trials per stage is a workload choice.
    """
    if not isinstance(design, BOP2DCRandomizedSurvivalDesign):
        raise TypeError("design must be BOP2DCRandomizedSurvivalDesign")
    futile = _truth_pair(futile_truth, "futile_truth")
    effective = _truth_pair(effective_truth, "effective_truth")
    futile_difference = futile[1] - futile[0]
    effective_difference = effective[1] - effective[0]
    if not futile_difference < effective_difference:
        raise ValueError("futile truth difference must be below effective truth difference")
    if effective_difference < design.median_cmv:
        raise ValueError("effective truth difference must be at least the design CMV")

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
    if arrival not in ("fixed", "poisson"):
        raise ValueError("arrival must be 'fixed' or 'poisson'")
    trials, validation_trials = _trial_count(n_trials), _trial_count(n_validation)
    rate, followup = scalar(accrual_rate, "accrual_rate"), scalar(final_followup, "final_followup")
    if rate <= 0 or followup < 0:
        raise ValueError("accrual_rate must be positive and final_followup nonnegative")
    interval = 1.0 / rate
    truth_medians = np.asarray((futile, effective), dtype=np.float64)
    event_means = truth_medians / _LOG2
    if (
        not np.isfinite(interval)
        or interval <= 0
        or np.any(~np.isfinite(event_means))
        or np.any(event_means <= 0)
    ):
        raise ArithmeticError("truth or accrual time scale is not representable")
    expected_last = design.max_subjects * interval
    expected_final = expected_last + followup
    if not np.isfinite(expected_last) or not np.isfinite(expected_final):
        raise ArithmeticError("planned calendar time overflows")
    if followup > 0 and expected_final <= expected_last:
        raise ArithmeticError("positive final_followup is lost at this calendar-time scale")

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
    candidate_count = prod(value.size for value in grids.values())
    if candidate_count > _MAX_CANDIDATES:
        raise ValueError(f"candidate grid contains more than {_MAX_CANDIDATES} combinations")
    _check_grid_cutoffs(design, grids)

    candidates = tuple(product(*(grids[key] for key in grids)))
    lambda_lrv, lambda_cmv, gamma_lrv, gamma_cmv = (
        np.fromiter((row[column] for row in candidates), dtype=np.float64) for column in range(4)
    )
    looks = design.looks
    look_count = int(looks.size)
    margin_count = int(design.median_lrv != 0) + int(design.median_cmv != 0)
    stage_trials = trials + validation_trials
    comparison_work = 2 * stage_trials * look_count * margin_count * _QUAD_EVALUATIONS_PER_INTEGRAL
    path_work = 2 * stage_trials * int(np.sum(looks, dtype=np.int64))
    decision_work = 4 * 2 * look_count * (candidate_count * trials + validation_trials)
    work_units = comparison_work + path_work + decision_work
    if comparison_work > _MAX_TOTAL_WORK or path_work > _MAX_LOOK_WORK:
        raise ValueError("calibration exceeds its calendar/quadrature work budget")
    if (
        len(looks) * margin_count * _QUAD_EVALUATIONS_PER_INTEGRAL
        > _MAX_REPLAY_QUADRATURE_EVALUATIONS
    ):
        raise ValueError("one trial exceeds the randomized survival replay quadrature bound")
    if decision_work > _MAX_TOTAL_WORK or work_units > _MAX_TOTAL_WORK:
        raise ValueError("calibration exceeds its candidate decision work budget")

    path_chunk_rows = _path_chunk_rows(design.max_subjects, margin_count)
    probability_cells = 2 * 4 * look_count * trials
    candidate_result_cells = candidate_count * (2 * len(_DECISIONS) * 3 + 32)
    validation_cells = validation_trials * 34 + 40
    path_chunk_cells = path_chunk_rows * design.max_subjects
    fixed_retained_cells = (
        probability_cells + candidate_result_cells + validation_cells + 8 * path_chunk_cells
    )
    remaining = _MAX_RETAINED_CELLS - fixed_retained_cells
    candidate_batch_cells = min(_MAX_CANDIDATE_BATCH_CELLS, remaining // 10)
    if candidate_batch_cells < trials:
        raise ValueError("calibration exceeds the combined live/retained cell budget")
    candidate_batch_rows = max(1, min(candidate_count, candidate_batch_cells // trials))
    retained_cells = fixed_retained_cells + 10 * candidate_batch_rows * trials
    if retained_cells > _MAX_RETAINED_CELLS:
        raise ValueError("calibration exceeds the combined live/retained cell budget")
    if probability_cells > _MAX_PROBABILITY_CELLS:
        raise ValueError("posterior comparison tables exceed their cell budget")
    path_cells = 2 * (trials + validation_trials) * design.max_subjects
    if path_cells > _MAX_PATH_CELLS:
        raise ValueError("calibration exceeds the survival path cell budget")

    master_seed = _replay_seed(rng)
    seed_sequence = np.random.SeedSequence(master_seed)
    calibration_seed, validation_seed = (
        int(child.generate_state(1, dtype=np.uint64)[0]) for child in seed_sequence.spawn(2)
    )

    calibration_tails = _posterior_tail_tables(
        design,
        (futile, effective),
        trials,
        arrival,
        rate,
        followup,
        calibration_seed,
        path_chunk_rows,
    )
    futile_counts, futile_mean_n, futile_n_mcse = _evaluate_candidates(
        design,
        calibration_tails[0],
        trials,
        lambda_lrv,
        lambda_cmv,
        gamma_lrv,
        gamma_cmv,
        candidate_batch_rows,
    )
    effective_counts, effective_mean_n, effective_n_mcse = _evaluate_candidates(
        design,
        calibration_tails[1],
        trials,
        lambda_lrv,
        lambda_cmv,
        gamma_lrv,
        gamma_cmv,
        candidate_batch_rows,
    )
    del calibration_tails

    decision_index = {label: index for index, label in enumerate(_DECISIONS)}
    fut_prob, fut_mcse = _decision_summary(futile_counts, trials)
    eff_prob, eff_mcse = _decision_summary(effective_counts, trials)
    false_go_count = (
        futile_counts[:, decision_index["graduate"]] + futile_counts[:, decision_index["final_go"]]
    )
    false_no_go_count = (
        effective_counts[:, decision_index["stop_no_go"]]
        + effective_counts[:, decision_index["final_no_go"]]
    )
    correct_go_count = (
        effective_counts[:, decision_index["graduate"]]
        + effective_counts[:, decision_index["final_go"]]
    )
    futile_consider_count = futile_counts[:, decision_index["final_consider"]]
    effective_consider_count = effective_counts[:, decision_index["final_consider"]]
    false_go, false_go_mcse = _binary_summary(false_go_count, trials)
    false_no_go, false_no_go_mcse = _binary_summary(false_no_go_count, trials)
    correct_go, correct_go_mcse = _binary_summary(correct_go_count, trials)
    futile_consider, futile_consider_mcse = _binary_summary(futile_consider_count, trials)
    effective_consider, effective_consider_mcse = _binary_summary(effective_consider_count, trials)
    false_consider = np.maximum(futile_consider, effective_consider)
    feasible = (false_go <= fg_limit) & (false_no_go <= fn_limit)
    if fc_limit is not None:
        feasible &= false_consider <= fc_limit
    selected_index = _select_candidate(objective, feasible, correct_go, futile_mean_n)
    selected_design = replace(
        design,
        lambda_lrv=float(lambda_lrv[selected_index]),
        lambda_cmv=float(lambda_cmv[selected_index]),
        gamma_lrv=float(gamma_lrv[selected_index]),
        gamma_cmv=float(gamma_cmv[selected_index]),
    )

    validation_futile_sim = simulate_bop2_dc_randomized_survival(
        selected_design,
        *futile,
        accrual_rate=rate,
        final_followup=followup,
        n_trials=validation_trials,
        arrival=arrival,
        rng=validation_seed,
    )
    validation_futile = _validation_oc(validation_futile_sim)
    del validation_futile_sim
    validation_effective_sim = simulate_bop2_dc_randomized_survival(
        selected_design,
        *effective,
        accrual_rate=rate,
        final_followup=followup,
        n_trials=validation_trials,
        arrival=arrival,
        rng=validation_seed,
    )
    validation_effective = _validation_oc(validation_effective_sim)
    del validation_effective_sim
    validation_false_go_count = _rate_count(
        validation_futile.decision_count,
        (decision_index["graduate"], decision_index["final_go"]),
    )
    validation_false_no_go_count = _rate_count(
        validation_effective.decision_count,
        (decision_index["stop_no_go"], decision_index["final_no_go"]),
    )
    validation_correct_go_count = _rate_count(
        validation_effective.decision_count,
        (decision_index["graduate"], decision_index["final_go"]),
    )
    validation_futile_consider_count = validation_futile.decision_count[
        decision_index["final_consider"]
    ]
    validation_effective_consider_count = validation_effective.decision_count[
        decision_index["final_consider"]
    ]
    validation_false_go, validation_false_go_mcse = _scalar_binary_summary(
        validation_false_go_count, validation_trials
    )
    validation_false_no_go, validation_false_no_go_mcse = _scalar_binary_summary(
        validation_false_no_go_count, validation_trials
    )
    validation_correct_go, validation_correct_go_mcse = _scalar_binary_summary(
        validation_correct_go_count, validation_trials
    )
    validation_futile_consider, validation_futile_consider_mcse = _scalar_binary_summary(
        int(validation_futile_consider_count), validation_trials
    )
    validation_effective_consider, validation_effective_consider_mcse = _scalar_binary_summary(
        int(validation_effective_consider_count), validation_trials
    )
    validation_false_consider = max(validation_futile_consider, validation_effective_consider)
    validation_feasible = validation_false_go <= fg_limit and validation_false_no_go <= fn_limit
    if fc_limit is not None:
        validation_feasible &= validation_false_consider <= fc_limit

    frozen_grid = MappingProxyType({key: _freeze(value) for key, value in grids.items()})
    return BOP2DCRandomizedSurvivalOptimization(
        selected_design,
        objective,
        candidate_count,
        selected_index,
        _DECISIONS,
        frozen_grid,
        _freeze(futile_counts),
        _freeze(fut_prob),
        _freeze(fut_mcse),
        _freeze(effective_counts),
        _freeze(eff_prob),
        _freeze(eff_mcse),
        _freeze(false_go),
        _freeze(false_go_mcse),
        _freeze(false_no_go),
        _freeze(false_no_go_mcse),
        _freeze(correct_go),
        _freeze(correct_go_mcse),
        _freeze(futile_consider),
        _freeze(futile_consider_mcse),
        _freeze(effective_consider),
        _freeze(effective_consider_mcse),
        _freeze(false_consider),
        _freeze(futile_mean_n),
        _freeze(futile_n_mcse),
        _freeze(effective_mean_n),
        _freeze(effective_n_mcse),
        _freeze(feasible),
        fg_limit,
        fn_limit,
        fc_limit,
        futile,
        effective,
        trials,
        validation_trials,
        validation_futile,
        validation_effective,
        validation_false_go,
        validation_false_go_mcse,
        validation_false_no_go,
        validation_false_no_go_mcse,
        validation_correct_go,
        validation_correct_go_mcse,
        validation_futile_consider,
        validation_futile_consider_mcse,
        validation_effective_consider,
        validation_effective_consider_mcse,
        validation_false_consider,
        bool(validation_feasible),
        master_seed,
        calibration_seed,
        validation_seed,
        work_units,
    )


def _posterior_tail_tables(
    design: BOP2DCRandomizedSurvivalDesign,
    truths: tuple[tuple[float, float], tuple[float, float]],
    trials: int,
    arrival: str,
    accrual_rate: float,
    final_followup: float,
    seed: int,
    path_chunk_rows: int,
) -> FloatArray:
    """Calculate each scenario/look's four posterior tail/error tables once."""
    looks = design.looks
    table = np.empty((2, 4, looks.size, trials), dtype=np.float64)
    generator = np.random.default_rng(seed)
    assignments = design.arm_assignments

    for global_start in range(0, trials, path_chunk_rows):
        chunk_count = min(path_chunk_rows, trials - global_start)
        for local_start, enrolled, base_times, followup in _survival_paths(
            design.max_subjects,
            1.0,
            accrual_rate,
            final_followup,
            chunk_count,
            arrival,
            generator,
        ):
            start = global_start + local_start
            stop = start + enrolled.shape[0]
            standard_times = base_times * _LOG2
            if np.any(~np.isfinite(standard_times)):
                raise ArithmeticError("standardized event-time draws are not representable")
            for scenario, truth in enumerate(truths):
                arm_means = np.asarray(truth, dtype=np.float64) / _LOG2
                times = standard_times * arm_means[assignments]
                if np.any(~np.isfinite(times)):
                    raise ArithmeticError("simulated event-time draws overflow")
                for look_index, raw_n in enumerate(looks):
                    n = int(raw_n)
                    is_final = n == design.max_subjects
                    last = enrolled[:, n - 1]
                    if is_final:
                        clock = last + followup
                        if np.any(~np.isfinite(clock)) or (followup > 0 and np.any(clock <= last)):
                            raise ArithmeticError(
                                "simulated final calendar time is not representable"
                            )
                    elapsed = last[:, None] - enrolled[:, :n]
                    if is_final:
                        elapsed = elapsed + followup
                    if np.any(elapsed < 0) or np.any(~np.isfinite(elapsed)):
                        raise ArithmeticError("simulated as-of follow-up is not representable")
                    observed = np.minimum(times[:, :n], elapsed)
                    event_observed = times[:, :n] <= elapsed
                    prefix = assignments[:n]
                    d_c = np.count_nonzero(event_observed[:, prefix == 0], axis=1)
                    d_e = np.count_nonzero(event_observed[:, prefix == 1], axis=1)
                    exposure_c = np.sum(observed[:, prefix == 0], axis=1, dtype=np.float64)
                    exposure_e = np.sum(observed[:, prefix == 1], axis=1, dtype=np.float64)
                    if np.any(~np.isfinite(exposure_c)) or np.any(~np.isfinite(exposure_e)):
                        raise ArithmeticError("simulated arm exposure overflows")
                    shape_c = design.control_prior[0] + d_c
                    scale_c = design.control_prior[1] + exposure_c
                    shape_e = design.treatment_prior[0] + d_e
                    scale_e = design.treatment_prior[1] + exposure_e
                    if (
                        np.any(~np.isfinite(shape_c))
                        or np.any(~np.isfinite(scale_c))
                        or np.any(~np.isfinite(shape_e))
                        or np.any(~np.isfinite(scale_e))
                        or np.any(shape_c <= 0)
                        or np.any(scale_c <= 0)
                        or np.any(shape_e <= 0)
                        or np.any(scale_e <= 0)
                    ):
                        raise ArithmeticError(
                            "posterior inverse-gamma parameters are not representable"
                        )
                    p_lrv, err_lrv, p_cmv, err_cmv = _posterior_comparison_tails(
                        design, shape_c, scale_c, shape_e, scale_e
                    )
                    table[scenario, :, look_index, start:stop] = np.stack(
                        (p_lrv, err_lrv, p_cmv, err_cmv)
                    )
    return table


def _evaluate_candidates(
    design: BOP2DCRandomizedSurvivalDesign,
    tails: FloatArray,
    trials: int,
    lambda_lrv: FloatArray,
    lambda_cmv: FloatArray,
    gamma_lrv: FloatArray,
    gamma_cmv: FloatArray,
    batch_rows: int,
) -> tuple[NDArray[np.int64], FloatArray, FloatArray]:
    count_candidates = lambda_lrv.size
    counts = np.zeros((count_candidates, len(_DECISIONS)), dtype=np.int64)
    mean_n = np.empty(count_candidates, dtype=np.float64)
    mcse_n = np.empty(count_candidates, dtype=np.float64)
    code_index = {label: index for index, label in enumerate(_DECISIONS)}
    n = design.max_subjects
    for start in range(0, count_candidates, batch_rows):
        stop = min(start + batch_rows, count_candidates)
        size = stop - start
        shape = (size, trials)
        codes = np.full(shape, -1, dtype=np.int8)
        sample_size = np.zeros(shape, dtype=np.int16)
        active = np.ones(shape, dtype=bool)
        ll = lambda_lrv[start:stop, None]
        lc = lambda_cmv[start:stop, None]
        gl = gamma_lrv[start:stop, None]
        gc = gamma_cmv[start:stop, None]

        for look_index, raw_n in enumerate(design.looks):
            look = int(raw_n)
            p_l, err_l, p_c, err_c = tails[:, look_index, :]
            p_l, err_l, p_c, err_c = (x[None, :] for x in (p_l, err_l, p_c, err_c))
            low_l = np.maximum(0.0, p_l - err_l)
            high_l = np.minimum(1.0, p_l + err_l)
            low_c = np.maximum(0.0, p_c - err_c)
            high_c = np.minimum(1.0, p_c + err_c)
            corners = (
                (low_l, low_c),
                (low_l, high_c),
                (high_l, low_c),
                (high_l, high_c),
            )
            reference: NDArray[np.int8] | None = None
            if look < n:
                fraction = look / n
                cutoff_l = ll * fraction**gl
                cutoff_c = lc * fraction**gc
                if np.any(cutoff_l == 0) or np.any(cutoff_c == 0):
                    raise ArithmeticError("candidate interim cutoff underflows")
                if design.graduate_at_interim:
                    graduate_l = erf(erfinv(ll) / np.sqrt(fraction))
                    graduate_c = erf(erfinv(lc) / np.sqrt(fraction))
                for p_left, p_right in corners:
                    action = np.full(shape, -1, dtype=np.int8)
                    no_go = (p_left < cutoff_l) & (p_right < cutoff_c)
                    action[no_go] = code_index["stop_no_go"]
                    if design.graduate_at_interim:
                        graduate = (p_left > graduate_l) & (p_right > graduate_c)
                        if np.any(no_go & graduate):
                            raise ArithmeticError("interim graduation and no-go rules overlap")
                        action[graduate] = code_index["graduate"]
                    if reference is None:
                        reference = action
                    elif np.any(active & (action != reference)):
                        raise ArithmeticError(
                            "quadrature uncertainty could change an interim candidate decision"
                        )
                assert reference is not None
                terminal = active & (reference >= 0)
                codes[terminal] = reference[terminal]
                sample_size[terminal] = look
                active[terminal] = False
                continue

            for p_left, p_right in corners:
                action = np.full(shape, code_index["final_consider"], dtype=np.int8)
                go = (p_left > ll) & (p_right > lc)
                no_go = (p_left < ll) & (p_right < lc)
                action[go] = code_index["final_go"]
                action[no_go] = code_index["final_no_go"]
                if reference is None:
                    reference = action
                elif np.any(active & (action != reference)):
                    raise ArithmeticError(
                        "quadrature uncertainty could change a final candidate decision"
                    )
            assert reference is not None
            codes[active] = reference[active]
            sample_size[active] = n
            active[:] = False

        if np.any(active) or np.any(codes < 0) or np.any(sample_size == 0):
            raise ArithmeticError("candidate evaluator left a trial without a terminal result")
        counts[start:stop] = np.stack(
            [np.count_nonzero(codes == index, axis=1) for index in range(len(_DECISIONS))],
            axis=1,
        )
        sample_float = sample_size.astype(np.float64)
        mean_n[start:stop] = np.mean(sample_float, axis=1)
        mcse_n[start:stop] = _row_mcse(sample_float)
    if np.any(np.sum(counts, axis=1, dtype=np.int64) != trials):
        raise ArithmeticError("candidate decisions do not conserve trial count")
    return counts, mean_n, mcse_n


def _validation_oc(
    simulation: BOP2DCRandomizedSurvivalSimulation,
) -> BOP2DCRandomizedSurvivalCalibrationOC:
    return BOP2DCRandomizedSurvivalCalibrationOC(
        simulation.decision_labels,
        _freeze(simulation.decision_count),
        _freeze(simulation.decision_probability),
        _freeze(simulation.decision_mcse),
        simulation.mean_enrollment,
        simulation.enrollment_mcse,
        _freeze(simulation.mean_events),
        _freeze(simulation.events_mcse),
        _freeze(simulation.mean_exposure),
        _freeze(simulation.exposure_mcse),
        simulation.trials,
    )


def _decision_summary(counts: NDArray[np.int64], trials: int) -> tuple[FloatArray, FloatArray]:
    probabilities = counts.astype(np.float64) / trials
    mcse = np.sqrt(probabilities * (1.0 - probabilities) / trials)
    return probabilities, mcse


def _binary_summary(counts: NDArray[np.int64], trials: int) -> tuple[FloatArray, FloatArray]:
    rates = counts.astype(np.float64) / trials
    return rates, np.sqrt(rates * (1.0 - rates) / trials)


def _scalar_binary_summary(count: int, trials: int) -> tuple[float, float]:
    rate = count / trials
    return rate, float(np.sqrt(rate * (1.0 - rate) / trials))


def _rate_count(counts: NDArray[np.int64], indices: tuple[int, ...]) -> int:
    return int(np.sum(counts[list(indices)], dtype=np.int64))


def _select_candidate(
    objective: str,
    feasible: NDArray[np.bool_],
    correct_go: FloatArray,
    expected_sample_size: FloatArray,
) -> int:
    candidates = np.flatnonzero(feasible)
    if not candidates.size:
        raise BOP2DCRandomizedSurvivalInfeasibleError(
            "no candidate on the supplied grid satisfies the empirical constraints"
        )
    if objective == "cgr":
        return int(
            min(
                candidates,
                key=lambda index: (-correct_go[index], expected_sample_size[index], index),
            )
        )
    return int(
        min(candidates, key=lambda index: (expected_sample_size[index], -correct_go[index], index))
    )


def _truth_pair(value: ArrayLike, name: str) -> tuple[float, float]:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) != 2 or any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must be a (control_median, treatment_median) pair")
        shape = (2,)
    else:
        shape = np.shape(value)
    if shape != (2,) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a real (control_median, treatment_median) pair")
    result = finite(value, name)
    if np.any(result <= 0):
        raise ValueError(f"{name} medians must be positive")
    return float(result[0]), float(result[1])


def _grid(value: ArrayLike, name: str) -> FloatArray:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) > _MAX_CANDIDATES:
            raise ValueError(f"{name} contains more than {_MAX_CANDIDATES} values")
        if any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must be one-dimensional")
        shape = (len(value),)
    else:
        shape = np.shape(value)
    if len(shape) != 1 or shape[0] == 0 or shape[0] > _MAX_CANDIDATES:
        raise ValueError(f"{name} must be a nonempty 1D grid with at most {_MAX_CANDIDATES} values")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    result = np.array(finite(value, name), dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _check_grid_cutoffs(
    design: BOP2DCRandomizedSurvivalDesign, grids: Mapping[str, FloatArray]
) -> None:
    if design.looks.size < 2:
        return
    fraction = float(design.looks[0]) / design.max_subjects
    for lambda_grid, gamma_grid in (
        (grids["lambda_lrv"], grids["gamma_lrv"]),
        (grids["lambda_cmv"], grids["gamma_cmv"]),
    ):
        if float(np.min(lambda_grid)) * fraction ** float(np.max(gamma_grid)) == 0:
            raise ArithmeticError("candidate interim cutoff underflows on the supplied grid")


def _path_chunk_rows(max_subjects: int, margins: int) -> int:
    per_row_quadrature = max(1, margins) * _QUAD_EVALUATIONS_PER_INTEGRAL
    by_quadrature = max(1, 1_700_000 // per_row_quadrature)
    return max(
        1, min(5_000, _MAX_PATH_CHUNK_CELLS // max_subjects, _MAX_MONITOR_CELLS, by_quadrature)
    )


def _row_mcse(values: FloatArray) -> FloatArray:
    if values.shape[1] < 2:
        return np.full(values.shape[0], np.nan)
    scale = np.max(np.abs(values), axis=1)
    scaled = np.divide(values, scale[:, None], out=np.zeros_like(values), where=scale[:, None] > 0)
    result = scale * np.std(scaled, axis=1, ddof=1) / np.sqrt(values.shape[1])
    if np.any(~np.isfinite(result)):
        raise ArithmeticError("sample-size Monte Carlo error is not representable")
    return result
