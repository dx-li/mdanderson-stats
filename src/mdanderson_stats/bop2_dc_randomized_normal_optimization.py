"""Bounded finite-grid calibration for randomized Normal BOP2-DC designs."""

from dataclasses import dataclass, replace
from itertools import product
from math import prod

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._bop2_dc_randomized_rules import randomized_dual_decisions
from ._validation import FloatArray, finite, scalar
from .boin import _owned
from .bop2_dc_randomized_normal import (
    BOP2DCRandomizedNormalDesign,
)
from .bop2_dc_survival_trial import _replay_seed

_MAX_CANDIDATES = 10_000
_MAX_TRIALS = 10_000
_MAX_PATH_CELLS = 1_000_000
_MAX_CALIBRATION_WORK = 50_000_000
_MAX_RETAINED_CELLS = 2_000_000
_SCENARIOS = ("futile", "effective")
_TERMINAL = ("stop_no_go", "graduate", "final_go", "final_consider", "final_no_go")
_CATEGORY = {name: index for index, name in enumerate(_TERMINAL)}


class BOP2DCRandomizedNormalInfeasibleError(ValueError):
    """No candidate on the declared finite grid satisfies the empirical limits."""


@dataclass(frozen=True)
class BOP2DCRandomizedNormalCandidateEvidence:
    """Complete candidate evidence in scenario then product-grid order."""

    parameters: FloatArray
    decision_probability: FloatArray
    decision_mcse: FloatArray
    expected_sample_size: FloatArray
    enrollment_mcse: FloatArray
    false_go_rate: FloatArray
    false_no_go_rate: FloatArray
    false_consider_rate: FloatArray
    correct_go_rate: FloatArray
    feasible: NDArray[np.bool_]
    maximum_quadrature_error: FloatArray


@dataclass(frozen=True)
class BOP2DCRandomizedNormalOperatingCharacteristics:
    """Selected design's independent simulation summary, ordered by truth."""

    scenarios: tuple[str, str]
    truth_mean: FloatArray
    truth_sd: FloatArray
    decision_labels: tuple[str, ...]
    decision_probability: FloatArray
    decision_mcse: FloatArray
    expected_sample_size: FloatArray
    enrollment_mcse: FloatArray
    n_trials: int
    rng_seed: int
    maximum_quadrature_error: FloatArray


@dataclass(frozen=True)
class BOP2DCRandomizedNormalOptimization:
    """Selected design, full calibration-grid evidence, and independent holdout."""

    design: BOP2DCRandomizedNormalDesign
    candidates: BOP2DCRandomizedNormalCandidateEvidence
    selected_index: int
    calibration_oc: BOP2DCRandomizedNormalOperatingCharacteristics
    validation_oc: BOP2DCRandomizedNormalOperatingCharacteristics
    objective: str
    false_go_limit: float
    false_no_go_limit: float
    false_consider_limit: float | None
    validation_false_go_rate: float
    validation_false_no_go_rate: float
    validation_false_consider_rate: float
    validation_correct_go_rate: float
    validation_feasible: bool
    rng_seed: int


def _grid(value: ArrayLike, name: str) -> FloatArray:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if not 1 <= len(value) <= _MAX_CANDIDATES or any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must be a nonempty bounded one-dimensional grid")
        shape = (len(value),)
    else:
        raise ValueError(f"{name} must be a one-dimensional grid")
    if len(shape) != 1 or not 1 <= int(shape[0]) <= _MAX_CANDIDATES:
        raise ValueError(f"{name} must be a nonempty bounded one-dimensional grid")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    result = np.asarray(finite(value, name), dtype=np.float64)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return np.array(result, copy=True)


def _truth(value: ArrayLike, name: str) -> FloatArray:
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) != 2 or any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must be (control_mean, treatment_mean)")
        shape = (2,)
    else:
        raise ValueError(f"{name} must be (control_mean, treatment_mean)")
    if shape != (2,) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a pair of finite real means")
    result = np.asarray(finite(value, name), dtype=np.float64).copy()
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    result.flags.writeable = False
    return result


def _truth_sds(value: ArrayLike, name: str) -> FloatArray:
    result = _truth(value, name)
    if np.any(result <= 0):
        raise ValueError(f"{name} standard deviations must be positive")
    return result


def _stage_paths(
    standardized: FloatArray,
    truth: FloatArray,
    truth_sd: FloatArray,
    design: BOP2DCRandomizedNormalDesign,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Compute paired-tail paths with common offsets removed before fitting."""
    trials, n, _ = standardized.shape
    assignments = design.arm_assignments
    observations = np.empty((trials, n), dtype=np.float64)
    c_mask, t_mask = assignments == 0, assignments == 1
    with np.errstate(over="ignore", invalid="ignore"):
        observations[:, c_mask] = standardized[:, c_mask, 0] * truth_sd[0]
        observations[:, t_mask] = (truth[1] - truth[0]) + standardized[:, t_mask, 1] * truth_sd[1]
    if np.any(~np.isfinite(observations)):
        raise ArithmeticError("centered randomized Normal outcomes are not representable")
    control_prior = (design.control_prior[0] - truth[0], *design.control_prior[1:])
    treatment_prior = (design.treatment_prior[0] - truth[0], *design.treatment_prior[1:])
    if not np.isfinite(control_prior[0]) or not np.isfinite(treatment_prior[0]):
        raise ArithmeticError("centered Normal prior location is not representable")
    centered_design = replace(design, control_prior=control_prior, treatment_prior=treatment_prior)
    lrv = np.empty((design.looks.size, trials), dtype=np.float64)
    cmv = np.empty_like(lrv)
    error_lrv = np.empty_like(lrv)
    error_cmv = np.empty_like(lrv)
    for trial in range(trials):
        path = observations[trial]
        for look_index, raw_n in enumerate(design.looks):
            n_used = int(raw_n)
            allocated = assignments[:n_used]
            posterior = centered_design._posterior_tails(
                path[:n_used][allocated == 0], path[:n_used][allocated == 1]
            )
            lrv[look_index, trial] = posterior.posterior_lrv
            cmv[look_index, trial] = posterior.posterior_cmv
            error_lrv[look_index, trial] = posterior.absolute_error_lrv
            error_cmv[look_index, trial] = posterior.absolute_error_cmv
    if any(np.any(~np.isfinite(a)) for a in (lrv, cmv, error_lrv, error_cmv)):
        raise ArithmeticError("randomized Normal calibration tails are not finite")
    return lrv, cmv, error_lrv, error_cmv


def _candidate_oc(
    design: BOP2DCRandomizedNormalDesign,
    tails_lrv: FloatArray,
    tails_cmv: FloatArray,
    error_lrv: FloatArray,
    error_cmv: FloatArray,
    parameters: FloatArray,
) -> tuple[NDArray[np.int64], float, float, float]:
    n_trials = tails_lrv.shape[1]
    labels = np.full(n_trials, "continue", dtype="U16")
    stopped_at = np.full(n_trials, design.max_subjects, dtype=np.int64)
    for look_index, raw_n in enumerate(design.looks):
        active = labels == "continue"
        if not np.any(active):
            break
        total = np.full(int(np.count_nonzero(active)), int(raw_n), dtype=np.int64)
        decisions = randomized_dual_decisions(
            total,
            tails_lrv[look_index, active],
            tails_cmv[look_index, active],
            error_lrv[look_index, active],
            error_cmv[look_index, active],
            max_subjects=design.max_subjects,
            looks=design.looks,
            lambda_lrv=float(parameters[0]),
            lambda_cmv=float(parameters[1]),
            gamma_lrv=float(parameters[2]),
            gamma_cmv=float(parameters[3]),
            graduate_at_interim=design.graduate_at_interim,
        )
        terminal = decisions != "continue"
        active_indices = np.flatnonzero(active)
        labels[active_indices[terminal]] = decisions[terminal]
        stopped_at[active_indices[terminal]] = int(raw_n)
    counts = np.array([np.count_nonzero(labels == label) for label in _TERMINAL], dtype=np.int64)
    mean_n = float(np.mean(stopped_at))
    mcse_n = float(np.std(stopped_at, ddof=1) / np.sqrt(n_trials)) if n_trials > 1 else np.nan
    return counts, mean_n, mcse_n, float(max(np.max(error_lrv), np.max(error_cmv)))


def _limits(value: float, name: str) -> float:
    result = scalar(value, name)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must lie in [0,1]")
    return result


def optimize_bop2_dc_randomized_normal(
    design: BOP2DCRandomizedNormalDesign,
    futile_truth: ArrayLike,
    effective_truth: ArrayLike,
    *,
    futile_truth_sd: ArrayLike,
    effective_truth_sd: ArrayLike,
    lambda_lrv_grid: ArrayLike,
    lambda_cmv_grid: ArrayLike,
    gamma_lrv_grid: ArrayLike,
    gamma_cmv_grid: ArrayLike,
    false_go_limit: float = 0.1,
    false_no_go_limit: float = 0.1,
    false_consider_limit: float | None = None,
    objective: str = "cgr",
    n_trials: int = 100,
    n_validation: int = 100,
    rng: np.random.Generator | int | None = None,
) -> BOP2DCRandomizedNormalOptimization:
    """Calibrate all four cutoff grids using common paths and an independent holdout.

    Truths are explicit ``(control_mean, treatment_mean)`` pairs with a separate
    ``(control_sd, treatment_sd)`` for each scenario. Treatment effects must be
    ordered and the effective effect must meet ``design.theta_cmv``. The fixed
    allocation tape and monitoring schedule come from ``design``. False-go
    includes interim graduation and final go; false-no-go includes interim and
    final no-go. Optional false-consider is the maximum final-consider rate over
    both scenarios. Candidate ties retain product/input order. The 100+100 trial
    defaults are Python resource choices, not paper settings.
    """
    truths = (_truth(futile_truth, "futile_truth"), _truth(effective_truth, "effective_truth"))
    truth_sds = (
        _truth_sds(futile_truth_sd, "futile_truth_sd"),
        _truth_sds(effective_truth_sd, "effective_truth_sd"),
    )
    differences = (float(truths[0][1] - truths[0][0]), float(truths[1][1] - truths[1][0]))
    if not differences[0] < differences[1] or differences[1] < design.theta_cmv:
        raise ValueError(
            "require futile effect < effective effect and effective effect >= theta_cmv"
        )
    fg_limit, fn_limit = (
        _limits(false_go_limit, "false_go_limit"),
        _limits(false_no_go_limit, "false_no_go_limit"),
    )
    fc_limit = (
        None
        if false_consider_limit is None
        else _limits(false_consider_limit, "false_consider_limit")
    )
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
    if any(np.any((g < 0) | (g > 1)) for g in grids[2:]):
        raise ValueError("gamma grids must lie in [0,1]")
    candidate_count = prod(int(g.size) for g in grids)
    if candidate_count > _MAX_CANDIDATES:
        raise ValueError(f"candidate grid exceeds {_MAX_CANDIDATES} combinations")
    n_cal = scalar(n_trials, "n_trials")
    n_val = scalar(n_validation, "n_validation")
    if (
        n_cal != int(n_cal)
        or n_val != int(n_val)
        or not 1 <= n_cal <= _MAX_TRIALS
        or not 1 <= n_val <= _MAX_TRIALS
    ):
        raise ValueError(f"n_trials and n_validation must lie in [1,{_MAX_TRIALS}]")
    n_cal, n_val = int(n_cal), int(n_val)

    parameter_matrix = np.asarray(list(product(*grids)), dtype=np.float64)
    total_paths = n_cal + n_val
    path_cells = total_paths * design.max_subjects * 3
    max_stage = max(n_cal, n_val)
    tail_cells = 4 * design.looks.size * max_stage
    # Count returned arrays, their frozen copies, and the candidate work arrays
    # that remain live while validation begins.
    result_cells = 80 * candidate_count + parameter_matrix.size
    comparison_calls = 2 * 21 * (2 * design.quadrature_limit - 1)
    comparison_work = 2 * (n_cal + n_val) * design.looks.size * comparison_calls
    decision_work = 4 * int(design.looks.size) ** 2 * 2 * (n_cal * candidate_count + n_val)
    if path_cells > _MAX_PATH_CELLS:
        raise ValueError("calibration and validation paths exceed the one-million-cell budget")
    live_path_cells = 3 * max_stage * design.max_subjects + tail_cells
    # Includes one path's Normal sufficient-statistic scratch, active-path
    # labels/indices, and the four-corner decision helper's temporary vectors.
    live_fit_cells = 16 * design.max_subjects + 64 * max_stage
    if result_cells + live_path_cells + live_fit_cells > _MAX_RETAINED_CELLS:
        raise ValueError(
            "candidate evidence and Normal path workspace exceed the two-million-cell budget"
        )
    if comparison_work + decision_work > _MAX_CALIBRATION_WORK:
        raise ValueError(
            "randomized Normal calibration exceeds its quadrature/decision work budget"
        )
    first_fraction = int(design.looks[0]) / design.max_subjects
    with np.errstate(under="ignore"):
        for lam, gam in ((grids[0], grids[2]), (grids[1], grids[3])):
            if float(np.min(lam)) * first_fraction ** float(np.max(gam)) == 0:
                raise ArithmeticError("an interim cutoff underflows for a candidate in the grid")

    master_seed = _replay_seed(rng)
    cal_seq, val_seq = np.random.SeedSequence(master_seed).spawn(2)
    cal_seed, val_seed = (
        int(seq.generate_state(1, dtype=np.uint64)[0]) for seq in (cal_seq, val_seq)
    )
    grids_rows = parameter_matrix
    decision_counts = np.zeros((2, candidate_count, len(_TERMINAL)), dtype=np.int64)
    expected_n = np.zeros((2, candidate_count), dtype=np.float64)
    enrollment_mcse = np.zeros_like(expected_n)
    max_error = np.zeros((2, candidate_count), dtype=np.float64)
    cal_rng = np.random.default_rng(cal_seed)
    standardized = cal_rng.standard_normal((n_cal, design.max_subjects, 2))
    for scenario in range(2):
        tails_lrv, tails_cmv, error_lrv, error_cmv = _stage_paths(
            standardized, truths[scenario], truth_sds[scenario], design
        )
        # The common tail arrays are reused for every candidate in this scenario.
        for ci, parameters in enumerate(grids_rows):
            counts, mean_n, mcse_n, maxerr = _candidate_oc(
                design, tails_lrv, tails_cmv, error_lrv, error_cmv, parameters
            )
            decision_counts[scenario, ci] = counts
            expected_n[scenario, ci] = mean_n
            enrollment_mcse[scenario, ci] = mcse_n
            max_error[scenario, ci] = maxerr
        del tails_lrv, tails_cmv, error_lrv, error_cmv

    decision_probs = decision_counts.astype(np.float64) / n_cal
    false_go = (decision_counts[0, :, 1] + decision_counts[0, :, 2]) / n_cal
    false_no_go = (decision_counts[1, :, 0] + decision_counts[1, :, 4]) / n_cal
    correct_go = (decision_counts[1, :, 1] + decision_counts[1, :, 2]) / n_cal
    false_consider = np.maximum(decision_counts[0, :, 3], decision_counts[1, :, 3]) / n_cal
    feasible = (false_go <= fg_limit) & (false_no_go <= fn_limit)
    if fc_limit is not None:
        feasible &= false_consider <= fc_limit
    selected: int | None = None
    selected_key: tuple[float, float] | None = None
    for ci in range(candidate_count):
        if not feasible[ci]:
            continue
        key = (
            (-correct_go[ci], expected_n[0, ci])
            if objective == "cgr"
            else (expected_n[0, ci], -correct_go[ci])
        )
        if selected_key is None or key < selected_key:
            selected, selected_key = ci, key
    if selected is None:
        raise BOP2DCRandomizedNormalInfeasibleError(
            "no finite-grid randomized Normal candidate satisfies the empirical constraints"
        )

    selected_design = replace(
        design,
        lambda_lrv=float(parameter_matrix[selected, 0]),
        lambda_cmv=float(parameter_matrix[selected, 1]),
        gamma_lrv=float(parameter_matrix[selected, 2]),
        gamma_cmv=float(parameter_matrix[selected, 3]),
    )
    candidate_result = BOP2DCRandomizedNormalCandidateEvidence(
        _owned(parameter_matrix),
        _owned(decision_probs),
        _owned(np.sqrt(decision_probs * (1 - decision_probs) / n_cal)),
        _owned(expected_n),
        _owned(enrollment_mcse),
        _owned(false_go),
        _owned(false_no_go),
        _owned(false_consider),
        _owned(correct_go),
        _owned(feasible),
        _owned(max_error),
    )
    del standardized
    val_rng = np.random.default_rng(val_seed)
    val_standardized = val_rng.standard_normal((n_val, design.max_subjects, 2))
    validation_counts = np.zeros((2, len(_TERMINAL)), dtype=np.int64)
    validation_n = np.zeros(2, dtype=np.float64)
    validation_n_mcse = np.zeros(2, dtype=np.float64)
    validation_error = np.zeros(2, dtype=np.float64)
    for scenario in range(2):
        tails_lrv, tails_cmv, error_lrv, error_cmv = _stage_paths(
            val_standardized, truths[scenario], truth_sds[scenario], design
        )
        counts, mean_n, mcse_n, maxerr = _candidate_oc(
            selected_design, tails_lrv, tails_cmv, error_lrv, error_cmv, parameter_matrix[selected]
        )
        validation_counts[scenario] = counts
        validation_n[scenario], validation_n_mcse[scenario] = mean_n, mcse_n
        validation_error[scenario] = maxerr
        del tails_lrv, tails_cmv, error_lrv, error_cmv
    validation_decision = validation_counts.astype(np.float64) / n_val
    val_fg = float((validation_counts[0, 1] + validation_counts[0, 2]) / n_val)
    val_fn = float((validation_counts[1, 0] + validation_counts[1, 4]) / n_val)
    val_fc = float(max(validation_counts[0, 3], validation_counts[1, 3]) / n_val)
    val_cgr = float((validation_counts[1, 1] + validation_counts[1, 2]) / n_val)
    val_feasible = val_fg <= fg_limit and val_fn <= fn_limit
    if fc_limit is not None:
        val_feasible &= val_fc <= fc_limit

    def oc_result(
        decisions: FloatArray,
        means: FloatArray,
        ses: FloatArray,
        count: int,
        seed: int,
        errors: FloatArray,
    ) -> BOP2DCRandomizedNormalOperatingCharacteristics:
        truth_table = np.asarray([[*truths[s], *truth_sds[s]] for s in range(2)])
        return BOP2DCRandomizedNormalOperatingCharacteristics(
            _SCENARIOS,
            _owned(truth_table[:, :2]),
            _owned(truth_table[:, 2:]),
            _TERMINAL,
            _owned(decisions),
            _owned(np.sqrt(decisions * (1 - decisions) / count)),
            _owned(means),
            _owned(ses),
            count,
            seed,
            _owned(errors),
        )

    calibration_oc = oc_result(
        decision_probs[:, selected],
        expected_n[:, selected],
        enrollment_mcse[:, selected],
        n_cal,
        cal_seed,
        max_error[:, selected],
    )
    validation_oc = oc_result(
        validation_decision,
        validation_n,
        validation_n_mcse,
        n_val,
        val_seed,
        validation_error,
    )
    return BOP2DCRandomizedNormalOptimization(
        selected_design,
        candidate_result,
        selected,
        calibration_oc,
        validation_oc,
        objective,
        fg_limit,
        fn_limit,
        fc_limit,
        val_fg,
        val_fn,
        val_fc,
        val_cgr,
        bool(val_feasible),
        master_seed,
    )
