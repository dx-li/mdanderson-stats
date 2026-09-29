"""Bounded finite-grid calibration for single-arm Normal BOP2-DC designs."""

from dataclasses import dataclass
from itertools import product
from math import prod

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite, scalar
from .boin import _owned
from .bop2_dc_normal import BOP2DCNormalDesign, bop2_dc_normal_design
from .bop2_dc_survival_trial import _replay_seed
from .normal_updating import NormalInverseGamma, NormalSample

_MAX_CANDIDATES = 10_000
_MAX_TRIALS = 100_000
_MAX_PATH_CELLS = 1_000_000
_MAX_CALIBRATION_WORK = 50_000_000
_MAX_RETAINED_CELLS = 2_000_000
_SCENARIOS = ("futile", "effective")
_DECISIONS = ("stop_no_go", "final_go", "final_consider", "final_no_go")


class BOP2DCNormalInfeasibleError(ValueError):
    """No candidate on the declared grid satisfies the empirical limits."""


@dataclass(frozen=True)
class BOP2DCNormalOperatingCharacteristics:
    """Decision and enrollment summaries ordered by futile/effective truth."""

    scenarios: tuple[str, str]
    true_mean: FloatArray
    decision_labels: tuple[str, ...]
    decision_probability: FloatArray
    decision_mcse: FloatArray
    expected_sample_size: FloatArray
    enrollment_mcse: FloatArray
    n_trials: int
    rng_seed: int


@dataclass(frozen=True)
class BOP2DCNormalCandidateEvaluation:
    """All candidate-grid operating characteristics in product/input order."""

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


@dataclass(frozen=True)
class BOP2DCNormalOptimization:
    """Selected finite-grid design and independent validation without reselection."""

    design: BOP2DCNormalDesign
    candidates: BOP2DCNormalCandidateEvaluation
    selected_index: int
    calibration_oc: BOP2DCNormalOperatingCharacteristics
    validation_oc: BOP2DCNormalOperatingCharacteristics
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
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) != 1 or int(shape[0]) == 0 or int(shape[0]) > _MAX_CANDIDATES:
            raise ValueError(f"{name} must be a nonempty one-dimensional grid")
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real")
    elif isinstance(value, (list, tuple)):
        if not 1 <= len(value) <= _MAX_CANDIDATES:
            raise ValueError(f"{name} must be a nonempty one-dimensional grid")
        if any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must be one-dimensional")
        if any(np.iscomplexobj(item) for item in value):
            raise ValueError(f"{name} must be real")
    else:
        raise ValueError(f"{name} must be a one-dimensional grid")
    result = finite(value, name)
    if result.ndim != 1 or not result.size or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional grid")
    return np.array(result, dtype=np.float64, copy=True)


def _tail_paths(
    standardized_paths: FloatArray,
    truth_mean: float,
    truth_sd: float,
    looks: NDArray[np.int64],
    base: BOP2DCNormalDesign,
) -> tuple[FloatArray, FloatArray]:
    """Compute NIG tails by look in bounded batches, centered per trial."""
    trials = standardized_paths.shape[0]
    lrv = np.empty((looks.size, trials), dtype=np.float64)
    cmv = np.empty_like(lrv)
    row_block = min(256, trials)
    for start in range(0, trials, row_block):
        stop = min(start + row_block, trials)
        # Work relative to the truth throughout. Constructing absolute outcomes
        # first loses fractional residuals when the truth has a large offset.
        paths = truth_sd * standardized_paths[start:stop]
        if np.any(~np.isfinite(paths)):
            raise ArithmeticError("calibration Normal outcomes are not finite")
        offset = paths[:, 0]
        prior_mean = (base.prior_mean - truth_mean) - offset
        theta_lrv = (base.theta_lrv - truth_mean) - offset
        theta_cmv = (base.theta_cmv - truth_mean) - offset
        if any(np.any(~np.isfinite(value)) for value in (prior_mean, theta_lrv, theta_cmv)):
            raise ArithmeticError("centered Normal calibration inputs are not representable")
        prior = NormalInverseGamma(
            prior_mean, base.prior_precision, base.prior_shape, base.prior_scale
        )
        for j, raw_n in enumerate(looks):
            sample = NormalSample.from_data(paths[:, : int(raw_n)] - offset[:, None])
            posterior = prior.update(sample)
            lrv[j, start:stop] = posterior.mean_sf(theta_lrv)
            cmv[j, start:stop] = posterior.mean_sf(theta_cmv)
    if np.any(~np.isfinite(lrv)) or np.any(~np.isfinite(cmv)):
        raise ArithmeticError("Normal posterior path tails are not finite")
    return lrv, cmv


def _grid_oc(
    lrv: FloatArray,
    cmv: FloatArray,
    looks: NDArray[np.int64],
    lambdas_lrv: FloatArray,
    lambdas_cmv: FloatArray,
    gammas_lrv: FloatArray,
    gammas_cmv: FloatArray,
    *,
    chunk_size: int | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Apply exact monitor comparisons for each candidate against common paths."""
    candidates = lambdas_lrv.size
    trials = lrv.shape[1]
    probabilities = np.empty((candidates, len(_DECISIONS)), dtype=np.float64)
    expected_n = np.empty(candidates, dtype=np.float64)
    enrollment_mcse = np.empty(candidates, dtype=np.float64)
    if chunk_size is None:
        chunk_size = max(1, min(candidates, 64, _MAX_RETAINED_CELLS // max(8 * trials, 1)))
    for start in range(0, candidates, chunk_size):
        stop = min(start + chunk_size, candidates)
        width = stop - start
        alive = np.ones((width, trials), dtype=bool)
        codes = np.full((width, trials), -1, dtype=np.int8)
        sample_size = np.full((width, trials), int(looks[-1]), dtype=np.int64)
        for j, raw_n in enumerate(looks[:-1]):
            fraction = int(raw_n) / int(looks[-1])
            cutoff_lrv = lambdas_lrv[start:stop] * fraction ** gammas_lrv[start:stop]
            cutoff_cmv = lambdas_cmv[start:stop] * fraction ** gammas_cmv[start:stop]
            if np.any(cutoff_lrv == 0) or np.any(cutoff_cmv == 0):
                raise ArithmeticError("interim cutoff underflows")
            stopped = alive & (lrv[j] < cutoff_lrv[:, None]) & (cmv[j] < cutoff_cmv[:, None])
            codes[stopped] = 0
            sample_size[stopped] = int(raw_n)
            alive[stopped] = False
        final_lrv, final_cmv = lrv[-1], cmv[-1]
        final_go = (
            alive
            & (final_lrv > lambdas_lrv[start:stop, None])
            & (final_cmv > lambdas_cmv[start:stop, None])
        )
        final_no_go = (
            alive
            & (final_lrv < lambdas_lrv[start:stop, None])
            & (final_cmv < lambdas_cmv[start:stop, None])
        )
        final_consider = alive & ~(final_go | final_no_go)
        codes[final_go], codes[final_consider], codes[final_no_go] = 1, 2, 3
        for category in range(len(_DECISIONS)):
            probabilities[start:stop, category] = np.mean(codes == category, axis=1)
        expected_n[start:stop] = np.mean(sample_size, axis=1)
        enrollment_mcse[start:stop] = (
            np.std(sample_size, axis=1, ddof=1) / np.sqrt(trials) if trials > 1 else np.nan
        )
    return probabilities, expected_n, enrollment_mcse


def optimize_bop2_dc_normal(
    max_subjects: int,
    theta_lrv: float,
    theta_cmv: float,
    theta_futile: float,
    theta_effective: float,
    *,
    truth_sd: float,
    prior_mean: float,
    prior_precision: float,
    prior_shape: float,
    prior_scale: float,
    lambda_lrv_grid: ArrayLike,
    lambda_cmv_grid: ArrayLike,
    gamma_lrv_grid: ArrayLike,
    gamma_cmv_grid: ArrayLike,
    false_go_limit: float = 0.1,
    false_no_go_limit: float = 0.1,
    false_consider_limit: float | None = None,
    objective: str = "cgr",
    looks: ArrayLike | None = None,
    min_subjects: int = 10,
    cohort_size: int = 5,
    n_trials: int = 5000,
    n_validation: int = 5000,
    rng: np.random.Generator | int | None = None,
) -> BOP2DCNormalOptimization:
    """Select cutoff scales on a common-path grid and validate without reselection.

    False-go is final go at ``theta_futile``; false-no-go includes interim and
    final no-go at ``theta_effective``; optional false-consider is the maximum
    final-consider rate across those truths. Both strict empirical error limits
    are enforced. ``cgr`` maximizes final go at the effective truth;
    ``ess_futile`` minimizes expected enrollment under the futile truth.
    """
    futile = scalar(theta_futile, "theta_futile")
    effective = scalar(theta_effective, "theta_effective")
    sd = scalar(truth_sd, "truth_sd")
    if not futile < effective or effective < scalar(theta_cmv, "theta_cmv") or sd <= 0:
        raise ValueError(
            "require theta_futile < theta_effective, theta_effective >= theta_cmv, and truth_sd > 0"
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

    lrv_grid = _grid(lambda_lrv_grid, "lambda_lrv_grid")
    cmv_grid = _grid(lambda_cmv_grid, "lambda_cmv_grid")
    gl_grid = _grid(gamma_lrv_grid, "gamma_lrv_grid")
    gc_grid = _grid(gamma_cmv_grid, "gamma_cmv_grid")
    if np.any((lrv_grid <= 0) | (lrv_grid >= 1)) or np.any((cmv_grid <= 0) | (cmv_grid >= 1)):
        raise ValueError("lambda grids must lie in (0,1)")
    if np.any((gl_grid < 0) | (gl_grid > 1)) or np.any((gc_grid < 0) | (gc_grid > 1)):
        raise ValueError("gamma grids must lie in [0,1]")
    candidate_count = prod((lrv_grid.size, cmv_grid.size, gl_grid.size, gc_grid.size))
    if candidate_count > _MAX_CANDIDATES:
        raise ValueError(f"candidate grid exceeds {_MAX_CANDIDATES} combinations")
    trial_value, validation_value = (
        scalar(n_trials, "n_trials"),
        scalar(n_validation, "n_validation"),
    )
    if (
        trial_value != int(trial_value)
        or validation_value != int(validation_value)
        or not 1 <= trial_value <= _MAX_TRIALS
        or not 1 <= validation_value <= _MAX_TRIALS
    ):
        raise ValueError(f"n_trials and n_validation must lie in [1,{_MAX_TRIALS}]")
    trial_count, validation_count = int(trial_value), int(validation_value)
    baseline = bop2_dc_normal_design(
        max_subjects,
        theta_lrv,
        theta_cmv,
        prior_mean=prior_mean,
        prior_precision=prior_precision,
        prior_shape=prior_shape,
        prior_scale=prior_scale,
        lambda_lrv=float(lrv_grid[0]),
        lambda_cmv=float(cmv_grid[0]),
        gamma_lrv=float(gl_grid[0]),
        gamma_cmv=float(gc_grid[0]),
        looks=looks,
        min_subjects=min_subjects,
        cohort_size=cohort_size,
    )
    total_paths = trial_count + validation_count
    path_cells = total_paths * baseline.max_subjects
    path_work = 2 * total_paths * int(np.sum(baseline.looks, dtype=np.int64))
    candidate_work = 2 * candidate_count * trial_count * baseline.looks.size
    result_cells = candidate_count * 64
    live_cells = (
        max(trial_count, validation_count) * baseline.max_subjects
        + 2 * baseline.looks.size * max(trial_count, validation_count)
        + 8 * min(256, max(trial_count, validation_count)) * baseline.max_subjects
    )
    if path_cells > _MAX_PATH_CELLS:
        raise ValueError("calibration and validation paths exceed the one-million-cell budget")
    if candidate_work + path_work > _MAX_CALIBRATION_WORK:
        raise ValueError("calibration exceeds the 50-million path/candidate work budget")
    available_chunk_cells = _MAX_RETAINED_CELLS - result_cells - live_cells
    candidate_chunk_width = min(
        candidate_count,
        64,
        available_chunk_cells // (8 * max(trial_count, validation_count)),
    )
    if candidate_chunk_width < 1:
        raise ValueError(
            "calibration evidence and path workspace exceed the two-million-cell budget"
        )

    master_seed = _replay_seed(rng)
    calibration_sequence, validation_sequence = np.random.SeedSequence(master_seed).spawn(2)
    calibration_seed, validation_seed = (
        int(sequence.generate_state(1, dtype=np.uint64)[0])
        for sequence in (calibration_sequence, validation_sequence)
    )
    calibration_rng = np.random.default_rng(calibration_seed)
    validation_rng = np.random.default_rng(validation_seed)
    candidate_rows = list(product(lrv_grid, cmv_grid, gl_grid, gc_grid))
    parameter_matrix = np.asarray(candidate_rows, dtype=np.float64)
    lambda_lrv = parameter_matrix[:, 0]
    lambda_cmv = parameter_matrix[:, 1]
    gamma_lrv = parameter_matrix[:, 2]
    gamma_cmv = parameter_matrix[:, 3]

    decision_prob = np.empty((2, candidate_count, len(_DECISIONS)), dtype=np.float64)
    expected_n = np.empty((2, candidate_count), dtype=np.float64)
    enrollment_mcse = np.empty_like(expected_n)
    truths = (futile, effective)
    standardized = calibration_rng.standard_normal((trial_count, baseline.max_subjects))
    for scenario, truth in enumerate(truths):
        tails_lrv, tails_cmv = _tail_paths(standardized, truth, sd, baseline.looks, baseline)
        p, en, en_mcse = _grid_oc(
            tails_lrv,
            tails_cmv,
            baseline.looks,
            lambda_lrv,
            lambda_cmv,
            gamma_lrv,
            gamma_cmv,
            chunk_size=candidate_chunk_width,
        )
        decision_prob[scenario], expected_n[scenario], enrollment_mcse[scenario] = p, en, en_mcse
        del tails_lrv, tails_cmv
    mcse = np.sqrt(decision_prob * (1 - decision_prob) / trial_count)
    false_go = decision_prob[0, :, 1]
    false_no_go = decision_prob[1, :, 0] + decision_prob[1, :, 3]
    false_consider = np.maximum(decision_prob[0, :, 2], decision_prob[1, :, 2])
    correct_go = decision_prob[1, :, 1]
    feasible = (false_go <= fg_limit) & (false_no_go <= fn_limit)
    if fc_limit is not None:
        feasible &= false_consider <= fc_limit
    best_index: int | None = None
    best_key: tuple[float, float] | None = None
    for index in range(candidate_count):
        if not feasible[index]:
            continue
        key = (
            (-correct_go[index], expected_n[0, index])
            if objective == "cgr"
            else (expected_n[0, index], -correct_go[index])
        )
        # Equal objectives retain product/input grid order.
        if best_key is None or key < best_key:
            best_key, best_index = key, index
    if best_index is None:
        raise BOP2DCNormalInfeasibleError(
            "no finite-grid Normal BOP2-DC candidate satisfies the empirical constraints"
        )
    selected = parameter_matrix[best_index]
    design = bop2_dc_normal_design(
        baseline.max_subjects,
        baseline.theta_lrv,
        baseline.theta_cmv,
        prior_mean=baseline.prior_mean,
        prior_precision=baseline.prior_precision,
        prior_shape=baseline.prior_shape,
        prior_scale=baseline.prior_scale,
        lambda_lrv=float(selected[0]),
        lambda_cmv=float(selected[1]),
        gamma_lrv=float(selected[2]),
        gamma_cmv=float(selected[3]),
        looks=baseline.looks,
    )
    candidate_result = BOP2DCNormalCandidateEvaluation(
        _owned(parameter_matrix),
        _owned(decision_prob),
        _owned(mcse),
        _owned(expected_n),
        _owned(enrollment_mcse),
        _owned(false_go),
        _owned(false_no_go),
        _owned(false_consider),
        _owned(correct_go),
        _owned(feasible),
    )
    # Calibration paths are no longer needed. Release them before allocating
    # independent holdout paths so the preflight matches peak live storage.
    del standardized

    def make_oc(
        probability: FloatArray,
        mean_n: FloatArray,
        mcse_n: FloatArray,
        count: int,
        seed: int,
    ) -> BOP2DCNormalOperatingCharacteristics:
        return BOP2DCNormalOperatingCharacteristics(
            _SCENARIOS,
            _owned(np.asarray(truths, dtype=np.float64)),
            _DECISIONS,
            _owned(probability),
            _owned(np.sqrt(probability * (1 - probability) / count)),
            _owned(mean_n),
            _owned(mcse_n),
            count,
            seed,
        )

    calibration_oc = make_oc(
        decision_prob[:, best_index],
        expected_n[:, best_index],
        enrollment_mcse[:, best_index],
        trial_count,
        calibration_seed,
    )
    validation_decisions = np.empty((2, len(_DECISIONS)), dtype=np.float64)
    validation_n = np.empty(2, dtype=np.float64)
    validation_mcse_n = np.empty(2, dtype=np.float64)
    standardized_validation = validation_rng.standard_normal(
        (validation_count, baseline.max_subjects)
    )
    for scenario, truth in enumerate(truths):
        tails_lrv, tails_cmv = _tail_paths(
            standardized_validation, truth, sd, baseline.looks, baseline
        )
        p, en, en_mcse = _grid_oc(
            tails_lrv,
            tails_cmv,
            baseline.looks,
            selected[0:1],
            selected[1:2],
            selected[2:3],
            selected[3:4],
            chunk_size=candidate_chunk_width,
        )
        validation_decisions[scenario] = p[0]
        validation_n[scenario] = en[0]
        validation_mcse_n[scenario] = en_mcse[0]
        del tails_lrv, tails_cmv
    validation_oc = make_oc(
        validation_decisions,
        validation_n,
        validation_mcse_n,
        validation_count,
        validation_seed,
    )
    val_false_go = float(validation_decisions[0, 1])
    val_false_no_go = float(validation_decisions[1, 0] + validation_decisions[1, 3])
    val_false_consider = float(max(validation_decisions[0, 2], validation_decisions[1, 2]))
    val_cgr = float(validation_decisions[1, 1])
    validation_feasible = val_false_go <= fg_limit and val_false_no_go <= fn_limit
    if fc_limit is not None:
        validation_feasible &= val_false_consider <= fc_limit
    return BOP2DCNormalOptimization(
        design,
        candidate_result,
        best_index,
        calibration_oc,
        validation_oc,
        objective,
        fg_limit,
        fn_limit,
        fc_limit,
        val_false_go,
        val_false_no_go,
        val_false_consider,
        val_cgr,
        bool(validation_feasible),
        master_seed,
    )
