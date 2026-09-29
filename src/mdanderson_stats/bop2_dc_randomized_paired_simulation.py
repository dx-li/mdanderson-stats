"""Bounded Monte Carlo operating characteristics for paired binary endpoints."""

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count
from .bop2_dc_randomized_paired import BOP2DCRandomizedPairedDesign

IntArray = NDArray[np.int64]
_SEED_DTYPE = np.dtype(np.uint64)
_TERMINAL = ("graduate", "stop_no_go", "final_go", "final_consider", "final_no_go")
_LOOK_ACTIONS = ("continue", "stop_no_go", "graduate", "final_go", "final_consider", "final_no_go")
_MAX_TRIALS = 5_000
_MAX_SIMULATION_PATIENTS = 1_000_000
_MAX_CACHE_ENTRIES = 20_000
_MAX_COMPARISON_WORK = 2_000_000_000
_MAX_RESULT_CELLS = 2_000_000
_QUAD_WORK_PER_CACHE_ENTRY = 4 * 2 * 3 * 21 * (2 * 300 - 1)


def _probability_row(value: ArrayLike, name: str) -> FloatArray:
    if isinstance(value, np.ndarray):
        shape = value.shape
        if value.size > 4:
            raise ValueError(f"{name} must be one four-cell probability vector")
    elif isinstance(value, (list, tuple)):
        if len(value) != 4 or any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must be one four-cell probability vector")
        shape = (len(value),)
    else:
        raise ValueError(f"{name} must be a four-cell probability vector")
    if shape != (4,) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a real length-four vector")
    result = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} must contain probabilities in [0,1]")
    total = float(np.sum(result))
    if abs(total - 1.0) > 1e-14:
        raise ValueError(f"{name} probabilities must sum to one")
    return result / total


def _freeze(value: ArrayLike, dtype: np.dtype | type = np.float64) -> NDArray:
    array = np.array(value, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


def _count_seed(seed: ArrayLike) -> IntArray:
    result = count(seed, "random seed")
    if result.ndim != 0 or result >= 2**53:
        raise ValueError("rng must be a NumPy Generator, integer seed, or None")
    return np.asarray(result, dtype=np.int64)


def _cache_entry_bound(design: BOP2DCRandomizedPairedDesign, n_trials: int) -> int:
    total = 0
    for look in design.looks:
        n = int(look)
        n_control = int(np.count_nonzero(design.arm_assignments[:n] == 0))
        n_treatment = n - n_control
        marginal_pairs = (n_control + 1) * (n_treatment + 1)
        total += min(2 * n_trials, marginal_pairs)
    return total


@dataclass(frozen=True)
class BOP2DCRandomizedPairedSimulation:
    """Compact summaries without patient histories.

    Per-look action probabilities and Monte Carlo errors use all simulated
    trials as denominator; ``look_reached_counts`` is the conditional-rate
    denominator for trials reaching that analysis.
    """

    control_probabilities: FloatArray
    treatment_probabilities: FloatArray
    trial_seeds: NDArray[np.uint64]
    terminal_names: tuple[str, ...]
    terminal_counts: IntArray
    terminal_probabilities: FloatArray
    terminal_mcse: FloatArray
    look_action_names: tuple[str, ...]
    look_reached_counts: IntArray
    look_action_counts: IntArray
    look_action_probabilities: FloatArray
    look_action_mcse: FloatArray
    terminal_sample_sizes: IntArray
    terminal_decisions: NDArray[np.str_]
    expected_sample_size: float
    expected_sample_size_mcse: float
    unique_marginal_comparisons: int
    maximum_quadrature_error: float


def simulate_bop2_dc_randomized_paired(
    design: BOP2DCRandomizedPairedDesign,
    control_probabilities: ArrayLike,
    treatment_probabilities: ArrayLike,
    *,
    n_trials: int = 100,
    rng: np.random.Generator | int | None = None,
) -> BOP2DCRandomizedPairedSimulation:
    """Simulate paired joint categories under a fixed randomized allocation.

    Truth vectors use the core category order: both, endpoint-1 only,
    endpoint-2 only, neither. One supplied seed/generator creates independent
    per-trial streams; returned ``trial_seeds`` replay each path separately.
    Work and retained summaries are bounded before any random draw.
    """
    if not isinstance(design, BOP2DCRandomizedPairedDesign):
        raise TypeError("design must be BOP2DCRandomizedPairedDesign")
    n_value = float(_count_seed(n_trials))
    if n_value != n_trials or not 1 <= n_value <= _MAX_TRIALS:
        raise ValueError(f"n_trials must be an integer in [1,{_MAX_TRIALS}]")
    trials = int(n_value)
    control_truth = _probability_row(control_probabilities, "control_probabilities")
    treatment_truth = _probability_row(treatment_probabilities, "treatment_probabilities")
    paths = trials * design.max_subjects
    if paths > _MAX_SIMULATION_PATIENTS:
        raise ValueError("paired simulation exceeds its patient-path work bound")
    look_count = int(design.looks.size)
    # Include retained outputs and their live accumulation/normalization copies.
    result_cells = 2 * (23 + 3 * trials + 19 * look_count)
    workspace_cells = result_cells + 10 * _cache_entry_bound(design, trials)
    if workspace_cells > _MAX_RESULT_CELLS:
        raise ValueError("paired simulation summary exceeds its retained-cell bound")
    cache_bound = _cache_entry_bound(design, trials)
    comparison_work = cache_bound * _QUAD_WORK_PER_CACHE_ENTRY
    if cache_bound > _MAX_CACHE_ENTRIES or comparison_work > _MAX_COMPARISON_WORK:
        raise ValueError("paired simulation exceeds its posterior comparison/cache budget")
    if isinstance(rng, np.random.Generator):
        master = rng
    elif rng is None:
        master = np.random.default_rng()
    elif np.isscalar(rng):
        master = np.random.default_rng(int(_count_seed(rng)))
    else:
        raise ValueError("rng must be a NumPy Generator, integer seed, or None")

    seeds = master.integers(0, np.iinfo(np.uint64).max, size=trials, dtype=np.uint64)
    terminal_counts = np.zeros(len(_TERMINAL), dtype=np.int64)
    look_reached = np.zeros(design.looks.size, dtype=np.int64)
    look_counts = np.zeros((design.looks.size, len(_LOOK_ACTIONS)), dtype=np.int64)
    terminal_n = np.zeros(trials, dtype=np.int64)
    terminal_decisions = np.full(trials, "", dtype="U16")
    tail_cache: dict[tuple[int, int, int], tuple[FloatArray, FloatArray]] = {}
    max_error = 0.0
    look_indices = {int(n): i for i, n in enumerate(design.looks)}

    for trial_index, raw_seed in enumerate(seeds):
        trial_rng = np.random.default_rng(int(raw_seed))
        arm_counts = np.zeros((2, 4), dtype=np.int64)
        last_look_n = 0
        terminal_label = "continue"
        for patient_index, arm in enumerate(design.arm_assignments, start=1):
            category = int(trial_rng.choice(4, p=control_truth if arm == 0 else treatment_truth))
            arm_counts[int(arm), category] += 1
            look_index = look_indices.get(patient_index)
            if look_index is None:
                continue
            look_reached[look_index] += 1
            control_n = int(np.sum(arm_counts[0]))
            treatment_n = int(np.sum(arm_counts[1]))
            c_success = (
                int(arm_counts[0, 0] + arm_counts[0, 1]),
                int(arm_counts[0, 0] + arm_counts[0, 2]),
            )
            t_success = (
                int(arm_counts[1, 0] + arm_counts[1, 1]),
                int(arm_counts[1, 0] + arm_counts[1, 2]),
            )
            probabilities = np.empty((1, 2, 2), dtype=np.float64)
            errors = np.empty_like(probabilities)
            for endpoint_index in range(2):
                key = (patient_index, c_success[endpoint_index], t_success[endpoint_index])
                cached = tail_cache.get(key)
                if cached is None:
                    repeated_c = np.full((1, 2), c_success[endpoint_index], dtype=np.int64)
                    repeated_t = np.full((1, 2), t_success[endpoint_index], dtype=np.int64)
                    cached = design._posterior_tails_from_success_counts(
                        control_n, treatment_n, repeated_c, repeated_t
                    )
                    cached = (_freeze(cached[0][0]), _freeze(cached[1][0]))
                    tail_cache[key] = cached
                probabilities[0, endpoint_index] = cached[0][endpoint_index]
                errors[0, endpoint_index] = cached[1][endpoint_index]
                max_error = max(max_error, float(np.max(cached[1][endpoint_index])))
            labels = design._paired_decisions_from_tails(
                probabilities, errors, total_n=patient_index
            )
            terminal_label = str(labels[0])
            last_look_n = patient_index
            look_counts[look_index, _LOOK_ACTIONS.index(terminal_label)] += 1
            if terminal_label != "continue":
                terminal_n[trial_index] = patient_index
                terminal_decisions[trial_index] = terminal_label
                terminal_counts[_TERMINAL.index(terminal_label)] += 1
                break
        if terminal_n[trial_index] == 0:
            if last_look_n != design.max_subjects or terminal_label not in _TERMINAL:
                raise ArithmeticError(
                    "fixed allocation ended without a terminal final-look decision"
                )
            terminal_n[trial_index] = last_look_n
            terminal_decisions[trial_index] = terminal_label
            terminal_counts[_TERMINAL.index(terminal_label)] += 1

    terminal_probability = terminal_counts.astype(np.float64) / trials
    terminal_mcse = np.sqrt(terminal_probability * (1.0 - terminal_probability) / trials)
    look_probability = look_counts.astype(np.float64) / trials
    look_mcse = np.sqrt(look_probability * (1.0 - look_probability) / trials)
    expected_n = float(np.mean(terminal_n))
    expected_n_mcse = (
        float(np.std(terminal_n, ddof=1) / sqrt(trials)) if trials > 1 else float("nan")
    )
    return BOP2DCRandomizedPairedSimulation(
        _freeze(control_truth),
        _freeze(treatment_truth),
        _freeze(seeds, _SEED_DTYPE),
        _TERMINAL,
        _freeze(terminal_counts, np.int64),
        _freeze(terminal_probability),
        _freeze(terminal_mcse),
        _LOOK_ACTIONS,
        _freeze(look_reached, np.int64),
        _freeze(look_counts, np.int64),
        _freeze(look_probability),
        _freeze(look_mcse),
        _freeze(terminal_n, np.int64),
        _freeze(terminal_decisions, np.dtype("U16")),
        expected_n,
        expected_n_mcse,
        len(tail_cache),
        max_error,
    )
