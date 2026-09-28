"""Bounded operating-characteristic simulation for TOP two-endpoint designs."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite, scalar
from .bayesian_monitoring import _integer
from .top_endpoints import TOPMultiEndpointDesign
from .top_multi_calendar import _run_top_multiendpoint_batch


@dataclass(frozen=True)
class TOPMultiEndpointSimulation:
    """Aggregate trial records without retaining per-look histories."""

    patients: NDArray[np.int64]
    events: NDArray[np.int64]
    pending: NDArray[np.int64]
    decision: NDArray[np.str_]
    duration: FloatArray
    interim_suspension_time: FloatArray
    final_followup_time: FloatArray
    success_probability: float
    success_mcse: float
    early_success_probability: float
    final_success_probability: float
    futility_stop_probability: float
    toxicity_stop_probability: float
    combined_stop_probability: float


@dataclass(frozen=True)
class _TopMultiUniforms:
    """Parameter-independent random numbers reused by calibration scenarios."""

    arrival: FloatArray
    joint_cell: FloatArray
    timing_component: FloatArray
    timing_within: FloatArray


def _draw_multiendpoint_uniforms(
    generator: np.random.Generator, shape: tuple[int, int]
) -> _TopMultiUniforms:
    return _TopMultiUniforms(
        generator.random(shape),
        generator.random(shape),
        generator.random((*shape, 2)),
        generator.random((*shape, 2)),
    )


def _multiendpoint_potential_from_uniforms(
    design: TOPMultiEndpointDesign,
    truth: FloatArray,
    accrual_rate: float,
    arrival: str,
    timing: FloatArray,
    uniforms: _TopMultiUniforms,
) -> tuple[FloatArray, FloatArray]:
    """Transform shared uniforms into one scenario's potential calendar data."""
    shape = uniforms.joint_cell.shape
    gaps = (
        np.full(shape, 1 / accrual_rate)
        if arrival == "fixed"
        else -np.log1p(-uniforms.arrival) / accrual_rate
    )
    cell_cdf = np.cumsum(truth)
    cell_cdf /= cell_cdf[-1]
    cell = np.searchsorted(cell_cdf, uniforms.joint_cell, side="right")
    event = np.stack((np.isin(cell, (0, 1)), np.isin(cell, (0, 2))), axis=-1)
    delays = np.full((*shape, 2), np.inf)
    windows = np.asarray(design.windows)
    for endpoint in range(2):
        component_cdf = np.cumsum(timing[endpoint])
        component_cdf /= component_cdf[-1]
        component = np.searchsorted(
            component_cdf,
            uniforms.timing_component[..., endpoint],
            side="right",
        )
        within = uniforms.timing_within[..., endpoint]
        potential = ((component + within) / 3) * windows[endpoint]
        delays[..., endpoint] = np.where(event[..., endpoint], potential, np.inf)
    return gaps, delays


def _timing_probabilities(value: ArrayLike, name: str) -> FloatArray:
    if np.shape(value) not in ((3,), (2, 3)):
        raise ValueError(f"{name} must be a triple or two nonnegative triples summing to one")
    raw = finite(value, name).copy()
    if raw.shape == (3,):
        raw = np.broadcast_to(raw, (2, 3)).copy()
    if (
        raw.shape != (2, 3)
        or np.any(raw < 0)
        or np.any(np.abs(raw.sum(axis=1) - 1) > 32 * np.finfo(float).eps)
    ):
        raise ValueError(f"{name} must be a triple or two nonnegative triples summing to one")
    raw /= raw.sum(axis=1, keepdims=True)
    return raw


def simulate_top_multiendpoint(
    design: TOPMultiEndpointDesign,
    true_joint_probabilities: ArrayLike,
    accrual_rate: float,
    *,
    trials: int = 10000,
    arrival: str = "exponential",
    truth_timing_probabilities: ArrayLike | None = None,
    rng: int | np.random.Generator | None = None,
) -> TOPMultiEndpointSimulation:
    """Simulate joint binary endpoint outcomes and their calendar ascertainment.

    Truth probabilities use the cell order ``(1,1), (1,0), (0,1), (0,0)``.
    Conditional event times are drawn independently by endpoint, given each
    patient's two-cell outcome, from the three-window mixture uniforms. If no
    separate truth timing is given, the design's analysis timing distribution is
    used. NumPy randomness and calendar bookkeeping are explicit Python choices;
    native seed/RNG parity is not claimed.

    At most 2 million endpoint-patient cells are materialized, and only compact
    per-trial summaries are retained (no trial-by-time histories).
    """
    if not isinstance(design, TOPMultiEndpointDesign):
        raise TypeError("design must be a TOPMultiEndpointDesign")
    n_trials = _integer(trials, "trials")
    if not 1 <= n_trials <= 100_000:
        raise ValueError("trials must be an integer from 1 through 100,000")
    cells = n_trials * design.max_subjects * 2
    if cells > 2_000_000:
        raise ValueError("simulation exceeds the 2,000,000 endpoint-patient-cell limit")
    rate = scalar(accrual_rate, "accrual_rate")
    if rate <= 0 or not np.isfinite(1 / rate):
        raise ValueError("accrual_rate must be positive with a representable mean gap")
    if arrival not in ("fixed", "exponential"):
        raise ValueError("arrival must be 'fixed' or 'exponential'")
    if np.shape(true_joint_probabilities) != (4,):
        raise ValueError("true_joint_probabilities must contain four cell probabilities")
    truth = finite(true_joint_probabilities, "true_joint_probabilities")
    if truth.shape != (4,) or np.any(truth < 0):
        raise ValueError(
            "true_joint_probabilities must contain four nonnegative cell probabilities"
        )
    total = float(truth.sum())
    if not np.isfinite(total) or abs(total - 1) > 32 * np.finfo(float).eps:
        raise ValueError("true_joint_probabilities must sum to one")
    truth = truth / total
    timing = (
        np.array(design.timing_probabilities, copy=True)
        if truth_timing_probabilities is None
        else _timing_probabilities(truth_timing_probabilities, "truth_timing_probabilities")
    )
    generator = np.random.default_rng(rng)
    shape = (n_trials, design.max_subjects)
    uniforms = _draw_multiendpoint_uniforms(generator, shape)
    gaps, delays = _multiendpoint_potential_from_uniforms(
        design, truth, rate, arrival, timing, uniforms
    )

    result = _run_top_multiendpoint_batch(design, gaps, delays)
    success = result.decisions == "success"
    success_probability = float(success.mean())
    early = success & (result.patients < design.max_subjects)
    final = success & (result.patients == design.max_subjects)
    return TOPMultiEndpointSimulation(
        result.patients,
        result.events,
        result.pending,
        result.decisions,
        result.final_times,
        result.interim_pauses,
        result.final_waits,
        success_probability,
        float(np.sqrt(success_probability * (1 - success_probability) / n_trials)),
        float(early.mean()),
        float(final.mean()),
        float((result.decisions == "stop_futility").mean()),
        float((result.decisions == "stop_toxicity").mean()),
        float((result.decisions == "stop_futility_toxicity").mean()),
    )
