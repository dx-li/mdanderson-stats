"""Integrated calendar simulation of the six-dose P12Xuelin workflow."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar
from .parallel_phase12_decision import (
    Phase12SourceDecision,
    phase12_source_decision,
    phase12_source_final_selection,
)
from .parallel_phase12_model import (
    Phase12ModelFit,
    Phase12Snapshot,
    fit_phase12_model,
    phase12_snapshot,
)
from .parallel_phase12_progression import phase12_accrual_ready, phase12_phase_one


@dataclass(frozen=True)
class Phase12CalendarAnalysis:
    time: float
    snapshot: Phase12Snapshot
    decision: Phase12SourceDecision | None
    max_split_rhat: float


@dataclass(frozen=True)
class Phase12CalendarTrial:
    records: FloatArray
    phases: FloatArray
    attempts: FloatArray
    analyses: tuple[Phase12CalendarAnalysis, ...]
    phase_two_start: int | None
    closed: np.ndarray
    suspended: np.ndarray
    phase_one_admissible: np.ndarray
    early_selected: int | None
    future_selected: int | None
    selected_eligible: bool | None
    reason: str
    stop_time: float
    final_analysis_time: float | None
    last_fit: Phase12ModelFit | None
    data_seed: int
    posterior_seed: int


def _integer(value: int, name: str, low: int, high: int) -> int:
    x = scalar(value, name)
    if x != np.floor(x) or not low <= x <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return int(x)


def simulate_phase12_calendar(
    toxicity_probability: ArrayLike,
    efficacy_probability: ArrayLike,
    *,
    accrual_per_year: float = 72,
    efficacy_window: float = 84,
    toxicity_window: float = 28,
    max_patients: int = 80,
    max_duration: float = float("inf"),
    max_attempts: int = 100000,
    complete_followup: bool = False,
    draws: int = 2000,
    warmup: int = 1000,
    chains: int = 4,
    rng: np.random.Generator,
) -> Phase12CalendarTrial:
    """Simulate phase I, blocked accrual attempts, phase II and source final selection.

    Source defaults use rounded exponential arrival gaps and rounded uniform
    event delays; negative outcomes resolve at their complete window. The source
    final analysis is at the last attempted arrival, not after full follow-up.
    complete_followup=True extends final follow-up only when the trial has not
    already closed/terminated. Early selections retain the documented source
    eligibility quirk. No exact native RNG or importance-sampler parity is claimed.
    """
    tox, eff = (
        finite(toxicity_probability, "toxicity_probability"),
        finite(efficacy_probability, "efficacy_probability"),
    )
    if (
        tox.shape != (6,)
        or eff.shape != (6,)
        or np.any((tox < 0) | (tox > 1))
        or np.any((eff < 0) | (eff > 1))
    ):
        raise ValueError("toxicity and efficacy must be six probabilities in [0,1]")
    rate = scalar(accrual_per_year, "accrual_per_year")
    ew, tw = scalar(efficacy_window, "efficacy_window"), scalar(toxicity_window, "toxicity_window")
    if rate <= 0 or not np.isfinite(365 / rate) or ew < 1 or tw < 1:
        raise ValueError(
            "accrual must give a finite positive mean gap; windows must be at least one day"
        )
    maximum = _integer(max_patients, "max_patients", 1, 10000)
    limit = _integer(max_attempts, "max_attempts", 1, 1000000)
    draws = _integer(draws, "draws", 8, 100000)
    warmup = _integer(warmup, "warmup", 0, 100000)
    chains = _integer(chains, "chains", 2, 16)
    duration = float(max_duration)
    if np.isnan(duration) or duration <= 0:
        raise ValueError("max_duration must be positive (infinity is allowed)")
    if not isinstance(complete_followup, (bool, np.bool_)):
        raise ValueError("complete_followup must be boolean")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    seeds = rng.integers(0, 2**63, size=2, dtype=np.int64)
    data_rng, posterior_rng = (np.random.default_rng(int(s)) for s in seeds)
    records: list[list[float]] = []
    phases: list[int] = []
    attempts: list[tuple[float, int, int, bool]] = []
    analyses: list[Phase12CalendarAnalysis] = []
    opened = np.array([1, 0, 0, 0, 0, 0], dtype=bool)
    phase_closed = np.zeros(6, dtype=bool)
    admissible = np.zeros(6, dtype=bool)
    closed = np.zeros(6, dtype=bool)
    suspended = np.zeros(6, dtype=bool)
    weights = np.array([1.0, 0, 0, 0, 0, 0])
    start: int | None = None
    current_dose = 0
    next_time = time = 0.0
    stopped = False
    reason = "maximum enrollment"
    early = future = None
    selected_eligible: bool | None = None
    last_fit: Phase12ModelFit | None = None
    cached_tally: FloatArray | None = None

    def analyze(at: float) -> tuple[Phase12Snapshot, Phase12ModelFit, float]:
        nonlocal last_fit, cached_tally
        snapshot = phase12_snapshot(records, time=at)
        if cached_tally is None or not np.array_equal(snapshot.tally, cached_tally):
            last_fit = fit_phase12_model(
                snapshot.tally, draws=draws, warmup=warmup, chains=chains, rng=posterior_rng
            )
            cached_tally = snapshot.tally
        assert last_fit is not None
        diagnostic = float(np.max(last_fit.coefficient_summary.split_rhat))
        return snapshot, last_fit, diagnostic

    def arrival(at: float) -> float:
        gap = float(np.floor(data_rng.exponential(365 / rate) + 0.5))
        value = at + gap
        if not np.isfinite(value):
            raise ArithmeticError("arrival time exceeds floating-point range")
        return value

    while len(records) < maximum and next_time < duration:
        if len(attempts) >= limit:
            raise RuntimeError(
                "max_attempts reached; no time advance or completed trial is fabricated"
            )
        time = next_time
        ready = phase12_accrual_ready(
            records, time=time, phase_two_start=start, max_patients=maximum
        )
        attempts.append((time, len(records), int(start is not None), not ready))
        if not ready:
            next_time = arrival(time)
            continue
        if start is None:
            snap = phase12_snapshot(records, time=time)
            step = phase12_phase_one(
                snap.enrolled,
                snap.tally[:, 3],
                current_dose=current_dose,
                opened=opened,
                closed=phase_closed,
                admissible=admissible,
            )
            opened, phase_closed, admissible = step.opened, step.closed, step.admissible
            weights = step.probability
            if step.done:
                closed = ~admissible
                if step.trial_closed or admissible.sum() <= 1:
                    reason = (
                        "phase-I toxicity" if step.trial_closed else "at most one admissible dose"
                    )
                    stopped = True
                    break
                start = len(records)
                weights = admissible / admissible.sum()
        if start is not None and len(records) > start and (len(records) - start) % 5 == 0:
            snapshot, fit, diagnostic = analyze(time)
            decision = phase12_source_decision(
                fit,
                snapshot.enrolled,
                phase_one_admissible=admissible,
                closed=closed,
                suspended=suspended,
            )
            closed, suspended = decision.closed, decision.suspended
            analyses.append(Phase12CalendarAnalysis(time, snapshot, decision, diagnostic))
            if decision.terminated or decision.arm_closed:
                reason, early = decision.reason, decision.selected
                selected_eligible = decision.selected_eligible
                stopped = True
                break
            weights = decision.probability
        current_dose = int(data_rng.choice(6, p=weights))
        response = int(data_rng.random() < eff[current_dose])
        response_delay = min(float(np.floor(data_rng.uniform(1, ew) + 0.5)), ew) if response else ew
        toxic = int(data_rng.random() < tox[current_dose])
        toxicity_delay = min(float(np.floor(data_rng.uniform(1, tw) + 0.5)), tw) if toxic else tw
        response_time, toxicity_time = time + response_delay, time + toxicity_delay
        if not np.all(np.isfinite([response_time, toxicity_time])):
            raise ArithmeticError("outcome time exceeds floating-point range")
        records.append([current_dose, time, response, response_time, toxic, toxicity_time])
        phases.append(int(start is not None))
        next_time = arrival(time)
    final_time = None
    if not stopped:
        reason = "maximum enrollment" if len(records) >= maximum else "maximum duration"
        final_time = time
        if complete_followup and records:
            final_time = max(time, float(np.max(np.asarray(records)[:, [3, 5]])))
        snapshot, fit, diagnostic = analyze(final_time)
        future = phase12_source_final_selection(fit, closed=closed, suspended=suspended)
        selected_eligible = True if future is not None else None
        analyses.append(Phase12CalendarAnalysis(final_time, snapshot, None, diagnostic))
    masks = [np.frombuffer(x.tobytes(), dtype=bool) for x in (closed, suspended, admissible)]
    return Phase12CalendarTrial(
        _freeze(np.asarray(records).reshape(-1, 6)),
        _freeze(phases),
        _freeze(attempts),
        tuple(analyses),
        start,
        masks[0],
        masks[1],
        masks[2],
        early,
        future,
        selected_eligible,
        reason,
        time,
        final_time,
        last_fit,
        int(seeds[0]),
        int(seeds[1]),
    )
