"""Conduct and bounded all-scenario simulation for recovered DSF studies."""

from dataclasses import dataclass, replace

import numpy as np

from ._cdflib import _freeze
from .dose_schedule import DoseSchedulePatient
from .dose_schedule_decision import DoseScheduleDecision
from .dose_schedule_desktop_kernel import (
    _profile,
    dose_schedule_desktop_decision,
    dose_schedule_desktop_event_time,
    dose_schedule_desktop_observation,
)
from .dose_schedule_desktop_study import DoseScheduleDesktopPatient, DoseScheduleDesktopStudy
from .dose_schedule_fit import DoseScheduleFit, fit_dose_schedule
from .dose_schedule_trial import _count
from .hierarchical_binomial import summarize_chains


@dataclass(frozen=True)
class DoseScheduleDesktopSampler:
    """Explicit Python posterior settings, separate from native integration speed."""

    draws: int = 1000
    warmup: int = 500
    chains: int = 4
    max_evaluations: int = 200000
    max_work: int = 20000000

    def __post_init__(self):
        for name, low, high in (
            ("draws", 8, 100000),
            ("warmup", 0, 100000),
            ("chains", 2, 8),
            ("max_evaluations", 1, 2000000),
            ("max_work", 1, 50000000),
        ):
            object.__setattr__(self, name, _count(getattr(self, name), name, low, high))


@dataclass(frozen=True)
class DoseScheduleDesktopAnalysis:
    study: DoseScheduleDesktopStudy
    profile: str
    seed: int
    sampler: DoseScheduleDesktopSampler
    final: bool
    fit: DoseScheduleFit
    decision: DoseScheduleDecision
    pending_patients: tuple[int, ...]
    final_ready: bool
    maximum_rhat: float
    maximum_risk_mcse: float
    maximum_tail_mcse: float


def _seed(value):
    return _count(value, "seed", 0, 2**63 - 1)


def analyze_dose_schedule_desktop(
    study: DoseScheduleDesktopStudy,
    *,
    seed: int,
    profile: str = "safe",
    final: bool = False,
    sampler: DoseScheduleDesktopSampler | None = None,
) -> DoseScheduleDesktopAnalysis:
    """Fit actual histories at the captured study day and compute one decision.

    Future patients and administrations are masked. Safe final analysis requires
    completed horizon follow-up and resolved relevant low-grade outcomes.
    Native legacy permits the source final-time convention but reports pending
    patients and final readiness independently. No posterior precision is implied
    by completion: R-hat and risk/tail Monte Carlo errors accompany the result.
    """
    if not isinstance(study, DoseScheduleDesktopStudy):
        raise ValueError("study must be DoseScheduleDesktopStudy")
    _profile(profile)
    seed = _seed(seed)
    if not isinstance(final, (bool, np.bool_)):
        raise ValueError("final must be boolean")
    sampler = DoseScheduleDesktopSampler() if sampler is None else sampler
    if not isinstance(sampler, DoseScheduleDesktopSampler):
        raise ValueError("sampler must be DoseScheduleDesktopSampler")
    patients = []
    pending = []
    ready = True
    treated = np.zeros((len(study.prior_probabilities), study.schedule_count), dtype=int)
    for index, record in enumerate(study.patients):
        observed = dose_schedule_desktop_observation(
            outcome=record.outcome,
            outcome_time=record.outcome_time,
            entry_time=record.entry_time,
            study_time=study.study_time,
            horizon=study.horizon,
            low_grade_delay=study.low_grade_delay,
            profile=profile,
        )
        if not observed.included:
            ready = False
            continue
        ready = ready and observed.final_ready
        if observed.pending:
            pending.append(index)
        times = np.asarray(record.administration_times)
        mask = times < observed.time if observed.event else times <= observed.time
        patients.append(
            DoseSchedulePatient(
                observed.time,
                observed.event,
                times[mask],
                np.asarray(record.dose_indices, dtype=int)[mask],
            )
        )
        treated[record.assigned_dose, record.assigned_schedule] += 1
    if final and (not patients or (profile == "safe" and not ready)):
        raise ValueError(
            "safe final analysis requires patients with completed follow-up/adjudication"
        )
    fit = fit_dose_schedule(
        patients,
        study.prior(),
        study.schedules(),
        study.horizon,
        draws=sampler.draws,
        warmup=sampler.warmup,
        chains=sampler.chains,
        rng=np.random.default_rng(seed),
        max_evaluations=sampler.max_evaluations,
        max_work=sampler.max_work,
    )
    tail = np.mean(fit.regimen_risk > study.toxicity_limit, axis=(0, 1))
    decision = dose_schedule_desktop_decision(
        fit.risk_summary.mean,
        tail,
        treated,
        target=study.target,
        upper_probability=study.upper_probability,
        profile=profile,
        final=final,
    )
    tail_summary = summarize_chains((fit.regimen_risk > study.toxicity_limit).astype(float))
    return DoseScheduleDesktopAnalysis(
        study,
        profile,
        seed,
        sampler,
        bool(final),
        fit,
        decision,
        tuple(pending),
        ready,
        float(np.max(fit.risk_summary.split_rhat)),
        float(np.max(fit.risk_summary.batch_mean_mcse)),
        float(np.max(tail_summary.batch_mean_mcse)),
    )


@dataclass(frozen=True)
class DoseScheduleDesktopTrial:
    study: DoseScheduleDesktopStudy
    truth: np.ndarray
    sampler: DoseScheduleDesktopSampler
    max_evaluations: int
    max_work: int
    arrival_uniforms: np.ndarray
    event_uniforms: np.ndarray
    grade_uniforms: np.ndarray
    posterior_seeds: tuple[int, ...]
    analysis: DoseScheduleDesktopAnalysis
    assignments: tuple[tuple[int, int], ...]
    decisions: tuple[DoseScheduleDecision, ...]
    enrollment: np.ndarray
    toxicities: np.ndarray
    duration: float
    stopped_early: bool
    likelihood_evaluations: int
    work_units: int
    maximum_rhat: float
    maximum_risk_mcse: float
    maximum_tail_mcse: float


def _trial_final_clock(records, study, current, profile):
    if profile == "desktop_legacy":
        duration = max(r.entry_time + r.outcome_time for r in records)
        return duration, duration + 1.0
    duration = max(
        float(current),
        max(r.entry_time + study.horizon for r in records),
        max(
            r.entry_time + (r.outcome_time + (study.low_grade_delay if r.outcome == 2 else 0))
            for r in records
        ),
    )
    return duration, duration


def replay_dose_schedule_desktop_trial(
    study: DoseScheduleDesktopStudy,
    truth,
    *,
    arrival_uniforms,
    event_uniforms,
    grade_uniforms,
    posterior_seeds,
    profile="safe",
    sampler=None,
    max_evaluations=2000000,
    max_work=50000000,
) -> DoseScheduleDesktopTrial:
    """Replay one trial with separately supplied arrival/event/grade/posterior tapes.

    Arrival one is day zero; subsequent gaps are exponential. Event generation
    uses the supplied regimen truth, not triangular-hazard model parameters.
    The original 0.01-day minimum event time is retained explicitly.
    Safe planned administrations stop at the generated onset; legacy retains the
    complete planned ledger and masks it only when constructing a likelihood.
    Low-grade events all
    qualify after the selected delay, matching this simulator's assumed law;
    this is not an inferred clinical adjudication or an adaptive dose policy.
    """
    if not isinstance(study, DoseScheduleDesktopStudy) or study.patients:
        raise ValueError("trial simulation requires a study with an empty patient ledger")
    _profile(profile)
    sampler = DoseScheduleDesktopSampler() if sampler is None else sampler
    if not isinstance(sampler, DoseScheduleDesktopSampler):
        raise ValueError("invalid sampler")
    shape = (len(study.prior_probabilities), study.schedule_count)
    truth = np.asarray(truth)
    if (
        truth.shape != shape
        or truth.dtype.kind not in "iuf"
        or np.any(~np.isfinite(truth))
        or np.any((truth < 0) | (truth > 1))
    ):
        raise ValueError("truth must be a matching finite probability matrix")
    size = study.maximum_patients
    tapes = []
    for name, values in (
        ("arrival_uniforms", arrival_uniforms),
        ("event_uniforms", event_uniforms),
        ("grade_uniforms", grade_uniforms),
    ):
        raw = np.asarray(values)
        if (
            raw.shape != (size,)
            or raw.dtype.kind not in "iuf"
            or np.any(~np.isfinite(raw))
            or np.any((raw <= 0) | (raw >= 1))
        ):
            raise ValueError(f"{name} must contain maximum_patients uniforms in (0,1)")
        tapes.append(raw)
    if len(posterior_seeds) != size + 1:
        raise ValueError("posterior_seeds requires maximum_patients+1 seeds")
    seeds = tuple(_seed(v) for v in posterior_seeds)
    max_evaluations = _count(max_evaluations, "max_evaluations", 1, 2000000)
    max_work = _count(max_work, "max_work", 1, 50000000)
    if (size + 1) * sampler.chains * (sampler.draws + sampler.warmup) > max_evaluations:
        raise ValueError("minimum trial posterior evaluations exceed budget")
    if study.horizon <= 0.01:
        raise ValueError("simulation horizon must exceed native minimum event time .01 days")
    arrivals, events, grades = tapes
    assignments = []
    decisions = []
    records: list[DoseScheduleDesktopPatient] = []
    enrollment = np.zeros(shape, int)
    toxicities = np.zeros(shape, int)
    current = 0.0
    evaluations = 0
    work = 0
    diagnostics = []
    stopped = False

    def analyze(snapshot, seed, final=False):
        nonlocal evaluations, work
        remaining = replace(
            sampler,
            max_evaluations=min(sampler.max_evaluations, max_evaluations - evaluations),
            max_work=min(sampler.max_work, max_work - work),
        )
        result = analyze_dose_schedule_desktop(
            snapshot, seed=seed, profile=profile, final=final, sampler=remaining
        )
        evaluations += result.fit.likelihood_evaluations
        work += result.fit.work_units
        diagnostics.append(
            (result.maximum_rhat, result.maximum_risk_mcse, result.maximum_tail_mcse)
        )
        return result

    for index in range(size):
        if index:
            current += -study.mean_arrival_interval * np.log1p(-arrivals[index])
        snapshot = replace(study, patients=tuple(records), study_time=float(current))
        result = analyze(snapshot, seeds[index])
        decisions.append(result.decision)
        pair = result.decision.pair
        if pair is None:
            stopped = True
            break
        onset = dose_schedule_desktop_event_time(
            float(events[index]),
            float(truth[pair]),
            study.horizon,
            distribution=study.distribution,
            shape=study.shape,
        )
        event = onset < study.horizon
        onset = max(0.01, onset) if event else study.horizon
        outcome = 2 if event and grades[index] < study.low_grade_fraction else int(event)
        planned = study.schedules()[pair[1]]
        delivered = planned[planned < onset] if event else planned[planned <= onset]
        if profile == "desktop_legacy":
            delivered = planned  # Native assignment installs the entire planned ledger.
        record = DoseScheduleDesktopPatient(
            float(current),
            pair[0],
            pair[1],
            outcome,
            onset,
            tuple(delivered),
            tuple([pair[0]] * len(delivered)),
        )
        records.append(record)
        assignments.append(pair)
        enrollment[pair] += 1
        toxicities[pair] += int(event)
    final_time, analysis_time = _trial_final_clock(records, study, current, profile)
    final_study = replace(study, patients=tuple(records), study_time=analysis_time)
    final_analysis = analyze(final_study, seeds[-1], final=True)
    if stopped and profile == "desktop_legacy":
        decision = replace(
            final_analysis.decision,
            action="stop",
            pair=None,
            distance=None,
            reason="native early stop does not perform a new final selection",
        )
        final_analysis = replace(final_analysis, decision=decision)
    maxima = np.max(np.asarray(diagnostics), axis=0)
    return DoseScheduleDesktopTrial(
        study,
        _freeze(truth),
        sampler,
        max_evaluations,
        max_work,
        _freeze(arrivals),
        _freeze(events),
        _freeze(grades),
        seeds,
        final_analysis,
        tuple(assignments),
        tuple(decisions),
        _freeze(enrollment),
        _freeze(toxicities),
        final_time,
        stopped,
        evaluations,
        work,
        *map(float, maxima),
    )


@dataclass(frozen=True)
class DoseScheduleDesktopSimulation:
    study: DoseScheduleDesktopStudy
    profile: str
    seed: int
    sampler: DoseScheduleDesktopSampler
    selection_probability: np.ndarray
    selection_mcse: np.ndarray
    mean_enrollment: np.ndarray
    enrollment_mcse: np.ndarray
    pooled_toxicity: np.ndarray
    mean_duration: np.ndarray
    duration_mcse: np.ndarray
    early_stop_probability: np.ndarray
    pending_final_probability: np.ndarray
    maximum_rhat: float
    maximum_risk_mcse: float
    maximum_tail_mcse: float
    likelihood_evaluations: int
    work_units: int


def simulate_dose_schedule_desktop(
    study,
    *,
    seed,
    profile="safe",
    sampler=None,
    max_evaluations=2000000,
    max_work=50000000,
    trial_callback=None,
):
    """Run every saved scenario serially with OC Monte Carlo errors and budgets.

    Selection bins are row-major regimens followed by no selection. Reports
    separate posterior precision from operating-characteristic Monte Carlo error.
    No native integration or native RNG stream equivalence is claimed.
    """
    if not isinstance(study, DoseScheduleDesktopStudy) or study.patients or not study.scenarios:
        raise ValueError("simulation needs scenarios and an empty patient ledger")
    if trial_callback is not None and not callable(trial_callback):
        raise ValueError("trial_callback must be callable or None")
    _profile(profile)
    seed = _seed(seed)
    sampler = DoseScheduleDesktopSampler() if sampler is None else sampler
    if not isinstance(sampler, DoseScheduleDesktopSampler):
        raise ValueError("invalid sampler")
    max_evaluations = _count(max_evaluations, "max_evaluations", 1, 2000000)
    max_work = _count(max_work, "max_work", 1, 50000000)
    trials = study.repetitions * len(study.scenarios)
    minimum = (
        trials * (study.maximum_patients + 1) * sampler.chains * (sampler.draws + sampler.warmup)
    )
    if minimum > max_evaluations or trials > 10000:
        raise ValueError("requested simulation exceeds bounded trial/evaluation budget")
    rng = np.random.default_rng(seed)
    shape = (len(study.prior_probabilities), study.schedule_count)
    cells = int(np.prod(shape))
    size = study.maximum_patients
    selection = []
    selection_error = []
    enrollment = []
    enrollment_error = []
    pooled = []
    duration = []
    duration_error = []
    stops = []
    pending = []
    evaluations = 0
    work = 0
    maxima = np.zeros(3)
    for scenario_index, truth in enumerate(study.scenarios):
        selected = np.zeros(cells + 1)
        total = np.zeros(shape)
        squared = np.zeros(shape)
        tox = np.zeros(shape)
        durations = []
        stop_count = 0
        pending_count = 0
        for trial_index in range(study.repetitions):
            # Uniform zero is mapped to the nearest representable positive value.
            tape = np.maximum(rng.random((3, size)), np.nextafter(0.0, 1.0))
            result = replay_dose_schedule_desktop_trial(
                study,
                truth,
                arrival_uniforms=tape[0],
                event_uniforms=tape[1],
                grade_uniforms=tape[2],
                posterior_seeds=rng.integers(0, 2**63 - 1, size=size + 1),
                profile=profile,
                sampler=sampler,
                max_evaluations=max_evaluations - evaluations,
                max_work=max_work - work,
            )
            if trial_callback is not None:
                trial_callback(scenario_index, trial_index, result)
            evaluations += result.likelihood_evaluations
            work += result.work_units
            pair = result.analysis.decision.pair
            selected[cells if pair is None else pair[0] * shape[1] + pair[1]] += 1
            total += result.enrollment
            squared += result.enrollment**2
            tox += result.toxicities
            durations.append(result.duration)
            stop_count += result.stopped_early
            pending_count += bool(result.analysis.pending_patients)
            maxima = np.maximum(
                maxima, [result.maximum_rhat, result.maximum_risk_mcse, result.maximum_tail_mcse]
            )
        n = study.repetitions
        prob = selected / n
        mean = total / n
        selection.append(prob)
        selection_error.append(np.sqrt(prob * (1 - prob) / n))
        enrollment.append(mean)
        enrollment_error.append(
            np.sqrt(np.maximum(0, squared - total * total / n) / (n * (n - 1)))
            if n > 1
            else np.full(shape, np.nan)
        )
        pooled.append(np.divide(tox, total, out=np.zeros(shape), where=total > 0))
        duration.append(float(np.mean(durations)))
        duration_error.append(
            float(np.std(durations, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        )
        stops.append(stop_count / n)
        pending.append(pending_count / n)
    return DoseScheduleDesktopSimulation(
        study,
        profile,
        seed,
        sampler,
        _freeze(selection),
        _freeze(selection_error),
        _freeze(enrollment),
        _freeze(enrollment_error),
        _freeze(pooled),
        _freeze(duration),
        _freeze(duration_error),
        _freeze(stops),
        _freeze(pending),
        float(maxima[0]),
        float(maxima[1]),
        float(maxima[2]),
        evaluations,
        work,
    )
