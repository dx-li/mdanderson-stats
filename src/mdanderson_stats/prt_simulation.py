"""Explicit PRT1.0 calendar compatibility, with numerical holds and honest batches.

Generator/controller arithmetic is recovered; Python MCMC and random streams are
independent. Native skipped final inference is an unresolved hold, not a license
to select from stale/uninitialized posterior summaries.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from .prt_calendar import _bounded_finite
from .prt_conduct import PRTConductResult, _analyze_prt_history
from .prt_hazard import calibrate_prt_hazard, prt_arrivals_from_uniforms, prt_hazard_delays
from .prt_legacy import prt_legacy_screening
from .prt_study import PRTPatient, PRTStudy, _integer, _number, prt_study_counts


@dataclass(frozen=True)
class PRTSimulationStep:
    time: float
    candidate_index: int
    enrolled: int
    current_dose: int
    final: bool
    action: str
    reason: str
    analysis: PRTConductResult | None


@dataclass(frozen=True)
class PRTSimulationResult:
    study: PRTStudy
    profile: str
    status: str
    selected_dose: int | None
    duration: float
    candidate_indices: tuple[int, ...]
    patient_doses: tuple[int, ...]
    patient_arrivals: tuple[float, ...]
    patient_delays: tuple[float, ...]
    patients_as_of_end: tuple[PRTPatient, ...]
    assigned: tuple[int, ...]
    events: tuple[int, ...]
    declined: int
    steps: tuple[PRTSimulationStep, ...]
    rng_state_json: str
    arrival_times: tuple[float, ...]
    potential_delays: tuple[tuple[float, ...], ...]
    limits: tuple[tuple[str, float | int], ...]
    warnings: tuple[str, ...]
    generation_state_json: str | None = None

    def to_json(self) -> str:
        data = asdict(self)
        data["study"] = json.loads(self.study.to_json())
        data["rng_state"] = json.loads(data.pop("rng_state_json"))
        generated_state = data.pop("generation_state_json")
        data["generation_rng_state"] = json.loads(generated_state) if generated_state else None
        for step, original in zip(data["steps"], self.steps, strict=True):
            if original.analysis is not None:
                step["analysis"] = json.loads(original.analysis.to_json())
        return (
            json.dumps(
                {"schema": "mdanderson_stats.prt.simulation", "version": 1, **data},
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )

    def to_csv(self) -> str:
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "dose_zero_based",
                "true_toxicity",
                "assigned",
                "events_as_of_end",
                "selected",
                "status",
                "duration_days",
                "profile",
            ]
        )
        for k in range(self.study.doses):
            writer.writerow(
                [
                    k,
                    self.study.true_toxicity[k],
                    self.assigned[k],
                    self.events[k],
                    int(self.selected_dose == k),
                    self.status,
                    self.duration,
                    self.profile,
                ]
            )
        return stream.getvalue()


def _history(study: PRTStudy, doses, arrivals, delays, clock: float) -> PRTStudy:
    """Recovered 53-bit startup arithmetic, including exact-window defect."""
    window = study.interval_endpoints[-1]
    patients = []
    for dose, arrival, delay in zip(doses, arrivals, delays, strict=True):
        event = clock >= arrival + delay and delay <= window
        elapsed = delay if event else clock - arrival
        daily = (
            window + 1 if elapsed > window else math.floor(elapsed) if 1 <= elapsed < window else 0
        )
        interval = int(np.searchsorted(study.interval_endpoints, daily, side="right")) + int(event)
        patients.append(PRTPatient(dose, interval, not event))
    return replace(
        study,
        patients=tuple(patients),
        interval_endpoints=(),
        max_patients=None,
        true_toxicity=(),
        mean_interarrival=None,
        hazard_alpha=None,
        hazard_beta=None,
        repetitions=None,
    )


def _limits(
    study, profile, rng, max_analyses, max_fits, max_rhat, max_mcse, max_likelihood_evaluations
):
    if not isinstance(study, PRTStudy) or study.max_patients is None:
        raise ValueError("require a saved simulation study")
    if profile != "legacy_1_0":
        raise ValueError("explicit profile='legacy_1_0' required")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    if any(x != int(x) for x in study.interval_endpoints) or study.interval_endpoints[-1] > 1e6:
        raise ValueError("native calendar requires integer-day endpoints at most1000000")
    analyses = _integer(max_analyses, "max_analyses", 1, 800)
    fits = _integer(max_fits, "max_fits", 1, 200)
    cap = _integer(max_likelihood_evaluations, "max_likelihood_evaluations", 1, 20000000)
    rhat, mcse = _number(max_rhat, "max_rhat"), _number(max_mcse, "max_mcse")
    if not 1 < rhat <= 1.1 or not 0 < mcse <= 0.05:
        raise ValueError("require max_rhat in (1,1.1] and max_mcse in (0,.05]")
    cells = study.chains * study.draws * study.intervals * study.doses
    work = (
        2 * study.chains * study.draws * study.intervals * study.doses**3
        + study.chains * (study.draws + study.warmup) * study.intervals * study.doses
        + cap * study.intervals * study.doses
    )
    if 8 * cells > 8000000 or work > 100000000 or fits * cap > 20000000:
        raise ValueError("simulation exceeds bounded fit/work budget")
    return analyses, fits, rhat, mcse, cap


def run_prt_simulation_calendar(
    study: PRTStudy,
    arrival_times: ArrayLike,
    potential_delays: ArrayLike,
    *,
    rng: np.random.Generator,
    profile: str,
    max_analyses: int = 200,
    max_fits: int = 50,
    max_rhat: float = 1.05,
    max_mcse: float = 0.02,
    max_likelihood_evaluations: int = 200000,
) -> PRTSimulationResult:
    """Replay supplied candidate-by-dose delays through the native calendar.

    All arrays must be finite; delays beyond the window represent non-events.
    Suspended candidates are declined. Early stops retain as-of event counts.
    Final time examines only the last three patients, as in the source. A hold
    terminates this computational replicate; it is never counted as a safety stop.
    """
    analyses, fits, rhat, mcse, cap = _limits(
        study, profile, rng, max_analyses, max_fits, max_rhat, max_mcse, max_likelihood_evaluations
    )
    arrivals = _bounded_finite(arrival_times, "arrival_times", 800, 1)
    delays = _bounded_finite(potential_delays, "potential_delays", 8000, 2)
    if (
        len(arrivals) < 3
        or delays.shape != (len(arrivals), study.doses)
        or np.any(arrivals < 0)
        or np.any(np.diff(arrivals) < 0)
        or np.any(delays < 0)
        or np.any(arrivals > 1e12)
        or np.any(delays > 1e100)
    ):
        raise ValueError(
            "require3..800 sorted nonnegative arrivals and finite candidate-by-dose delays"
        )
    state = json.dumps(rng.bit_generator.state, default=lambda x: x.tolist(), sort_keys=True)
    indices: list[int] = []
    doses: list[int] = []
    entries: list[float] = []
    times: list[float] = []
    steps: list[PRTSimulationStep] = []
    index, current, declined, fit_count = 0, 0, 0, 0
    maximum = cast(int, study.max_patients)
    window = study.interval_endpoints[-1]
    clock = 0.0
    action, status, selected = "stay", "analysis_budget", None
    history = _history(study, [], [], [], clock)
    for _ in range(analyses):
        if action == "suspend":
            index += 1
            declined += 1
        else:
            cohort = min(3, maximum - len(indices))
            # Need the following candidate's clock unless this cohort completes enrollment.
            needed = cohort + int(len(indices) + cohort < maximum)
            if index + needed > len(arrivals):
                status = "arrival_exhausted"
                break
            for j in range(index, index + cohort):
                indices.append(j)
                doses.append(current)
                entries.append(float(arrivals[j]))
                times.append(float(delays[j, current]))
            index += cohort
        final = len(indices) == maximum
        if not final and index >= len(arrivals):
            status = "arrival_exhausted"
            break
        clock = (
            max(a + min(t, window) for a, t in zip(entries[-3:], times[-3:], strict=True))
            if final
            else float(arrivals[index])
        )
        history = _history(study, doses, entries, times, clock)
        _, _, assigned, assessed, _ = prt_study_counts(history)
        observed = sum(p.dose == current and p.observed_interval > 0 for p in history.patients)
        screen = prt_legacy_screening(
            assigned,
            int(assessed[current]),
            observed,
            history.patients[0].observed_interval,
            current,
            final=final,
        )
        if not screen.infer:
            action = "suspend" if screen.suspend else "stay"
            steps.append(
                PRTSimulationStep(
                    clock, index, len(indices), current, final, action, screen.reason, None
                )
            )
            if final:
                status = "final_inference_skipped"
                break
            continue
        if fit_count >= fits:
            status = "fit_budget"
            break
        fit_count += 1
        try:
            result = _analyze_prt_history(
                history,
                rng=rng,
                profile=profile,
                controller="simulation_final" if final else "simulation",
                max_rhat=rhat,
                max_mcse=mcse,
                max_likelihood_evaluations=cap,
            )
        except ValueError as exc:
            steps.append(
                PRTSimulationStep(
                    clock, index, len(indices), current, final, "hold", str(exc), None
                )
            )
            status = "numerical_hold"
            break
        action = result.action
        steps.append(
            PRTSimulationStep(
                clock, index, len(indices), current, final, action, result.rule, result
            )
        )
        if result.status != "computed":
            status = "numerical_hold"
            break
        if action == "stop":
            status = "safety_stop"
            break
        if final:
            status, selected = "selected", result.dose
            break
        if result.dose is not None:
            current = result.dose
    _, event_counts, assigned, _, _ = prt_study_counts(history)
    return PRTSimulationResult(
        study,
        profile,
        status,
        selected,
        clock,
        tuple(indices),
        tuple(doses),
        tuple(entries),
        tuple(times),
        history.patients,
        tuple(map(int, assigned)),
        tuple(map(int, event_counts.sum(axis=0))),
        declined,
        tuple(steps),
        state,
        tuple(map(float, arrivals)),
        tuple(tuple(map(float, row)) for row in delays),
        tuple(
            zip(
                ("max_analyses", "max_fits", "max_rhat", "max_mcse", "max_likelihood_evaluations"),
                (analyses, fits, rhat, mcse, cap),
                strict=True,
            )
        ),
        (
            "Explicit native arithmetic and calendar defects; independent Python sampler/RNG.",
            "Events and duration are as of termination; unresolved trials are not safety stops.",
            "Skipped final inference is unresolved; stale native summaries are not reused.",
        ),
    )


def simulate_prt(
    study: PRTStudy,
    *,
    rng: np.random.Generator,
    profile: str,
    max_analyses: int = 200,
    max_fits: int = 50,
    max_rhat: float = 1.05,
    max_mcse: float = 0.02,
    max_likelihood_evaluations: int = 200000,
) -> PRTSimulationResult:
    """Generate800 uniform-order-statistic arrivals and dose-specific legacy delays.

    Potential delays are generated up front for every candidate/dose, independently
    of the inference stream. This is distributional compatibility, not native RNG
    ordering, seeds, Gibbs posterior draws, or pathwise executable parity.
    """
    _limits(
        study, profile, rng, max_analyses, max_fits, max_rhat, max_mcse, max_likelihood_evaluations
    )
    models = [
        calibrate_prt_hazard(
            p,
            study.interval_endpoints[-1],
            cast(float, study.hazard_alpha),
            cast(float, study.hazard_beta),
            profile=profile,
        )
        for p in study.true_toxicity
    ]
    # Split streams before generation so inference effort cannot change potential outcomes.
    state = json.dumps(rng.bit_generator.state, default=lambda x: x.tolist(), sort_keys=True)
    seeds = rng.integers(0, 2**63, size=2)
    generated, inference = (np.random.default_rng(int(seed)) for seed in seeds)
    arrivals = prt_arrivals_from_uniforms(
        cast(float, study.mean_interarrival), generated.random(800)
    )
    delays = np.column_stack([prt_hazard_delays(model, generated.random(800)) for model in models])
    result = run_prt_simulation_calendar(
        study,
        arrivals,
        delays,
        rng=inference,
        profile=profile,
        max_analyses=max_analyses,
        max_fits=max_fits,
        max_rhat=max_rhat,
        max_mcse=max_mcse,
        max_likelihood_evaluations=max_likelihood_evaluations,
    )

    return replace(result, generation_state_json=state)


@dataclass(frozen=True)
class PRTSimulationBatch:
    study: PRTStudy
    trials: tuple[PRTSimulationResult, ...]
    rng_state_json: str

    def summary(self) -> dict:
        n = len(self.trials)
        counts = Counter(t.status for t in self.trials)
        unresolved = n - counts["selected"] - counts["safety_stop"]
        return dict(
            repetitions=n,
            status_counts=dict(sorted(counts.items())),
            unresolved=unresolved,
            complete=unresolved == 0,
            selection_probability=[
                sum(t.selected_dose == k for t in self.trials) / n for k in range(self.study.doses)
            ],
            safety_stop_probability=counts["safety_stop"] / n,
            unresolved_probability=unresolved / n,
            mean_assigned_as_of_end=np.mean([t.assigned for t in self.trials], axis=0).tolist(),
            mean_events_as_of_end=np.mean([t.events for t in self.trials], axis=0).tolist(),
            mean_duration_days_as_of_end=float(np.mean([t.duration for t in self.trials])),
            interpretation=(
                "All requested replicates form every denominator. "
                "Unresolved runs are incomplete operating characteristics."
            ),
        )

    def to_json(self) -> str:
        return (
            json.dumps(
                dict(
                    schema="mdanderson_stats.prt.simulation_batch",
                    version=1,
                    study=json.loads(self.study.to_json()),
                    rng_state=json.loads(self.rng_state_json),
                    summary=self.summary(),
                    trials=[json.loads(t.to_json()) for t in self.trials],
                ),
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )

    def to_csv(self) -> str:
        summary = self.summary()
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "dose_zero_based",
                "true_toxicity",
                "selection_probability_all_runs",
                "mean_assigned_as_of_end",
                "mean_events_as_of_end",
                "repetitions",
                "unresolved",
                "safety_stop_probability",
                "mean_duration_days_as_of_end",
            ]
        )
        for k in range(self.study.doses):
            writer.writerow(
                [
                    k,
                    self.study.true_toxicity[k],
                    summary["selection_probability"][k],
                    summary["mean_assigned_as_of_end"][k],
                    summary["mean_events_as_of_end"][k],
                    len(self.trials),
                    summary["unresolved"],
                    summary["safety_stop_probability"],
                    summary["mean_duration_days_as_of_end"],
                ]
            )
        return stream.getvalue()


def simulate_prt_batch(
    study: PRTStudy,
    *,
    rng: np.random.Generator,
    profile: str,
    max_analyses: int = 200,
    max_fits: int = 50,
    max_rhat: float = 1.05,
    max_mcse: float = 0.02,
    max_likelihood_evaluations: int = 200000,
) -> PRTSimulationBatch:
    """Run the saved repetition count, retaining holds in unconditional denominators.

    At most100 runs and one billion permitted likelihood calls per batch. Larger
    jobs must be explicitly split and combined without dropping unresolved runs.
    """
    _, fits, _, _, cap = _limits(
        study, profile, rng, max_analyses, max_fits, max_rhat, max_mcse, max_likelihood_evaluations
    )
    n = cast(int, study.repetitions)
    if (
        n > 100
        or n * fits * cap > 1000000000
        or n * max_analyses * cast(int, study.max_patients) > 2000000
    ):
        raise ValueError("batch exceeds bounded run, likelihood-call, or retained-history budget")
    state = json.dumps(rng.bit_generator.state, default=lambda x: x.tolist(), sort_keys=True)
    trials = tuple(
        simulate_prt(
            study,
            rng=rng,
            profile=profile,
            max_analyses=max_analyses,
            max_fits=max_fits,
            max_rhat=max_rhat,
            max_mcse=max_mcse,
            max_likelihood_evaluations=cap,
        )
        for _ in range(n)
    )
    return PRTSimulationBatch(study, trials, state)
