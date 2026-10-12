"""Versioned Python studies and serial simulation for the ARAND native profile.

JSON and reports are explicitly Python formats. Native XML defaults, wrapper RNG
streams and native report quantiles are not silently inferred by these APIs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from html import escape

import numpy as np

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar
from .arand_native import (
    ArandNativeArmLimits,
    ArandNativeGates,
    ArandNativeMonitor,
    _integer,
    _unit,
)
from .arand_native_trial import ArandNativeRules, ArandNativeTrial, arand_native_replay
from .arand_posterior import arand_binary_posterior, arand_survival_posterior
from .arand_simulation import _nonnegative_seed, _trial_seed_streams


@dataclass(frozen=True)
class ArandNativeStudy:
    prior: tuple[tuple[float, float], ...]
    rules: ArandNativeRules
    monitor: ArandNativeMonitor
    true_parameters: tuple[float, ...]
    accrual_rate: float
    family: str = "binary"
    parameter: str = "mean"
    binary_delay: float | tuple[float, ...] = 1
    censor_time: float | None = None
    maximize: bool = True
    futility_threshold: float | tuple[float, ...] | None = None
    absolute_tolerance: float = 1e-9
    max_candidates: int = 1000

    def __post_init__(self) -> None:
        k = len(self.rules.arms)
        prior = finite(self.prior, "prior")
        truth = finite(self.true_parameters, "true_parameters")
        if prior.shape != (k, 2) or np.any(prior <= 0) or truth.shape != (k,):
            raise ValueError("one positive prior pair and true parameter per arm are required")
        if self.family not in ("binary", "exponential") or self.parameter not in ("mean", "median"):
            raise ValueError("unsupported model")
        if self.family == "binary":
            if np.any((truth < 0) | (truth > 1)) or self.censor_time is not None:
                raise ValueError("binary truth must be in [0,1]; censor_time is survival-only")
        elif np.any(truth <= 0):
            raise ValueError("survival truth must be positive")
        rate = scalar(self.accrual_rate, "accrual_rate")
        if rate <= 0 or not np.isfinite(1 / rate):
            raise ValueError("accrual_rate must be positive with a finite reciprocal")
        delays = finite(self.binary_delay, "binary_delay")
        if delays.ndim > 1 or (delays.ndim == 1 and delays.shape != (k,)) or np.any(delays < 0):
            raise ValueError("binary_delay must be nonnegative scalar or one value per arm")
        if delays.ndim == 1:
            object.__setattr__(self, "binary_delay", tuple(float(x) for x in delays))
        if self.futility_threshold is not None and np.asarray(self.futility_threshold).ndim == 1:
            object.__setattr__(
                self,
                "futility_threshold",
                tuple(float(x) for x in np.asarray(self.futility_threshold)),
            )
        if self.censor_time is not None and scalar(self.censor_time, "censor_time") <= 0:
            raise ValueError("censor_time must be positive")
        object.__setattr__(
            self, "max_candidates", _integer(self.max_candidates, "max_candidates", 1)
        )
        if self.max_candidates > 10000:
            raise ValueError("max_candidates must be at most 10000")
        if not isinstance(self.maximize, bool):
            raise ValueError("maximize must be boolean")
        object.__setattr__(self, "prior", tuple((float(a), float(b)) for a, b in prior))
        object.__setattr__(self, "true_parameters", tuple(float(t) for t in truth))
        # Validate the complete posterior contract even if a trial stops immediately.
        if self.family == "binary":
            arand_binary_posterior(
                [0] * k,
                [0] * k,
                prior=prior,
                tuning=0,
                maximize=self.maximize,
                threshold=self.futility_threshold,
                absolute_tolerance=self.absolute_tolerance,
            )
        else:
            arand_survival_posterior(
                [0] * k,
                [0] * k,
                prior=prior,
                tuning=0,
                maximize=self.maximize,
                parameter=self.parameter,
                threshold=self.futility_threshold,
                absolute_tolerance=self.absolute_tolerance,
            )


def arand_native_study_json(study: ArandNativeStudy) -> str:
    """Save explicit version-1 Python inputs; this is not an ARAND XML export."""
    return (
        json.dumps(
            {"schema": "mdanderson-stats/arand-native-study", "version": 1, "study": asdict(study)},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def read_arand_native_study(text: str) -> ArandNativeStudy:
    """Strict bounded JSON reader; reject unknown fields and unsupported versions."""
    if not isinstance(text, str) or len(text.encode("utf-8")) > 1_000_000:
        raise ValueError("study must be at most one megabyte of JSON text")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        obj = json.loads(text, object_pairs_hook=unique)
        if (
            set(obj) != {"schema", "version", "study"}
            or obj["schema"] != "mdanderson-stats/arand-native-study"
            or type(obj["version"]) is not int
            or obj["version"] != 1
        ):
            raise ValueError("unsupported study schema")
        data = obj["study"]
        rules = data.pop("rules")
        rules["arms"] = tuple(ArandNativeArmLimits(**a) for a in rules["arms"])
        if "gates" in rules:
            rules["gates"] = ArandNativeGates(**rules["gates"])
        data["rules"] = ArandNativeRules(**rules)
        data["monitor"] = ArandNativeMonitor(**data["monitor"])
        return ArandNativeStudy(**data)
    except (TypeError, KeyError, AttributeError, RecursionError) as exc:
        raise ValueError("invalid native-profile study") from exc


def simulate_arand_native_trial(study: ArandNativeStudy, *, seed: int) -> ArandNativeTrial:
    """Generate fixed-window binary or exponential/censored TTE candidate tapes.

    Independent NumPy child streams generate interarrival gaps, outcomes and
    assignments. Mean/median conversion affects survival times, not months.
    The first candidate starts at zero; later gaps are exponential. No native
    RNG identity or native seed mapping is asserted.
    """
    seed = _nonnegative_seed(seed)
    arrival_rng, outcome_rng, assignment_rng = _trial_seed_streams(seed)
    n, k = study.max_candidates, len(study.rules.arms)
    # Original SimulateSingleTrial starts the first candidate at zero, then
    # draws a new exponential gap after each enrollment/attempt.
    arrivals = np.concatenate(
        ([0.0], np.cumsum(arrival_rng.exponential(1 / study.accrual_rate, size=n - 1)))
    )
    u = assignment_rng.random(n)
    if study.family == "binary":
        values = (outcome_rng.random((n, k)) < np.asarray(study.true_parameters)).astype(int)
        delays = np.broadcast_to(np.asarray(study.binary_delay), (n, k)).copy()
    else:
        means = np.asarray(study.true_parameters) / (
            np.log(2) if study.parameter == "median" else 1
        )
        raw = outcome_rng.exponential(means, size=(n, k))
        values = (
            np.ones((n, k), dtype=int)
            if study.censor_time is None
            else (raw <= study.censor_time).astype(int)
        )
        delays = raw if study.censor_time is None else np.minimum(raw, study.censor_time)
    return arand_native_replay(
        arrivals,
        u,
        delays,
        values,
        prior=study.prior,
        rules=study.rules,
        monitor=study.monitor,
        family=study.family,
        parameter=study.parameter,
        maximize=study.maximize,
        futility_threshold=study.futility_threshold,
        absolute_tolerance=study.absolute_tolerance,
    )


@dataclass(frozen=True)
class ArandNativeSummary:
    study_json: str
    trial_seeds: tuple[int, ...]
    selected: tuple[int, ...]
    no_selection: int
    completed: int
    early_selection: int
    mean_enrollment: FloatArray
    enrollment_mcse: FloatArray
    enrollment_quantiles: FloatArray
    duration_mean: float
    duration_mcse: float
    selected_early: tuple[int, ...]
    dropped_early: tuple[int, ...]
    final_status_counts: tuple[tuple[int, ...], ...]
    legacy_enrollment_quantiles: FloatArray | None


def _legacy_enrollment_percentiles(counts: FloatArray) -> FloatArray | None:
    n = len(counts)
    if n < 100:
        return None
    indices = (np.ceil(n * np.array([0.025, 0.975]) - 1) + 1).astype(int)
    return _freeze(np.sort(counts, axis=0)[indices].T)


def simulate_arand_native(
    study: ArandNativeStudy,
    *,
    trials: int,
    seed: int,
) -> ArandNativeSummary:
    """Serial operating characteristics with explicit Python reporting conventions.

    Selection includes every stopping path; denominators include no-selection
    trials. Enrollment intervals are type-7 empirical 2.5/97.5 percentiles, not
    confidence intervals. MCSE is sample SD / sqrt(trials); one trial gives NaN.
    """
    trials = _integer(trials, "trials", 1)
    seed = _nonnegative_seed(seed)
    if trials > 10000 or trials * study.max_candidates * len(study.rules.arms) > 2_000_000:
        raise ValueError("simulation exceeds bounded trial/candidate work")
    sequence = np.random.SeedSequence(seed)
    seeds = tuple(
        int(child.generate_state(1, dtype=np.uint64)[0]) for child in sequence.spawn(trials)
    )
    k = len(study.rules.arms)
    counts = np.zeros((trials, k))
    durations = np.zeros(trials)
    selected = [0] * k
    selected_early = [0] * k
    dropped_early = [0] * k
    status_counts = [[0] * 10 for _ in range(k)]
    no_selection = completed = early = 0
    for i, trial_seed in enumerate(seeds):
        trial = simulate_arand_native_trial(study, seed=trial_seed)
        for record in trial.records:
            counts[i, record.arm] += 1
        final = trial.final
        durations[i] = final.time
        completed += int(final.completed)
        for arm, status in enumerate(final.state.statuses):
            status_counts[arm][status] += 1
            # Count terminal posterior/futility closures, including permanent
            # closure (3). Temporary closures that reopened are not final drops.
            dropped_early[arm] += int(status in (1, 2, 3))
        assert final.selection is not None
        early += int(final.selection.early)
        if final.selection.winner is None:
            no_selection += 1
        else:
            selected[final.selection.winner] += 1
            selected_early[final.selection.winner] += int(final.selection.early)
    return ArandNativeSummary(
        arand_native_study_json(study),
        seeds,
        tuple(selected),
        no_selection,
        completed,
        early,
        _freeze(counts.mean(axis=0)),
        _freeze(counts.std(axis=0, ddof=1) / np.sqrt(trials) if trials > 1 else np.full(k, np.nan)),
        _freeze(np.quantile(counts, [0.025, 0.975], axis=0, method="linear").T),
        float(durations.mean()),
        float(durations.std(ddof=1) / np.sqrt(trials)) if trials > 1 else float("nan"),
        tuple(selected_early),
        tuple(dropped_early),
        tuple(tuple(row) for row in status_counts),
        _legacy_enrollment_percentiles(counts),
    )


def arand_native_report(summary: ArandNativeSummary, *, percentile_profile: str = "type7") -> str:
    """Self-contained HTML summary with captured versioned inputs and replay seeds."""
    if percentile_profile not in ("type7", "legacy-5.2.2"):
        raise ValueError("percentile_profile must be type7 or legacy-5.2.2")
    legacy = percentile_profile == "legacy-5.2.2"
    quantiles = summary.legacy_enrollment_quantiles if legacy else summary.enrollment_quantiles
    convention = (
        "Legacy 5.2.2 percentiles use the original zero-based ceil(n*p) indices; "
        "they are omitted below 100 trials."
        if legacy
        else "Percentiles use type 7; they are not confidence intervals."
    )
    n = len(summary.trial_seeds)
    rows = []
    for i, selected in enumerate(summary.selected):
        probability = selected / n
        _unit(probability, "selection probability")
        mcse = np.sqrt(probability * (1 - probability) / (n - 1)) if n > 1 else float("nan")
        interval = (
            "omitted below 100 trials"
            if quantiles is None
            else f"{quantiles[i, 0]:.6g}, {quantiles[i, 1]:.6g}"
        )
        rows.append(
            f"<tr><td>{i}</td><td>{selected}/{n}</td><td>{probability:.6g}</td>"
            f"<td>{mcse:.6g}</td><td>{summary.selected_early[i] / n:.6g}</td>"
            f"<td>{summary.dropped_early[i] / n:.6g}</td><td>{summary.mean_enrollment[i]:.6g}</td>"
            f"<td>{summary.enrollment_mcse[i]:.6g}</td><td>{interval}</td></tr>"
        )
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>ARAND native profile</title>"
        "<h1>ARAND native controller — Python simulation</h1>"
        "<p>Independent NumPy random streams; Python report conventions. No native RNG, "
        "XML or report-format identity and no clinical validation are asserted.</p>"
        f"<p>Trials: {n}; completed by cap: {summary.completed}; "
        f"no selection: {summary.no_selection}; "
        f"early selection: {summary.early_selection}. Final time mean: {summary.duration_mean:.6g} "
        f"months; MCSE: {summary.duration_mcse:.6g}.</p>"
        "<table><tr><th>Arm (zero-based)</th><th>Selected</th><th>Probability</th><th>MCSE</th>"
        "<th>Early selection probability</th><th>Early drop probability</th>"
        "<th>Mean enrollment</th><th>MCSE</th><th>Enrollment 2.5/97.5 percentiles</th></tr>"
        + "".join(rows)
        + f"</table><p>{convention} "
        "Early drops count final temporary, other-winner, or permanent closures (statuses 1–3); "
        "a reopened arm is not counted as a final drop. Full final status counts are retained "
        "in the summary. NaN MCSE means fewer than two trials.</p><h2>Saved inputs</h2><pre>"
        + escape(summary.study_json)
        + "</pre><h2>Trial replay seeds</h2><pre>"
        + escape(json.dumps(summary.trial_seeds))
        + "</pre></html>"
    )
