"""Bounded current UAROET simulation jobs, independent streams and compatible merges."""

import csv
import hashlib
import io
import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .uaroet import _integer, _real_matrix
from .uaroet_desktop_simulation import UAROETCalendarTrial, run_uaroet_desktop_trial
from .uaroet_desktop_study import (
    UAROETDesktopScenario,
    UAROETDesktopStudy,
    read_uaroet_desktop_study,
)


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _metrics(D: int, E: int, T: int) -> tuple[str, ...]:
    return (
        "no_selection",
        *(f"select_d{d + 1}" for d in range(D)),
        "early_stop",
        "enrollment",
        "followup_duration",
        "decision_time",
        "discarded_arrivals",
        *(f"assigned_d{d + 1}" for d in range(D)),
        *(f"outcome_d{d + 1}_e{e}_t{t}" for d in range(D) for e in range(E) for t in range(T)),
    )


@dataclass(frozen=True)
class UAROETDesktopBatch:
    """Mergeable sums and squared sums; uncertainty measures Monte Carlo error only.

    Stream segments identify(seed, first_trial, count). Each trial uses
    SeedSequence([seed, index]) with explicit PCG64. Merges reject overlapping
    segments and differing exact configurations. This guards accidental stream
    reuse, not cryptographic proof of independently produced simulation jobs.
    """

    configuration: str
    segments: tuple[tuple[int, int, int], ...]
    metrics: tuple[str, ...]
    totals: tuple[float, ...]
    squared_totals: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.configuration, str):
            raise ValueError("configuration must be canonical JSON")
        try:
            config = json.loads(self.configuration)
            if not isinstance(config, dict) or set(config) != {
                "study",
                "grid",
                "scenario",
                "options",
                "numpy_version",
                "generator",
                "stream_contract",
            }:
                raise ValueError("unsupported configuration fields")
            design = read_uaroet_desktop_study(_canonical(config["study"]))
            if design.patients or config["grid"] != [
                design.dose_count,
                design.efficacy_levels,
                design.toxicity_levels,
            ]:
                raise ValueError("configuration grid must match an empty study")
            scenario = UAROETDesktopScenario(**config["scenario"])
            if scenario.efficacy.shape != (design.dose_count, design.efficacy_levels) or (
                scenario.toxicity.shape != (design.dose_count, design.toxicity_levels)
            ):
                raise ValueError("configuration scenario must match study")
            expected = _metrics(*config["grid"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError("invalid batch configuration") from exc
        if self.configuration != _canonical(config) or tuple(self.metrics) != expected:
            raise ValueError("configuration or metric layout is not canonical")
        if not self.segments or len(self.segments) > 100_000:
            raise ValueError("require bounded nonempty stream segments")
        segments = []
        for segment in self.segments:
            if len(segment) != 3:
                raise ValueError("stream segments require seed, first_trial, count")
            seed = _integer(segment[0], "seed", 0, 2**63 - 1)
            first = _integer(segment[1], "first_trial", 0, 10**9)
            n = _integer(segment[2], "trial count", 1, 100_000)
            segments.append((seed, first, n))
        segments.sort()
        for a, b in zip(segments, segments[1:]):
            if a[0] == b[0] and a[1] + a[2] > b[1]:
                raise ValueError("overlapping trial streams cannot be combined")
        n = sum(s[2] for s in segments)
        total = _real_matrix(self.totals, "totals")
        squares = _real_matrix(self.squared_totals, "squared_totals")
        if total.shape != (len(expected),) or squares.shape != total.shape:
            raise ValueError("batch moments must match metric layout")
        if (
            np.any(total < 0)
            or np.any(squares < 0)
            or np.any(squares * n + 1e-8 * np.maximum(1, total**2) < total**2)
        ):
            raise ValueError("invalid nonnegative batch moments")
        D = config["grid"][0]
        if not np.isclose(sum(total[: D + 1]), n, rtol=0, atol=1e-8):
            raise ValueError("selection totals must cover every trial")
        if not np.array_equal(total[: D + 2], squares[: D + 2]):
            raise ValueError("binary metric sums and squared sums must match")
        enrollment = total[D + 2]
        assigned = total[D + 6 : 2 * D + 6]
        outcomes = total[2 * D + 6 :].reshape(config["grid"])
        if (
            total[D + 1] > n
            or not n <= enrollment <= n * design.trial_parameters["NumPatients"]
            or not np.isclose(assigned.sum(), enrollment, rtol=0, atol=1e-8)
            or not np.allclose(outcomes.sum((1, 2)), assigned, rtol=0, atol=1e-8)
        ):
            raise ValueError("enrollment and outcome counts are inconsistent")
        count_indices = list(range(D + 3)) + list(range(D + 5, len(total)))
        if np.any(total[count_indices] != np.floor(total[count_indices])):
            raise ValueError("count metrics must be integral")
        object.__setattr__(self, "segments", tuple(segments))
        object.__setattr__(self, "metrics", expected)
        object.__setattr__(self, "totals", tuple(float(x) for x in total))
        object.__setattr__(self, "squared_totals", tuple(float(x) for x in squares))

    @property
    def trials(self) -> int:
        return sum(segment[2] for segment in self.segments)

    @property
    def configuration_sha256(self) -> str:
        return hashlib.sha256(self.configuration.encode()).hexdigest()

    def summary(self) -> dict[str, dict[str, float | None]]:
        n = self.trials
        result = {}
        for key, total, square in zip(self.metrics, self.totals, self.squared_totals, strict=True):
            variance = max(0.0, (square - total * total / n) / (n - 1)) if n > 1 else None
            result[key] = {
                "mean": total / n,
                "mcse": None if variance is None else float(np.sqrt(variance / n)),
            }
        return result

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    "schema": "mdanderson-stats.uaroet-batch",
                    "version": 1,
                    "configuration": json.loads(self.configuration),
                    "configuration_sha256": self.configuration_sha256,
                    "segments": self.segments,
                    "metrics": self.metrics,
                    "totals": self.totals,
                    "squared_totals": self.squared_totals,
                    "clinical_validation": False,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )

    def to_csv(self) -> str:
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(["metric", "mean", "mcse", "trials", "configuration_sha256"])
        for key, row in self.summary().items():
            writer.writerow([key, row["mean"], row["mcse"], self.trials, self.configuration_sha256])
        return out.getvalue()


def read_uaroet_desktop_batch(text: str) -> UAROETDesktopBatch:
    """Read the portable job schema, validating layout, moments and stream overlap."""
    if not isinstance(text, str) or len(text) > 16_000_000:
        raise ValueError("require a bounded JSON batch string")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=unique)
    keys = {
        "schema",
        "version",
        "configuration",
        "configuration_sha256",
        "segments",
        "metrics",
        "totals",
        "squared_totals",
        "clinical_validation",
    }
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or (
            value["schema"] != "mdanderson-stats.uaroet-batch"
            or type(value["version"]) is not int
            or value["version"] != 1
            or value["clinical_validation"] is not False
        )
    ):
        raise ValueError("unsupported batch schema")
    result = UAROETDesktopBatch(
        _canonical(value["configuration"]),
        tuple(tuple(x) for x in value["segments"]),
        tuple(value["metrics"]),
        tuple(value["totals"]),
        tuple(value["squared_totals"]),
    )
    if result.configuration_sha256 != value["configuration_sha256"]:
        raise ValueError("configuration fingerprint mismatch")
    return result


def merge_uaroet_desktop_batches(*batches: UAROETDesktopBatch) -> UAROETDesktopBatch:
    if not batches or not all(isinstance(b, UAROETDesktopBatch) for b in batches):
        raise ValueError("require at least one UAROETDesktopBatch")
    first = batches[0]
    if any(b.configuration != first.configuration for b in batches):
        raise ValueError("only identical study, scenario and computational settings can be merged")
    return UAROETDesktopBatch(
        first.configuration,
        tuple(s for b in batches for s in b.segments),
        first.metrics,
        tuple(np.sum([b.totals for b in batches], axis=0)),
        tuple(np.sum([b.squared_totals for b in batches], axis=0)),
    )


def simulate_uaroet_desktop(
    study: UAROETDesktopStudy,
    scenario: UAROETDesktopScenario,
    *,
    repetitions: int,
    seed: int,
    first_trial: int = 0,
    profile: str = "safe",
    target: float | None = None,
    chunk_draws: int = 1000,
    max_draws: int = 100_000,
    max_evaluations: int = 2_000_000,
    maximum_rhat: float = 1.05,
    max_discarded_arrivals: int = 100_000,
    on_trial: Callable[[int, UAROETCalendarTrial], None] | None = None,
) -> UAROETDesktopBatch:
    """Run an independently indexed job with online moments and optional trial export.

    Failure returns no completed job. Callback output from an interrupted job is
    diagnostic only. Retention is bounded by one trial, not the replication count.
    Splitting a seed's disjoint index range preserves its per-trial random streams.
    """
    n = _integer(repetitions, "repetitions", 1, 100_000)
    seed = _integer(seed, "seed", 0, 2**63 - 1)
    first = _integer(first_trial, "first_trial", 0, 10**9)
    if not isinstance(study, UAROETDesktopStudy) or not isinstance(scenario, UAROETDesktopScenario):
        raise ValueError("require a desktop study and scenario")
    if on_trial is not None and not callable(on_trial):
        raise ValueError("on_trial must be callable")
    options = dict(
        profile=profile,
        target=target,
        chunk_draws=chunk_draws,
        max_draws=max_draws,
        max_evaluations=max_evaluations,
        maximum_rhat=maximum_rhat,
        max_discarded_arrivals=max_discarded_arrivals,
    )
    D, E, T = study.dose_count, study.efficacy_levels, study.toxicity_levels
    metrics = _metrics(D, E, T)
    configuration = _canonical(
        {
            "study": json.loads(study.to_json()),
            "grid": [D, E, T],
            "scenario": {
                "identifier": scenario.identifier,
                "efficacy": scenario.efficacy.tolist(),
                "toxicity": scenario.toxicity.tolist(),
                "association": scenario.association,
            },
            "options": options,
            "numpy_version": np.__version__,
            "generator": "PCG64",
            "stream_contract": "SeedSequence([seed, zero_based_trial_index])",
        }
    )
    totals = np.zeros(len(metrics))
    squares = np.zeros(len(metrics))
    for index in range(first, first + n):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, index])))
        trial = run_uaroet_desktop_trial(
            study,
            scenario,
            rng=rng,
            profile=profile,
            target=target,
            chunk_draws=chunk_draws,
            max_draws=max_draws,
            max_evaluations=max_evaluations,
            maximum_rhat=maximum_rhat,
            max_discarded_arrivals=max_discarded_arrivals,
        )
        selected = np.zeros(D + 1)
        selected[0 if trial.selected is None else trial.selected + 1] = 1
        assigned = np.zeros(D)
        outcomes = np.zeros((D, E, T))
        for patient in trial.patients:
            assigned[patient.dose] += 1
            outcomes[patient.dose, patient.efficacy, patient.toxicity] += 1
        row = np.concatenate(
            (
                selected,
                [
                    trial.stopped_early,
                    len(trial.patients),
                    trial.followup_duration,
                    trial.decision_time,
                    trial.discarded_arrivals,
                ],
                assigned,
                outcomes.ravel(),
            )
        )
        totals += row
        squares += row * row
        if on_trial is not None:
            on_trial(index, trial)
    return UAROETDesktopBatch(
        configuration, ((seed, first, n),), metrics, tuple(totals), tuple(squares)
    )
