"""Named, reproducible scenario batches for the four-arm Phase I/II design."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .parallel_phase12_oc import ParallelPhase12OC, simulate_parallel_phase12_oc

_MAX_SCENARIOS = 20
_MAX_PATIENT_WORK = 1_000_000
_MAX_TRIALS_PER_SCENARIO = 10_000


def _array_text(value: ArrayLike) -> str:
    return np.array2string(
        np.asarray(value),
        precision=17,
        floatmode="unique",
        formatter={"float_kind": lambda item: f"{item:.17g}"},
    )


def _probability_vector(value: ArrayLike, name: str) -> tuple[float, float, float, float]:
    if isinstance(value, (list, tuple)):
        if len(value) != 4 or any(np.ndim(item) != 0 for item in value):
            raise ValueError(f"{name} must be a real four-vector")
    elif not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a real four-vector")
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.shape != (4,) or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a real four-vector")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real four-vector") from exc
    if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} entries must be finite probabilities in [0,1]")
    return tuple(float(item) for item in array)  # type: ignore[return-value]


def _integer(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return result


@dataclass(frozen=True)
class ParallelPhase12Scenario:
    """One named truth scenario and a reproducible serial OC request."""

    label: str
    toxicity_probability: ArrayLike
    efficacy_probability: ArrayLike
    n_trials: int
    seed: int
    optimal_arms: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip() or len(self.label) > 120:
            raise ValueError("label must be a nonempty string of at most 120 characters")
        object.__setattr__(
            self,
            "toxicity_probability",
            _probability_vector(self.toxicity_probability, "toxicity_probability"),
        )
        object.__setattr__(
            self,
            "efficacy_probability",
            _probability_vector(self.efficacy_probability, "efficacy_probability"),
        )
        object.__setattr__(
            self, "n_trials", _integer(self.n_trials, "n_trials", 1, _MAX_TRIALS_PER_SCENARIO)
        )
        object.__setattr__(self, "seed", _integer(self.seed, "seed", 0, 2**63 - 1))
        if self.optimal_arms is not None:
            if not isinstance(self.optimal_arms, (tuple, list, np.ndarray)):
                raise ValueError("optimal_arms must contain unique integer indices 0..3")
            if isinstance(self.optimal_arms, np.ndarray):
                if self.optimal_arms.ndim != 1 or self.optimal_arms.shape[0] > 4:
                    raise ValueError("optimal_arms must contain unique integer indices 0..3")
            else:
                if len(self.optimal_arms) > 4 or any(
                    isinstance(item, (tuple, list))
                    or (isinstance(item, np.ndarray) and item.ndim != 0)
                    for item in self.optimal_arms
                ):
                    raise ValueError("optimal_arms must contain unique integer indices 0..3")
            raw = np.asarray(self.optimal_arms)
            if (
                raw.ndim != 1
                or raw.size == 0
                or raw.dtype.kind not in "iu"
                or raw.dtype.kind == "b"
                or np.any((raw < 0) | (raw > 3))
                or np.unique(raw).size != raw.size
            ):
                raise ValueError("optimal_arms must contain unique integer indices 0..3")
            object.__setattr__(self, "optimal_arms", tuple(sorted(int(x) for x in raw)))


@dataclass(frozen=True)
class ParallelPhase12ScenarioSummary:
    """Captured scenario inputs paired with the OC computed from those inputs."""

    label: str
    toxicity_probability: tuple[float, float, float, float]
    efficacy_probability: tuple[float, float, float, float]
    n_trials: int
    seed: int
    optimal_arms: tuple[int, ...] | None
    operating_characteristics: ParallelPhase12OC


@dataclass(frozen=True)
class ParallelPhase12ScenarioBatch:
    """Ordered scenario summaries with a bounded text report and atomic save."""

    scenarios: tuple[ParallelPhase12ScenarioSummary, ...]

    def report(self) -> str:
        lines = [
            "Parallel Phase I/II four-arm scenario report",
            "Python implementation of the archived four-arm C workflow; zero-based arm order.",
            "Fixed design: N<=100; phase-II looks every 5 patients; toxicity Beta(1,9);",
            "efficacy Beta(0.1,1.9); source stopping/allocation rules as documented.",
            "",
        ]
        for item in self.scenarios:
            oc = item.operating_characteristics
            lines.extend(
                (
                    f"Scenario: {item.label}",
                    f"  seed={item.seed}; trials={item.n_trials}; optimal_arms={item.optimal_arms}",
                    f"  toxicity_truth={item.toxicity_probability!r}",
                    f"  efficacy_truth={item.efficacy_probability!r}",
                    f"  selection_probability={_array_text(oc.selection_probability)}",
                    f"  selection_mcse={_array_text(oc.selection_mcse)}",
                    f"  no_selection={oc.no_selection_probability:.17g} "
                    f"(MCSE {oc.no_selection_mcse:.17g})",
                    "  stopping_probabilities="
                    + ", ".join(
                        f"{name}={prob:.17g} (MCSE {error:.17g})"
                        for name, prob, error in zip(
                            oc.stopping_reasons,
                            oc.stopping_probability,
                            oc.stopping_mcse,
                            strict=True,
                        )
                    ),
                    f"  mean_total_enrollment={oc.mean_total_enrollment:.17g} "
                    f"(MCSE {oc.mcse_total_enrollment:.17g})",
                    f"  mean_phase_one_enrollment={oc.mean_phase_one_enrollment:.17g} "
                    f"(MCSE {oc.mcse_phase_one_enrollment:.17g})",
                    f"  mean_treated={_array_text(oc.mean_treated)}",
                    f"  mean_treated_mcse={_array_text(oc.mcse_treated)}",
                    f"  mean_toxicities={_array_text(oc.mean_toxicities)}",
                    f"  mean_toxicities_mcse={_array_text(oc.mcse_toxicities)}",
                    f"  mean_responses={_array_text(oc.mean_responses)}",
                    f"  mean_responses_mcse={_array_text(oc.mcse_responses)}",
                    f"  toxicity_rate={_array_text(oc.toxicity_rate)}",
                    f"  toxicity_rate_mcse={_array_text(oc.toxicity_rate_mcse)}",
                    f"  response_rate={_array_text(oc.response_rate)}",
                    f"  response_rate_mcse={_array_text(oc.response_rate_mcse)}",
                    f"  admissibility_probability={_array_text(oc.admissibility_probability)}",
                    f"  admissibility_mcse={_array_text(oc.admissibility_mcse)}",
                    f"  optimal_selection_probability={oc.optimal_selection_probability!r}",
                    f"  optimal_selection_mcse={oc.optimal_selection_mcse!r}",
                    "",
                )
            )
        return "\n".join(lines).rstrip() + "\n"

    def write_report(self, path: str | os.PathLike[str]) -> Path:
        """Atomically write the rendered report without replacing it on failure."""
        destination = Path(path)
        parent = destination.parent
        if not parent.is_dir():
            raise ValueError("report parent directory must already exist")
        text = self.report()
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=parent, delete=False
            ) as stream:
                temporary = stream.name
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except OSError:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
            raise
        return destination


def parallel_phase12_scenario_from_native_input(
    text: str,
    *,
    label: str,
    n_trials: int,
    seed: int,
    optimal_arms: tuple[int, ...] | None = None,
) -> ParallelPhase12Scenario:
    """Parse the archived C parameter-file order: four efficacy then four toxicity rates.

    The native file contains no trial settings, replication count, or seed. Those
    remain explicit Python arguments and are not inferred from the file.
    """
    if not isinstance(text, str):
        raise ValueError("native parameter input must be text")
    if len(text) > 8_192:
        raise ValueError("native parameter input exceeds the 8192-character limit")
    tokens = text.split()
    if len(tokens) != 8:
        raise ValueError("native parameter input must contain exactly eight probabilities")
    try:
        values = tuple(float(token) for token in tokens)
    except (ValueError, OverflowError) as exc:
        raise ValueError("native parameter input must contain eight numeric probabilities") from exc
    efficacy = values[:4]
    toxicity = values[4:]
    return ParallelPhase12Scenario(
        label=label,
        toxicity_probability=toxicity,
        efficacy_probability=efficacy,
        n_trials=n_trials,
        seed=seed,
        optimal_arms=optimal_arms,
    )


def simulate_parallel_phase12_scenarios(
    scenarios: Sequence[ParallelPhase12Scenario],
) -> ParallelPhase12ScenarioBatch:
    """Run named truth scenarios serially, enforcing the aggregate work cap first."""
    if not isinstance(scenarios, Sequence) or isinstance(scenarios, (str, bytes)):
        raise ValueError("scenarios must be a bounded sequence of ParallelPhase12Scenario values")
    if not 1 <= len(scenarios) <= _MAX_SCENARIOS:
        raise ValueError(f"scenarios must contain 1..{_MAX_SCENARIOS} entries")
    if any(not isinstance(item, ParallelPhase12Scenario) for item in scenarios):
        raise ValueError("every scenario must be a ParallelPhase12Scenario")
    labels = [item.label for item in scenarios]
    if len(set(labels)) != len(labels):
        raise ValueError("scenario labels must be unique")
    patient_work = sum(item.n_trials * 100 for item in scenarios)
    if patient_work > _MAX_PATIENT_WORK:
        raise ValueError("aggregate scenario workload exceeds the bounded patient-work limit")

    results: list[ParallelPhase12ScenarioSummary] = []
    for item in scenarios:
        toxicity = _probability_vector(item.toxicity_probability, "toxicity_probability")
        efficacy = _probability_vector(item.efficacy_probability, "efficacy_probability")
        oc = simulate_parallel_phase12_oc(
            toxicity,
            efficacy,
            n_trials=item.n_trials,
            seed=item.seed,
            optimal_arms=item.optimal_arms,
        )
        results.append(
            ParallelPhase12ScenarioSummary(
                label=item.label,
                toxicity_probability=toxicity,
                efficacy_probability=efficacy,
                n_trials=item.n_trials,
                seed=item.seed,
                optimal_arms=item.optimal_arms,
                operating_characteristics=oc,
            )
        )
    return ParallelPhase12ScenarioBatch(tuple(results))
