"""Saved-input and scenario-report workflow for Multc Lean Python designs."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike

from ._validation import finite, scalar
from .multc_core import (
    MultcLeanDesign,
    MultcOperatingCharacteristics,
    _Historical,
    multc_lean_design,
)
from .multc_simulation import (
    MultcSimulationConfig,
    MultcSimulationResult,
    simulate_multc,
)


def _json_value(value: object) -> object:
    if is_dataclass(value):
        return asdict(cast(Any, value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} in a Multc study")


def _scenario_shape(value: ArrayLike) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)) and (
        not 1 <= len(value) <= 100
        or any(not isinstance(row, (list, tuple, np.ndarray)) or len(row) != 4 for row in value)
    ):
        raise ValueError("scenario_probabilities must have shape (1..100, 4)")
    shape = np.shape(value)
    if len(shape) != 2 or not 1 <= shape[0] <= 100 or shape[1] != 4:
        raise ValueError("scenario_probabilities must have shape (1..100, 4)")
    return shape


def _validate_scenario_names(names: object, count: int) -> tuple[str, ...]:
    if names is None:
        return tuple(f"scenario {i + 1}" for i in range(count))
    if (
        not isinstance(names, (list, tuple))
        or len(names) != count
        or any(
            not isinstance(name, str)
            or not name.strip()
            or len(name) > 80
            or any(ord(character) < 32 or ord(character) == 127 for character in name)
            for name in names
        )
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            "scenario_names must be unique nonempty names of at most 80 printable chars"
        )
    return tuple(names)


def _atomic_write(path: str | Path, content: str) -> Path:
    destination = Path(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


@dataclass(frozen=True)
class MultcStudySpecification:
    """Complete Python inputs for one design and its ordered joint-truth scenarios.

    Historical references are fixed rates or beta-shape pairs. Scenario vectors
    use the existing Multc category order: both, response only, toxicity only,
    neither. These saved JSON fields and reports are Python conventions; they do
    not claim compatibility with Multc Lean's native files.
    """

    max_subjects: int
    response_prior: tuple[float, float]
    toxicity_prior: tuple[float, float]
    historical_response: float | tuple[float, float]
    historical_toxicity: float | tuple[float, float]
    scenario_probabilities: ArrayLike
    scenario_names: tuple[str, ...] | None = None
    simulation: MultcStudySimulationSettings | None = None
    max_total_exact_work: int = 50_000_000
    response_margin: float = 0.0
    toxicity_margin: float = 0.0
    response_cutoff: float = 0.95
    toxicity_cutoff: float = 0.95
    min_subjects: int = 1
    cohort_size: int = 1
    pretrial_check: bool = True
    format_version: int = 1

    def run(self) -> MultcStudy:
        if (
            isinstance(self.format_version, bool)
            or not isinstance(self.format_version, (int, np.integer))
            or self.format_version != 1
        ):
            raise ValueError("unsupported Multc study format_version")
        raw = self.scenario_probabilities
        raw_shape = _scenario_shape(raw)
        if len(raw_shape) != 2 or not 1 <= raw_shape[0] <= 100 or raw_shape[1] != 4:
            raise ValueError("scenario_probabilities must have shape (1..100, 4)")
        probabilities = finite(raw, "scenario_probabilities")
        if np.any(probabilities < 0) or not np.allclose(
            probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("each scenario row must be nonnegative and sum to one")
        if (
            isinstance(self.max_subjects, (bool, np.bool_))
            or not isinstance(self.max_subjects, (int, np.integer))
            or not 3 <= self.max_subjects <= 1000
        ):
            raise ValueError("max_subjects must be an integer in [3,1000]")
        if (
            isinstance(self.max_total_exact_work, (bool, np.bool_))
            or not isinstance(self.max_total_exact_work, (int, np.integer))
            or not 1 <= self.max_total_exact_work <= 2_000_000_000
        ):
            raise ValueError("max_total_exact_work must be an integer in [1,2000000000]")
        exact_work_per_scenario = sum(
            (sample_size + 1) ** 2 for sample_size in range(1, self.max_subjects + 1)
        )
        if exact_work_per_scenario > 10_000_000 or (
            exact_work_per_scenario * probabilities.shape[0] > self.max_total_exact_work
        ):
            raise ValueError("exact OC scenarios exceed the configured aggregate work limit")
        names = _validate_scenario_names(self.scenario_names, probabilities.shape[0])

        simulation_settings = self.simulation
        if isinstance(simulation_settings, dict):
            simulation_settings = MultcStudySimulationSettings(**simulation_settings)
        if simulation_settings is not None and not isinstance(
            simulation_settings, MultcStudySimulationSettings
        ):
            raise TypeError("simulation must be MultcStudySimulationSettings or None")
        if simulation_settings is not None:
            simulation_settings = _validate_simulation_settings(simulation_settings)
            per_trial_work = 10 * self.max_subjects**2 + 4 * self.max_subjects + 1000
            if (
                simulation_settings.trials * probabilities.shape[0] * per_trial_work
                > simulation_settings.max_total_work
            ):
                raise ValueError("simulation scenarios exceed max_total_work")
            retained_summary = (
                simulation_settings.trials * 112 + (self.max_subjects + 1) * 8
            ) * probabilities.shape[0]
            per_trial_storage = (
                self.max_subjects * 256 + (1000 + 2 * self.max_subjects) * 768 + 32_768
            )
            if retained_summary + per_trial_storage > simulation_settings.max_total_storage_bytes:
                raise ValueError("simulation scenarios exceed max_total_storage_bytes")

        def historical(value: Any) -> float | tuple[float, float]:
            if isinstance(value, list):
                return tuple(value)
            return value

        design = multc_lean_design(
            self.max_subjects,
            _pair(self.response_prior, "response_prior"),
            _pair(self.toxicity_prior, "toxicity_prior"),
            historical_response=historical(self.historical_response),
            historical_toxicity=historical(self.historical_toxicity),
            response_margin=self.response_margin,
            toxicity_margin=self.toxicity_margin,
            response_cutoff=self.response_cutoff,
            toxicity_cutoff=self.toxicity_cutoff,
            min_subjects=self.min_subjects,
            cohort_size=self.cohort_size,
            pretrial_check=self.pretrial_check,
        )
        # Keep an owned immutable snapshot of all user-provided settings and rows.
        snapshot = replace(
            self,
            max_subjects=design.max_subjects,
            response_prior=(float(design.response_prior.alpha), float(design.response_prior.beta)),
            toxicity_prior=(float(design.toxicity_prior.alpha), float(design.toxicity_prior.beta)),
            historical_response=_historical_setting(design.historical_response),
            historical_toxicity=_historical_setting(design.historical_toxicity),
            response_margin=design.response_margin,
            toxicity_margin=design.toxicity_margin,
            response_cutoff=design.response_cutoff,
            toxicity_cutoff=design.toxicity_cutoff,
            min_subjects=design.min_subjects,
            cohort_size=design.cohort_size,
            pretrial_check=design.pretrial_check,
            scenario_probabilities=tuple(tuple(map(float, row)) for row in probabilities),
            scenario_names=names,
            simulation=simulation_settings,
        )
        outcomes = tuple(design.operating_characteristics(row) for row in probabilities)
        simulations: tuple[MultcSimulationResult, ...] = ()
        simulation_seeds: tuple[int, ...] = ()
        if simulation_settings is not None:
            sim_config = MultcSimulationConfig(
                design,
                simulation_settings.response_window,
                simulation_settings.toxicity_delay,
                simulation_settings.response_timing,
            )
            child_seeds = np.random.SeedSequence(simulation_settings.seed).spawn(len(names))
            simulation_seeds = tuple(
                int(child.generate_state(1, dtype=np.uint64)[0]) for child in child_seeds
            )
            simulations = tuple(
                simulate_multc(
                    sim_config,
                    row,
                    accrual_rate=simulation_settings.accrual_rate,
                    trials=simulation_settings.trials,
                    seed=trial_seed,
                    max_total_work=simulation_settings.max_total_work,
                    max_total_storage_bytes=simulation_settings.max_total_storage_bytes,
                )
                for row, trial_seed in zip(probabilities, simulation_seeds, strict=True)
            )
        return MultcStudy(snapshot, design, outcomes, simulations, simulation_seeds)

    def to_json(self) -> str:
        shape = _scenario_shape(self.scenario_probabilities)
        _validate_scenario_names(self.scenario_names, shape[0])
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        result = json.dumps(values, default=_json_value, indent=2, allow_nan=False) + "\n"
        if len(result.encode("utf-8")) > 1_048_576:
            raise ValueError("Multc study JSON exceeds the 1 MiB input limit")
        return result

    @classmethod
    def from_json(cls, text: str) -> MultcStudySpecification:
        if not isinstance(text, str) or len(text) > 1_048_576:
            raise ValueError("Multc study JSON must be text no larger than 1 MiB")
        if len(text.encode("utf-8")) > 1_048_576:
            raise ValueError("Multc study JSON must be text no larger than 1 MiB")

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate Multc study JSON key: {key}")
                result[key] = value
            return result

        def reject_constant(value: str) -> None:
            raise ValueError(f"invalid non-finite JSON number: {value}")

        values = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
        if not isinstance(values, dict):
            raise ValueError("Multc study specification must be a JSON object")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown Multc study fields: {', '.join(sorted(unknown))}")
        if isinstance(values.get("simulation"), dict):
            values["simulation"] = MultcStudySimulationSettings(**values["simulation"])
        return cls(**values)


@dataclass(frozen=True)
class MultcStudySimulationSettings:
    """Optional Python duration-simulation inputs using explicit timing choices."""

    response_window: float
    toxicity_delay: float
    response_timing: Literal["conditional_truncated_exponential", "clip_at_window"]
    accrual_rate: float
    trials: int
    seed: int
    max_total_work: int = 500_000_000
    max_total_storage_bytes: int = 512_000_000


def _pair(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain two beta shapes")
    return float(value[0]), float(value[1])


def _historical_setting(value: _Historical) -> float | tuple[float, float]:
    if value.constant is not None:
        return float(value.constant)
    assert value.prior is not None
    prior = value.prior
    return float(prior.alpha), float(prior.beta)


def _validate_simulation_settings(
    settings: MultcStudySimulationSettings,
) -> MultcStudySimulationSettings:
    if (
        isinstance(settings.trials, (bool, np.bool_))
        or not isinstance(settings.trials, (int, np.integer))
        or not 1 <= settings.trials <= 10_000
    ):
        raise ValueError("simulation trials must be an integer in [1,10000]")
    if (
        isinstance(settings.seed, (bool, np.bool_))
        or not isinstance(settings.seed, (int, np.integer))
        or settings.seed < 0
    ):
        raise ValueError("simulation seed must be a nonnegative integer")
    for value, name, maximum in (
        (settings.max_total_work, "max_total_work", 2_000_000_000),
        (settings.max_total_storage_bytes, "max_total_storage_bytes", 2_000_000_000),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not 1 <= value <= maximum
        ):
            raise ValueError(f"{name} must be an integer in [1,{maximum}]")
    window = scalar(settings.response_window, "response_window")
    delay = scalar(settings.toxicity_delay, "toxicity_delay")
    rate = scalar(settings.accrual_rate, "accrual_rate")
    if window <= 0 or delay < 0 or rate <= 0 or not np.isfinite(1.0 / rate):
        raise ValueError(
            "require positive response window/accrual rate and nonnegative toxicity delay"
        )
    if settings.response_timing not in (
        "conditional_truncated_exponential",
        "clip_at_window",
    ):
        raise ValueError("response_timing must select an explicit truncation convention")
    return replace(
        settings,
        response_window=window,
        toxicity_delay=delay,
        accrual_rate=rate,
        trials=int(settings.trials),
        seed=int(settings.seed),
        max_total_work=int(settings.max_total_work),
        max_total_storage_bytes=int(settings.max_total_storage_bytes),
    )


@dataclass(frozen=True)
class MultcStudy:
    specification: MultcStudySpecification
    design: MultcLeanDesign
    operating_characteristics: tuple[MultcOperatingCharacteristics, ...]
    simulations: tuple[MultcSimulationResult, ...] = ()
    simulation_seeds: tuple[int, ...] = ()

    def report(self, *, digits: int = 8) -> str:
        if (
            isinstance(digits, (bool, np.bool_))
            or not isinstance(digits, (int, np.integer))
            or not 1 <= digits <= 17
        ):
            raise ValueError("digits must be an integer from 1 to 17")

        def number(value: Any) -> str:
            return f"{float(cast(Any, value)):.{digits}g}"

        s, d = self.specification, self.design
        rows = ["Multc Lean Python study", "Setting\tJSON value"]
        for name, value in asdict(s).items():
            rows.append(name + "\t" + json.dumps(value, default=_json_value, allow_nan=False))
        rows.extend(
            [
                "",
                "Native-rule mapping",
                "Response: stop when Pr(historical + margin > experimental) exceeds "
                "response_cutoff.",
                "Toxicity: stop when Pr(historical + margin < experimental) exceeds "
                "toxicity_cutoff.",
                "Paired outcome category order: both, response only, toxicity only, neither.",
                "Report and JSON layout are Python conventions, not native Multc Lean files.",
                "",
                "Full stopping boundaries (inclusive event counts)",
                "N\tResponse stop at or below\tToxicity stop at or above",
            ]
        )
        bounds = d.stopping_bounds()
        for n, r, t in zip(
            bounds.looks, bounds.response_stop_max, bounds.toxicity_stop_min, strict=True
        ):
            rows.append(f"{int(n)}\t{int(r)}\t{int(t)}")
        potential = d.potential_boundaries()
        rows.extend(["", "Reachable stopping intervals", "N\tResponse\tToxicity"])
        for n, r, t in zip(
            bounds.looks, potential.response_stop, potential.toxicity_stop, strict=True
        ):
            rows.append(f"{int(n)}\t{_interval(r)}\t{_interval(t)}")
        rows.extend(
            [
                "",
                "Exact joint operating characteristics",
                "Scenario\tP(both,response-only,toxicity-only,neither)\tStop response only"
                "\tStop toxicity only\tStop both\tCap completion\tExpected N\tSD N"
                "\tExpected responses\tExpected toxicities",
            ]
        )
        for name, oc in zip(s.scenario_names or (), self.operating_characteristics, strict=True):
            probabilities = ",".join(number(v) for v in oc.scenario_probability)
            rows.append(
                "\t".join(
                    [
                        name,
                        probabilities,
                        number(oc.stop_response_only),
                        number(oc.stop_toxicity_only),
                        number(oc.stop_both),
                        number(oc.cap_completion),
                        number(oc.expected_sample_size),
                        number(oc.sample_size_sd),
                        number(oc.expected_responses),
                        number(oc.expected_toxicities),
                    ]
                )
            )
        if self.simulations:
            rows.extend(
                [
                    "",
                    "Calendar simulation (Monte Carlo estimates; separate from exact OCs)",
                    "Scenario\tSeed\tTrials\tMean enrolled\tMCSE enrolled\tMean responses"
                    "\tMCSE responses\tMean toxicities\tMCSE toxicities\tMean duration"
                    "\tMCSE duration\tDuration 95% empirical interval\tMean accrual duration"
                    "\tMean paused duration\tMean last follow-up\tDecision counts"
                    "\tDecision probabilities\tDecision MCSEs",
                ]
            )
            for name, seed, result in zip(
                s.scenario_names or (), self.simulation_seeds, self.simulations, strict=True
            ):
                rows.append(
                    "\t".join(
                        [
                            name,
                            str(seed),
                            str(result.trials),
                            number(result.mean_enrolled),
                            number(result.enrolled_mcse),
                            number(result.mean_responses),
                            number(result.responses_mcse),
                            number(result.mean_toxicities),
                            number(result.toxicities_mcse),
                            number(result.mean_duration),
                            number(result.duration_mcse),
                            ",".join(number(v) for v in result.duration_interval),
                            number(result.mean_accrual_duration),
                            number(result.mean_paused_duration),
                            number(result.mean_last_followup_time),
                            ",".join(
                                f"{decision}:{int(count)}"
                                for decision, count in zip(
                                    result.decisions, result.decision_count, strict=True
                                )
                            ),
                            ",".join(
                                f"{decision}:{number(probability)}"
                                for decision, probability in zip(
                                    result.decisions, result.decision_probability, strict=True
                                )
                            ),
                            ",".join(
                                f"{decision}:{number(error)}"
                                for decision, error in zip(
                                    result.decisions, result.decision_mcse, strict=True
                                )
                            ),
                        ]
                    )
                )
        return "\n".join(rows) + "\n"

    def write_report(self, path: str | Path, *, digits: int = 8) -> Path:
        return _atomic_write(path, self.report(digits=digits))

    def write_specification(self, path: str | Path) -> Path:
        return _atomic_write(path, self.specification.to_json())


def _interval(value: tuple[int, int]) -> str:
    return "empty" if value[0] > value[1] else f"{value[0]}..{value[1]}"
