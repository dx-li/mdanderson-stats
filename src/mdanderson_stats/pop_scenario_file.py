"""Versioned, explicit JSON input bundles for reproducible PoP reports."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._validation import finite, scalar
from .pop_design import PoPDesign
from .pop_protocol_report import PoPProtocolReport, PoPReportScenario, run_pop_protocol

_SCHEMA = "mdanderson-stats/pop-scenario-input/v1"
_MAX_JSON_CHARS = 1_000_000


@dataclass(frozen=True, slots=True)
class PoPInputScenario:
    """A named monotone true-toxicity vector stored in original dose order."""

    label: str
    true_toxicity: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip() or len(self.label) > 256:
            raise ValueError("scenario label must be nonempty and at most 256 characters")
        truth = _truth(self.true_toxicity, "true_toxicity")
        object.__setattr__(self, "true_toxicity", tuple(float(x) for x in truth))


@dataclass(frozen=True, slots=True)
class PoPScenarioInput:
    """Portable simulation request that runs through the existing PoP report API."""

    design: PoPDesign
    scenarios: tuple[PoPInputScenario, ...]
    total_patients: int
    cohort_size: int
    trials: int
    start_dose: int
    titration: bool
    earlyterm: bool
    risk_cutoff: float
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.design, PoPDesign):
            raise TypeError("design must be a PoPDesign")
        if not isinstance(self.scenarios, tuple) or not 1 <= len(self.scenarios) <= 20:
            raise ValueError("scenarios must contain 1..20 entries")
        if any(not isinstance(item, PoPInputScenario) for item in self.scenarios):
            raise TypeError("scenarios must contain PoPInputScenario values")
        labels = [item.label for item in self.scenarios]
        if len(set(labels)) != len(labels):
            raise ValueError("scenario labels must be unique")
        dose_count = len(self.scenarios[0].true_toxicity)
        if any(len(item.true_toxicity) != dose_count for item in self.scenarios):
            raise ValueError("all scenarios must have the same number of doses")
        object.__setattr__(
            self, "total_patients", _int(self.total_patients, "total_patients", 1, 1000)
        )
        object.__setattr__(self, "cohort_size", _int(self.cohort_size, "cohort_size", 1, 4))
        object.__setattr__(self, "trials", _int(self.trials, "trials", 1, 100_000))
        object.__setattr__(self, "start_dose", _int(self.start_dose, "start_dose", 1, dose_count))
        object.__setattr__(self, "seed", _int(self.seed, "seed", 0, 2**32 - 1))
        if not isinstance(self.titration, bool) or not isinstance(self.earlyterm, bool):
            raise ValueError("titration and earlyterm must be boolean")
        risk = scalar(self.risk_cutoff, "risk_cutoff")
        if not 0 <= risk <= 1:
            raise ValueError("risk_cutoff must be in [0,1]")
        object.__setattr__(self, "risk_cutoff", risk)
        if self.trials * self.total_patients > 100_000 or self.trials * dose_count > 100_000:
            raise ValueError(
                "each scenario must satisfy trials*patients and trials*doses <= 100000"
            )
        if self.trials * self.total_patients * len(self.scenarios) > 2_000_000:
            raise ValueError("aggregate planned patient replications exceed 2000000")

    def to_json(self) -> str:
        """Serialize deterministic UTF-8-ready JSON with no executable content."""
        payload = {
            "schema": _SCHEMA,
            "design": {
                "target": self.design.target,
                "cutoff": self.design.cutoff,
                "exclusion_cutoff": self.design.exclusion_cutoff,
                "safety_min_patients": self.design.safety_min_patients,
            },
            "simulation": {
                "total_patients": self.total_patients,
                "cohort_size": self.cohort_size,
                "trials": self.trials,
                "start_dose": self.start_dose,
                "titration": self.titration,
                "earlyterm": self.earlyterm,
                "risk_cutoff": self.risk_cutoff,
                "seed": self.seed,
            },
            "scenarios": [
                {"label": scenario.label, "true_toxicity": list(scenario.true_toxicity)}
                for scenario in self.scenarios
            ],
        }
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        if len(text) > _MAX_JSON_CHARS:
            raise ValueError("serialized scenario input exceeds the 1 MB limit")
        return text

    def write_json(self, path: str | Path) -> Path:
        """Atomically save the scenario input as UTF-8 JSON."""
        destination = Path(path)
        content = self.to_json()
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = stream.name
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except Exception:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
            raise
        return destination

    @classmethod
    def from_json(cls, text: str) -> PoPScenarioInput:
        """Parse a bounded, versioned PoP scenario input document."""
        if not isinstance(text, str) or len(text) > _MAX_JSON_CHARS:
            raise ValueError("scenario JSON must be text no larger than 1 MB")

        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in items:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        def reject_constant(value: str) -> None:
            raise ValueError(f"nonstandard JSON number is not allowed: {value}")

        try:
            raw = json.loads(text, object_pairs_hook=pairs, parse_constant=reject_constant)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("invalid PoP scenario JSON") from exc
        if not isinstance(raw, dict) or set(raw) != {"schema", "design", "simulation", "scenarios"}:
            raise ValueError("scenario JSON must contain schema, design, simulation, and scenarios")
        if raw["schema"] != _SCHEMA:
            raise ValueError(f"unsupported scenario schema: {raw['schema']!r}")
        design_raw, simulation = raw["design"], raw["simulation"]
        if not isinstance(design_raw, dict) or set(design_raw) != {
            "target",
            "cutoff",
            "exclusion_cutoff",
            "safety_min_patients",
        }:
            raise ValueError("design must contain exactly the four PoP design settings")
        sim_keys = {
            "total_patients",
            "cohort_size",
            "trials",
            "start_dose",
            "titration",
            "earlyterm",
            "risk_cutoff",
            "seed",
        }
        if not isinstance(simulation, dict) or set(simulation) != sim_keys:
            raise ValueError("simulation must contain exactly the documented PoP settings")
        raw_scenarios = raw["scenarios"]
        if not isinstance(raw_scenarios, list) or not 1 <= len(raw_scenarios) <= 20:
            raise ValueError("scenarios must be a list containing 1..20 entries")
        scenarios = []
        for item in raw_scenarios:
            if not isinstance(item, dict) or set(item) != {"label", "true_toxicity"}:
                raise ValueError("each scenario must contain only label and true_toxicity")
            if not isinstance(item["true_toxicity"], list):
                raise ValueError("scenario true_toxicity must be a JSON array")
            scenarios.append(PoPInputScenario(item["label"], tuple(item["true_toxicity"])))
        return cls(
            PoPDesign(**design_raw),
            tuple(scenarios),
            **simulation,
        )

    @classmethod
    def read_json(cls, path: str | Path) -> PoPScenarioInput:
        source = Path(path)
        if source.stat().st_size > _MAX_JSON_CHARS:
            raise ValueError("scenario input file exceeds the 1 MB limit")
        with source.open("rb") as stream:
            contents = stream.read(_MAX_JSON_CHARS + 1)
        if len(contents) > _MAX_JSON_CHARS:
            raise ValueError("scenario input file exceeds the 1 MB limit")
        try:
            text = contents.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("scenario input file must be UTF-8 JSON") from exc
        return cls.from_json(text)

    def run(self) -> PoPProtocolReport:
        """Run the existing serial PoP report from this captured request."""
        scenarios = tuple(
            PoPReportScenario(item.label, item.true_toxicity) for item in self.scenarios
        )
        return run_pop_protocol(
            self.design,
            scenarios,
            total_patients=self.total_patients,
            cohort_size=self.cohort_size,
            trials=self.trials,
            start_dose=self.start_dose,
            titration=self.titration,
            earlyterm=self.earlyterm,
            risk_cutoff=self.risk_cutoff,
            seed=self.seed,
        )


def _int(value: int, name: str, lower: int, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    parsed = scalar(value, name)
    if parsed != int(parsed) or not lower <= parsed <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return int(parsed)


def _truth(value: ArrayLike, name: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if (
            value.ndim != 1
            or value.size < 2
            or value.size > 100
            or np.iscomplexobj(value)
            or value.dtype.kind == "b"
        ):
            raise ValueError(f"{name} must be a real vector with 2..100 probabilities")
    elif isinstance(value, (list, tuple)):
        if (
            len(value) < 2
            or len(value) > 100
            or any(
                isinstance(
                    v, (list, tuple, np.ndarray, bool, np.bool_, complex, np.complexfloating)
                )
                for v in value
            )
        ):
            raise ValueError(f"{name} must be a real vector with 2..100 probabilities")
    else:
        raise ValueError(f"{name} must be a bounded vector with 2..100 probabilities")
    values = finite(value, name)
    if np.any((values < 0) | (values > 1)) or np.any(np.diff(values) < 0):
        raise ValueError(f"{name} must be nondecreasing probabilities in [0,1]")
    return values
