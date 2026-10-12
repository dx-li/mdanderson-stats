"""Bounded, versioned PRT studies and original desktop text input conversion.

Patient doses are zero-based in Python. The original third column means censored,
not event. Import preserves input contracts, not the native inference/RNG engine.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np

from ._cdflib import _freeze
from ._validation import FloatArray


def _integer(value: object, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if not low <= value <= high or not isfinite(value) or int(value) != value:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class PRTPatient:
    dose: int
    observed_interval: int
    censored: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "dose", _integer(self.dose, "dose", 0, 9))
        object.__setattr__(
            self, "observed_interval", _integer(self.observed_interval, "interval", 0, 10)
        )
        if type(self.censored) is not bool:
            raise ValueError("censored must be boolean")
        if not self.censored and self.observed_interval == 0:
            raise ValueError("an event requires an observed interval in 1..C")


@dataclass(frozen=True)
class PRTStudy:
    intervals: int
    doses: int
    target: float = 0.3
    epsilon: float = 0.05
    negligible_cutoff: float = 0.3
    excessive_cutoff: float = 0.9
    prior_mean: float = -14.0
    prior_variance: float = 28.0
    patients: tuple[PRTPatient, ...] = ()
    interval_endpoints: tuple[float, ...] = ()
    max_patients: int | None = None
    true_toxicity: tuple[float, ...] = ()
    mean_interarrival: float | None = None
    hazard_alpha: float | None = None
    hazard_beta: float | None = None
    repetitions: int | None = None
    draws: int = 2000
    warmup: int = 1000
    chains: int = 4
    source_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, low, high in [
            ("intervals", 3, 10),
            ("doses", 2, 10),
            ("draws", 8, 100000),
            ("warmup", 0, 100000),
            ("chains", 2, 16),
        ]:
            object.__setattr__(self, name, _integer(getattr(self, name), name, low, high))
        for name in (
            "target",
            "epsilon",
            "negligible_cutoff",
            "excessive_cutoff",
            "prior_mean",
            "prior_variance",
        ):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if not 0 < self.target < 1 or not 0 <= self.epsilon <= 1:
            raise ValueError("require target in (0,1) and epsilon in [0,1]")
        if not 0 < self.negligible_cutoff < self.excessive_cutoff < 1 or self.prior_variance <= 0:
            raise ValueError("require ordered cutoffs in (0,1) and positive prior variance")
        for name in ("patients", "source_notes", "interval_endpoints", "true_toxicity"):
            if not isinstance(getattr(self, name), (list, tuple)):
                raise ValueError(f"{name} must be a sequence")
        patients = tuple(self.patients)
        if len(patients) > 1000 or any(not isinstance(p, PRTPatient) for p in patients):
            raise ValueError("at most 1000 PRTPatient records are supported")
        if any(p.dose >= self.doses or p.observed_interval > self.intervals for p in patients):
            raise ValueError("patient dose/interval exceeds study dimensions")
        object.__setattr__(self, "patients", patients)
        notes = tuple(self.source_notes)
        if len(notes) > 10 or any(not isinstance(s, str) or len(s) > 500 for s in notes):
            raise ValueError("source notes must contain at most ten short strings")
        object.__setattr__(self, "source_notes", notes)
        endpoints = tuple(_number(v, "endpoint") for v in self.interval_endpoints)
        truth = tuple(_number(v, "true_toxicity") for v in self.true_toxicity)
        object.__setattr__(self, "interval_endpoints", endpoints)
        object.__setattr__(self, "true_toxicity", truth)
        simulation = self.max_patients is not None
        if not simulation:
            if (
                endpoints
                or truth
                or any(
                    getattr(self, n) is not None
                    for n in ("mean_interarrival", "hazard_alpha", "hazard_beta", "repetitions")
                )
            ):
                raise ValueError("simulation settings must be supplied together")
        else:
            if patients:
                raise ValueError("simulation study must start with an empty patient history")
            object.__setattr__(
                self, "max_patients", _integer(self.max_patients, "max_patients", 3, 500)
            )
            object.__setattr__(
                self, "repetitions", _integer(self.repetitions, "repetitions", 1, 100000)
            )
            if (
                len(endpoints) != self.intervals
                or endpoints[0] <= 0
                or any(b <= a for a, b in zip(endpoints, endpoints[1:]))
            ):
                raise ValueError("require C strictly increasing positive interval endpoints")
            if len(truth) != self.doses or any(not 0 < p < 1 for p in truth):
                raise ValueError("require one true probability in (0,1) per dose")
            for name in ("mean_interarrival", "hazard_alpha", "hazard_beta"):
                value = _number(getattr(self, name), name)
                if value <= 0:
                    raise ValueError(f"{name} must be positive")
                object.__setattr__(self, name, value)

    def to_json(self) -> str:
        return (
            json.dumps(
                {"schema": "mdanderson_stats.prt.study", "version": 1, **asdict(self)},
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )


def read_prt_study(text: str) -> PRTStudy:
    """Read strict version1 JSON; unknown fields and duplicate keys are errors."""
    _text(text)

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    data = json.loads(text, object_pairs_hook=unique)
    if (
        not isinstance(data, dict)
        or data.pop("schema", None) != "mdanderson_stats.prt.study"
        or type(data.get("version")) is not int
        or data.pop("version", None) != 1
    ):
        raise ValueError("unsupported PRT study schema/version")
    if set(data) != set(PRTStudy.__dataclass_fields__):
        raise ValueError("study fields do not match version1")
    if not isinstance(data["patients"], list):
        raise ValueError("patients must be a list")
    try:
        data["patients"] = tuple(PRTPatient(**p) for p in data["patients"])
        return PRTStudy(**data)
    except TypeError as exc:
        raise ValueError("invalid PRT study fields") from exc


def _text(text: str) -> None:
    if not isinstance(text, str) or len(text.encode("utf-8")) > 1_000_000:
        raise ValueError("input must be text no larger than 1 MB")


def _rows(text: str) -> list[list[float]]:
    _text(text)
    try:
        rows = [[float(x) for x in line.split()] for line in text.splitlines() if line.strip()]
    except ValueError as exc:
        raise ValueError("expected numeric input rows") from exc
    if any(not isfinite(x) for row in rows for x in row):
        raise ValueError("input values must be finite")
    return rows


def import_prt_conduct(
    settings: str, patients: str, *, allow_unused_seed_lines: bool = False
) -> PRTStudy:
    """Read original eight-line settings and three-column patient data.

    The shipped file adds two seed-looking lines ignored by the native program.
    Accepting those lines requires explicit acknowledgement; they never seed Python.
    """
    rows = _rows(settings)
    if type(allow_unused_seed_lines) is not bool:
        raise ValueError("allow_unused_seed_lines must be boolean")
    notes: tuple[str, ...] = ()
    if len(rows) == 10 and allow_unused_seed_lines:
        if any(len(row) != 1 for row in rows[8:]):
            raise ValueError("unused seed lines must each contain one number")
        notes = (
            "Two trailing native-unused settings lines were discarded; "
            "Python RNG remains explicit.",
        )
        rows = rows[:8]
    if len(rows) != 8 or any(len(row) != 1 for row in rows):
        raise ValueError(
            "require eight settings lines; acknowledge two native-unused seed lines explicitly"
        )
    C, D, target, epsilon, lo, hi, mean, var = (row[0] for row in rows)
    C = _integer(C, "intervals", 3, 10)
    D = _integer(D, "doses", 2, 10)
    history = []
    for row in _rows(patients):
        if len(row) != 3:
            raise ValueError("patient rows require dose, observed interval, censoring indicator")
        dose = _integer(row[0], "native dose", 1, 10) - 1
        interval = _integer(row[1], "observed interval", 0, 10)
        censor = _integer(row[2], "censoring indicator", 0, 1)
        history.append(PRTPatient(dose, interval, bool(censor)))
    return PRTStudy(
        C, D, target, epsilon, lo, hi, mean, var, patients=tuple(history), source_notes=notes
    )


def import_prt_simulation(settings: str) -> PRTStudy:
    """Read the original four labelled simulation sections, with exact row widths."""
    _text(settings)
    lines = [line.strip() for line in settings.splitlines() if line.strip()]
    if len(lines) != 21 or any(
        lines[i] != label
        for i, label in [
            (0, "Trial Inputs"),
            (6, "Design Inputs"),
            (13, "MCMC Inputs"),
            (16, "Simulation Inputs"),
        ]
    ):
        raise ValueError("expected four original simulation sections and 17 numeric rows")
    rows = _rows("\n".join(line for i, line in enumerate(lines) if i not in (0, 6, 13, 16)))
    C, D, N, window = (rows[i][0] for i in range(4))
    _integer(window, "native assessment window", 1, 1000000)
    C = _integer(C, "intervals", 3, 10)
    D = _integer(D, "doses", 2, 10)
    for i, row in enumerate(rows):
        width = int(C) if i == 4 else int(D) if i == 13 else 2 if i == 15 else 1
        if len(row) != width:
            raise ValueError("simulation row width mismatch")
    if rows[4][-1] != window:
        raise ValueError("last endpoint must equal assessment window")
    target, epsilon, lo, hi, mean, var = (rows[i][0] for i in range(5, 11))
    return PRTStudy(
        _integer(C, "intervals", 3, 10),
        _integer(D, "doses", 2, 10),
        target,
        epsilon,
        lo,
        hi,
        mean,
        var,
        interval_endpoints=tuple(rows[4]),
        max_patients=_integer(N, "max_patients", 3, 500),
        true_toxicity=tuple(rows[13]),
        mean_interarrival=rows[14][0],
        hazard_alpha=rows[15][0],
        hazard_beta=rows[15][1],
        repetitions=_integer(rows[16][0], "repetitions", 1, 100000),
        warmup=_integer(rows[11][0], "warmup", 0, 100000),
        draws=_integer(rows[12][0], "draws", 8, 100000),
    )


def prt_study_counts(
    study: PRTStudy,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return survived, events, assigned, assessed and pending-by-interval counts."""
    if not isinstance(study, PRTStudy):
        raise TypeError("study must be PRTStudy")
    survived = np.zeros((study.intervals, study.doses))
    events = np.zeros_like(survived)
    assigned = np.zeros(study.doses)
    assessed = np.zeros(study.doses)
    pending = np.zeros_like(survived)
    for p in study.patients:
        assigned[p.dose] += 1
        if p.censored:
            survived[: p.observed_interval, p.dose] += 1
            if p.observed_interval == study.intervals:
                assessed[p.dose] += 1
            else:
                pending[p.observed_interval, p.dose] += 1
        else:
            survived[: p.observed_interval - 1, p.dose] += 1
            events[p.observed_interval - 1, p.dose] += 1
            assessed[p.dose] += 1
    return (
        _freeze(survived),
        _freeze(events),
        _freeze(assigned),
        _freeze(assessed),
        _freeze(pending),
    )
