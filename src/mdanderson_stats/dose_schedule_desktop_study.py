"""Bounded, immutable DSF studies and original managed saved-input import."""

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np

from .dose_schedule import validate_schedules
from .dose_schedule_decision import _scalar
from .dose_schedule_prior import dose_schedule_moment_prior
from .dose_schedule_trial import _count

_SCHEMA = "mdanderson-stats/dose-schedule-desktop/v1"
_GUID = "4DE4157E-8DBC-4c28-B3DE-807264EA0D80"
_MAX_INPUT = 4_000_000


def _time(value, name, *, positive=False):
    value = _scalar(value, name)
    if value < 0 or value > 1e6 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'} and <=1e6")
    return value


def _vector(value, name, size):
    raw = np.asarray(value)
    if raw.shape != (size,) or raw.dtype.kind not in "iuf" or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must contain {size} finite real values")
    return tuple(float(v) for v in raw)


@dataclass(frozen=True)
class DoseScheduleDesktopPatient:
    """Assigned pair and actual administration ledger; indices are zero-based."""

    entry_time: float
    assigned_dose: int
    assigned_schedule: int
    outcome: int
    outcome_time: float
    administration_times: tuple[float, ...]
    dose_indices: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "entry_time", _time(self.entry_time, "entry_time"))
        object.__setattr__(self, "outcome_time", _time(self.outcome_time, "outcome_time"))
        for name, high in (("assigned_dose", 6), ("assigned_schedule", 4), ("outcome", 2)):
            object.__setattr__(self, name, _count(getattr(self, name), name, 0, high))
        if len(self.administration_times) > 100 or len(self.dose_indices) != len(
            self.administration_times
        ):
            raise ValueError("each patient permits at most 100 matched administrations")
        times = tuple(_time(v, "administration_time") for v in self.administration_times)
        doses = tuple(_count(v, "dose_index", 0, 6) for v in self.dose_indices)
        if any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError("actual administration times must strictly increase")
        object.__setattr__(self, "administration_times", times)
        object.__setattr__(self, "dose_indices", doses)


@dataclass(frozen=True)
class DoseScheduleDesktopStudy:
    """Scientific input shared by conduct, simulation and captured-input reports.

    Integration speed and extra-sample-factor settings of the old integration
    engine are metadata on import, not silently mapped to Python sampler draws.
    Python sampler settings are explicit execution inputs.
    """

    prior_probabilities: tuple[float, ...]
    peak_times: tuple[float, ...]
    tail_times: tuple[float, ...]
    schedule_count: int
    administrations_per_cycle: int
    cycle_interval: float
    horizon: float
    target: float
    toxicity_limit: float
    upper_probability: float
    maximum_patients: int = 30
    repetitions: int = 100
    mean_arrival_interval: float = 7.0
    nu_area: float = 1.5
    nu_time: float = 1.5
    distribution: str = "exponential"
    shape: float = 1.0
    low_grade_fraction: float = 0.0
    low_grade_delay: float = 14.0
    scenarios: tuple[tuple[tuple[float, ...], ...], ...] = ()
    patients: tuple[DoseScheduleDesktopPatient, ...] = ()
    study_time: float = 0.0
    original_integration_speed: int | None = None
    original_extra_sample_factor: int | None = None

    def __post_init__(self):
        doses = _count(len(self.prior_probabilities), "dose_count", 1, 7)
        for name in ("prior_probabilities", "peak_times", "tail_times"):
            object.__setattr__(self, name, _vector(getattr(self, name), name, doses))
        for name, low, high in (
            ("schedule_count", 1, 5),
            ("administrations_per_cycle", 1, 20),
            ("maximum_patients", 1, 200),
            ("repetitions", 1, 100000),
        ):
            object.__setattr__(self, name, _count(getattr(self, name), name, low, high))
        for name in ("cycle_interval", "horizon", "mean_arrival_interval"):
            object.__setattr__(self, name, _time(getattr(self, name), name, positive=True))
        for name in ("study_time", "low_grade_delay"):
            object.__setattr__(self, name, _time(getattr(self, name), name))
        for name in ("target", "toxicity_limit", "upper_probability", "low_grade_fraction"):
            value = _scalar(getattr(self, name), name)
            if not 0 <= value <= 1 or (name != "low_grade_fraction" and value in (0, 1)):
                raise ValueError(f"{name} must be a valid probability")
            object.__setattr__(self, name, value)
        for name in ("nu_area", "nu_time"):
            value = _scalar(getattr(self, name), name)
            if value <= 1:
                raise ValueError(f"{name} must exceed one")
            object.__setattr__(self, name, value)
        if self.cycle_interval < self.administrations_per_cycle:
            raise ValueError("cycles cannot overlap")
        self.schedules()
        self.prior()
        if self.distribution not in ("exponential", "weibull", "lognormal"):
            raise ValueError("unknown event distribution")
        shape = _scalar(self.shape, "shape")
        if not 0 < shape <= 100:
            raise ValueError("shape must be in (0,100]")
        object.__setattr__(self, "shape", shape)
        if len(self.scenarios) > 100:
            raise ValueError("at most 100 scenarios are supported")
        scenarios = []
        for scenario in self.scenarios:
            raw = np.asarray(scenario)
            if raw.shape != (doses, self.schedule_count) or raw.dtype.kind not in "iuf":
                raise ValueError("scenario must match dose/schedule dimensions")
            if np.any(~np.isfinite(raw)) or np.any((raw < 0) | (raw > 1)):
                raise ValueError("scenario probabilities must be finite in [0,1]")
            if np.any(np.diff(raw, axis=0) < 0) or np.any(np.diff(raw, axis=1) < 0):
                raise ValueError("scenario toxicity must not decrease with dose or schedule")
            scenarios.append(tuple(tuple(float(v) for v in row) for row in raw))
        object.__setattr__(self, "scenarios", tuple(scenarios))
        patients = tuple(self.patients)
        if len(patients) > self.maximum_patients or any(
            not isinstance(p, DoseScheduleDesktopPatient) for p in patients
        ):
            raise ValueError("invalid patient ledger or maximum patient count exceeded")
        for p in patients:
            if (
                p.assigned_dose >= doses
                or p.assigned_schedule >= self.schedule_count
                or any(d >= doses for d in p.dose_indices)
            ):
                raise ValueError("patient refers to a regimen outside study")
        if any(a.entry_time > b.entry_time for a, b in zip(patients, patients[1:])):
            raise ValueError("patient entries must be in calendar order")
        object.__setattr__(self, "patients", patients)
        for name in ("original_integration_speed", "original_extra_sample_factor"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _count(getattr(self, name), name, 0, 100000))

    def schedules(self):
        values = [
            tuple(
                c * self.cycle_interval + j
                for c in range(s)
                for j in range(self.administrations_per_cycle)
            )
            for s in range(1, self.schedule_count + 1)
        ]
        return validate_schedules(values, self.horizon)

    def prior(self):
        return dose_schedule_moment_prior(
            self.prior_probabilities,
            self.administrations_per_cycle,
            self.peak_times,
            self.tail_times,
            lambda1=self.nu_area,
            lambda2=self.nu_time,
        )

    def to_json(self):
        return (
            json.dumps(
                {"schema": _SCHEMA, "study": asdict(self)},
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate saved field: " + key)
        result[key] = value
    return result


def read_dose_schedule_desktop_study(text: str) -> DoseScheduleDesktopStudy:
    """Read versioned Python JSON or original DSF2.2 XML/HTML saved inputs."""
    if not isinstance(text, str) or len(text.encode("utf-8")) > _MAX_INPUT:
        raise ValueError("study must be text of at most four MB")
    if text.lstrip().startswith("{"):
        try:
            value = json.loads(text, object_pairs_hook=_unique)
            if set(value) != {"schema", "study"} or value["schema"] != _SCHEMA:
                raise ValueError("unknown study schema")
            data = value["study"]
            data["patients"] = tuple(
                DoseScheduleDesktopPatient(**v) for v in data.get("patients", ())
            )
            return DoseScheduleDesktopStudy(**data)
        except (TypeError, KeyError, AttributeError) as exc:
            raise ValueError("invalid study JSON") from exc
    begin = "begin" + _GUID
    end = "end" + _GUID
    if begin in text:
        if text.count(begin) != 1 or text.count(end) != 1:
            raise ValueError("ambiguous original study envelope")
        text = text.split(begin, 1)[1].split(end, 1)[0].strip()
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ValueError("XML declarations are not accepted")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError("invalid original study XML") from exc
    if root.tag != "DSFSimApp" or root.get("ProgID") != _GUID or root.get("FileVersion") != "2.2.0":
        raise ValueError("not a supported DSF2.2 saved study")
    for node in root.iter():
        # Primitive arrays/collections repeat element names intentionally.
        if node.tag not in ("m_pDSFScenarios", "m_pDSFTreatments") and not node.tag.startswith(
            "m_v"
        ):
            tags = [child.tag for child in node]
            if len(tags) != len(set(tags)):
                raise ValueError("duplicate XML scalar/container")
    data = root.find("DSFSimulatorData")
    if data is None:
        raise ValueError("missing simulator data")
    params = data.find("m_pDSFParameters")
    if params is None:
        raise ValueError("missing model parameters")

    def number(node, name, integer=False):
        if node is None:
            raise ValueError("missing original container")
        raw = node.findtext(name)
        if raw is None:
            raise ValueError("missing original field: " + name)
        value = float(raw)
        if not isfinite(value) or (integer and value != int(value)):
            raise ValueError("invalid original numeric field: " + name)
        return int(value) if integer else value

    def vector(node, name, integer=False):
        elements = node.findall(name + "/int" if integer else name + "/double")
        values = [float(e.text) for e in elements]
        if any(not isfinite(v) or (integer and v != int(v)) for v in values):
            raise ValueError("invalid original vector")
        return tuple(int(v) if integer else v for v in values)

    doses = _count(number(params, "m_numDoses", True), "dose_count", 1, 7)
    schedules = _count(number(params, "m_numSchedules", True), "schedule_count", 1, 5)
    scenarios = []
    for node in data.findall("m_pDSFScenarios/DSFScenario"):
        if (
            number(node, "m_numDoses", True) != doses
            or number(node, "m_numSchedules", True) != schedules
        ):
            raise ValueError("scenario dimensions differ from study")
        values = vector(node, "m_vTrueCumuProbToxByTau")
        if len(values) != 35:
            raise ValueError("native scenario vector must have fixed stride five and length 35")
        scenarios.append(
            tuple(tuple(values[d * 5 + s] for s in range(schedules)) for d in range(doses))
        )
    patients = []
    for node in data.findall("m_pDSFTreatments/DSFTreatment"):
        times = vector(node, "m_vDaysGiven", True)
        indices = vector(node, "m_vDosesGiven", True)
        if len(times) != number(node, "m_numAdmins", True):
            raise ValueError("native administration count mismatch")
        patients.append(
            DoseScheduleDesktopPatient(
                number(node, "m_iDayOfEntry"),
                number(node, "m_iDoseAssigned", True),
                number(node, "m_iScheduleAssigned", True),
                number(node, "m_iOutcome", True),
                number(node, "m_iDaysToOutcome"),
                times,
                indices,
            )
        )
    distribution = number(params, "m_iDistributionTimeObservation", True)
    if distribution not in (0, 1, 2):
        raise ValueError("unknown native event distribution")
    return DoseScheduleDesktopStudy(
        vector(params, "m_vPriorCumuProbTox")[:doses],
        vector(params, "m_vPriorTimeToPeak")[:doses],
        vector(params, "m_vPriorTimeToEnd")[:doses],
        schedules,
        number(params, "m_numAdminsPerCycle", True),
        number(params, "m_numDaysBetweenCycleStarts"),
        number(params, "m_maxDaysFollowUp"),
        number(params, "m_dTargetCumuProbToxByTau"),
        number(params, "m_dMaxCumuProbToxByTau"),
        number(params, "m_dUpperAcceptableProbCutoff"),
        maximum_patients=number(params, "m_maxPatients", True),
        repetitions=number(params, "m_numRepetitions", True),
        mean_arrival_interval=number(params, "m_dMeanTimeBetweenArrivals"),
        nu_area=number(params, "m_nu_area"),
        nu_time=number(params, "m_nu_time"),
        distribution=("exponential", "weibull", "lognormal")[distribution],
        shape=number(params, "m_dParameterTimeObservation"),
        low_grade_fraction=number(params, "m_dFractionLowGradeTox"),
        low_grade_delay=number(params, "m_dTimeToResolveLowGrade"),
        scenarios=tuple(scenarios),
        patients=tuple(patients),
        study_time=number(data.find("m_pDSFTreatmentInfo"), "m_iDayOfAssignment"),
        original_integration_speed=number(params, "m_iIntegrationSpeed", True),
        original_extra_sample_factor=number(params, "m_nExtraSampleFactor", True),
    )
