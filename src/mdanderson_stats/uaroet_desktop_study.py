"""Explicit current UAROET saved-input contracts; no native computed results imported."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray
from .uaroet import _integer, _joint_from_marginals, _real_matrix, _scalar, uaroet_parameter_names


def _lines(text: str) -> list[list[str]]:
    if not isinstance(text, str) or len(text.encode("utf-8")) > 2_000_000:
        raise ValueError("saved input must be text of at most 2MB")
    return [
        line.split()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("%")
    ]


def _numbers(text: str) -> list[float]:
    try:
        values = [float(word) for line in _lines(text) for word in line]
    except (TypeError, ValueError) as exc:
        raise ValueError("saved numeric input contains a nonnumeric value") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("saved numeric input must be finite")
    return values


def _keywords(text: str) -> dict[str, float]:
    result = {}
    for row in _lines(text):
        if len(row) != 2 or row[0] in result:
            raise ValueError("keyword input requires unique key/value pairs")
        try:
            value = float(row[1])
        except ValueError as exc:
            raise ValueError("keyword values must be numeric") from exc
        if not np.isfinite(value):
            raise ValueError("keyword values must be finite")
        result[row[0]] = value
    return result


def _whole(value: float, name: str, low: int, high: int) -> int:
    if not np.isfinite(value) or value != int(value) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return int(value)


def uaroet_desktop_prior_coordinates(
    values: ArrayLike, *, dose_count: int, efficacy_levels: int, toxicity_levels: int
) -> tuple[FloatArray, float]:
    """Map native threshold-major prior values to Python coordinates; retain rho metadata."""
    names = uaroet_parameter_names(dose_count, efficacy_levels, toxicity_levels)
    a = _real_matrix(values, "native prior values")
    if a.ndim != 1 or a.size != len(names) + 1:
        raise ValueError(
            "native prior file must include every active coordinate and association metadata"
        )
    parts: list[float] = []
    offset = 0
    for levels in (efficacy_levels, toxicity_levels):
        n = dose_count * (levels - 1)
        block = a[offset : offset + n].reshape(levels - 1, dose_count)
        parts.extend(block[:, 0])
        parts.extend(block[:, 1:].T.ravel())
        offset += n
    return _freeze(np.asarray(parts)), float(a[-1])


@dataclass(frozen=True)
class UAROETDesktopScenario:
    identifier: int
    efficacy: FloatArray
    toxicity: FloatArray
    association: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "identifier", _integer(self.identifier, "scenario identifier", 1, 9999)
        )
        e = _real_matrix(self.efficacy, "scenario efficacy")
        t = _real_matrix(self.toxicity, "scenario toxicity")
        if (
            e.ndim != 2
            or t.ndim != 2
            or not 2 <= e.shape[0] <= 5
            or t.shape[0] != e.shape[0]
            or not 2 <= e.shape[1] <= 4
            or not 2 <= t.shape[1] <= 4
        ):
            raise ValueError(
                "scenario dimensions must be 2–5 doses and 2–4 categories per endpoint"
            )
        for a in (e, t):
            if np.any(a < 0) or np.any(a > 1) or not np.allclose(a.sum(1), 1, rtol=0, atol=1e-12):
                raise ValueError("scenario marginal rows must be probabilities summing to one")
        association = _scalar(self.association, "scenario association")
        if not -1 < association < 1:
            raise ValueError("native scenario association must be strictly between-1 and 1")
        object.__setattr__(self, "association", association)
        object.__setattr__(self, "efficacy", _freeze(e))
        object.__setattr__(self, "toxicity", _freeze(t))

    def joint(self) -> FloatArray:
        result, _ = _joint_from_marginals(
            self.efficacy / self.efficacy.sum(1, keepdims=True),
            self.toxicity / self.toxicity.sum(1, keepdims=True),
            float(self.association),
            budget=[0, 0],
            max_evaluations=10000,
        )
        return _freeze(result)


@dataclass(frozen=True)
class UAROETDesktopCounts:
    """Disjoint observed counts and native escalation bookkeeping (not a decision)."""

    joint: NDArray[np.int64]
    toxicity_only: NDArray[np.int64]
    assigned: NDArray[np.int64]
    complete: NDArray[np.int64]
    likelihood: NDArray[np.int64]
    complete_efficacy: NDArray[np.int64]
    complete_toxicity: NDArray[np.int64]
    highest_assigned: int
    legacy_highest_allowed: int


def _frozen_counts(value: np.ndarray) -> NDArray[np.int64]:
    return np.frombuffer(np.asarray(value, dtype=np.int64).tobytes(), dtype=np.int64).reshape(
        value.shape
    )


@dataclass(frozen=True)
class UAROETDesktopStudy:
    dose_count: int
    efficacy_levels: int
    toxicity_levels: int
    prior_mean: FloatArray
    prior_sd: FloatArray
    utility: FloatArray
    trial_parameters: Mapping[str, float]
    simulation_parameters: Mapping[str, float]
    scenarios: tuple[UAROETDesktopScenario, ...] = ()
    patients: tuple[tuple[int, int, int, int], ...] = ()
    input_notes: tuple[str, ...] = ()
    native_association_mean: float = 0.0

    def __post_init__(self) -> None:
        D = _integer(self.dose_count, "dose_count", 2, 5)
        E = _integer(self.efficacy_levels, "efficacy_levels", 2, 4)
        T = _integer(self.toxicity_levels, "toxicity_levels", 2, 4)
        _validate_settings(self.trial_parameters, self.simulation_parameters, D, E, T)
        try:
            object.__setattr__(self, "scenarios", tuple(self.scenarios))
            object.__setattr__(self, "patients", tuple(tuple(row) for row in self.patients))
            object.__setattr__(self, "input_notes", tuple(self.input_notes))
        except TypeError as exc:
            raise ValueError("scenarios, patients and notes must be sequences") from exc
        object.__setattr__(
            self,
            "native_association_mean",
            _scalar(self.native_association_mean, "native association metadata"),
        )
        if any(not isinstance(s, UAROETDesktopScenario) for s in self.scenarios):
            raise ValueError("scenarios must contain UAROETDesktopScenario objects")
        if (
            any(not isinstance(note, str) or len(note) > 2000 for note in self.input_notes)
            or len(self.input_notes) > 100
        ):
            raise ValueError("input notes exceed bounded text limits")
        for key, dimension in (("dose_count", D), ("efficacy_levels", E), ("toxicity_levels", T)):
            object.__setattr__(self, key, dimension)
        n = len(uaroet_parameter_names(D, E, T))
        mean = _real_matrix(self.prior_mean, "prior_mean")
        sd = _real_matrix(self.prior_sd, "prior_sd")
        utility = _real_matrix(self.utility, "utility")
        if mean.shape != (n,) or sd.shape != (n,) or np.any(sd <= 0):
            raise ValueError(
                "prior vectors must match all Python model coordinates with positive SDs"
            )
        if (
            utility.shape != (E, T)
            or np.any(utility < 0)
            or np.any(utility > 100)
            or np.any(np.diff(utility, axis=0) < 0)
            or np.any(np.diff(utility, axis=1) > 0)
        ):
            raise ValueError(
                "utility must be 0–100, increasing in efficacy and decreasing in toxicity"
            )
        if len(self.scenarios) > 100 or len({s.identifier for s in self.scenarios}) != len(
            self.scenarios
        ):
            raise ValueError("study supports at most 100 uniquely identified scenarios")
        if any(s.efficacy.shape != (D, E) or s.toxicity.shape != (D, T) for s in self.scenarios):
            raise ValueError("scenario dimensions must match the study")
        previous = 0
        if len(self.patients) > self.trial_parameters["NumPatients"] or any(
            len(row) != 4 for row in self.patients
        ):
            raise ValueError("patient input must have four fields and fit NumPatients")
        for index, (identifier, dose, eff, tox) in enumerate(self.patients):
            _integer(identifier, "patient identifier", 1, 1_000_000)
            if identifier <= previous or index == 0 and identifier != 1:
                raise ValueError("patient identifiers must increase strictly, starting at 1")
            _integer(dose, "patient dose", 0, D - 1)
            _integer(eff, "efficacy category", -1, E - 1)
            _integer(tox, "toxicity category", -1, T - 1)
            previous = identifier
        object.__setattr__(
            self, "patients", tuple(tuple(int(v) for v in row) for row in self.patients)
        )
        for key, value in (("prior_mean", mean), ("prior_sd", sd), ("utility", utility)):
            object.__setattr__(self, key, _freeze(value))
        object.__setattr__(
            self,
            "trial_parameters",
            MappingProxyType({k: float(v) for k, v in self.trial_parameters.items()}),
        )
        object.__setattr__(
            self,
            "simulation_parameters",
            MappingProxyType({k: float(v) for k, v in self.simulation_parameters.items()}),
        )

    def counts(self) -> UAROETDesktopCounts:
        """Account for observation patterns without imputing pending endpoints.

        Native complete-case gates differ from likelihood counts. The native
        escalation ceiling opens the next dose only when the highest assigned
        dose has at least one toxicity-observed subject. It does not itself
        authorize an assignment or apply posterior safety screening.
        """
        joint = np.zeros(
            (self.dose_count, self.efficacy_levels, self.toxicity_levels), dtype=np.int64
        )
        partial = np.zeros((self.dose_count, self.toxicity_levels), dtype=np.int64)
        treated = np.zeros(self.dose_count, dtype=np.int64)
        for _, dose, eff, tox in self.patients:
            treated[dose] += 1
            if tox >= 0:
                if eff >= 0:
                    joint[dose, eff, tox] += 1
                else:
                    partial[dose, tox] += 1
        complete = joint.sum(axis=(1, 2))
        used = complete + partial.sum(axis=1)
        high = max((row[1] for row in self.patients), default=0)
        ceiling = min(self.dose_count - 1, high + 1) if used[high] else high
        return UAROETDesktopCounts(
            _frozen_counts(joint),
            _frozen_counts(partial),
            _frozen_counts(treated),
            _frozen_counts(complete),
            _frozen_counts(used),
            _frozen_counts(joint.sum(axis=2)),
            _frozen_counts(joint.sum(axis=1)),
            high,
            ceiling,
        )

    def outcome_counts(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return complete-joint, additional toxicity-only, and assigned counts."""
        counts = self.counts()
        return counts.joint, counts.toxicity_only, counts.assigned

    def to_json(self) -> str:
        return json.dumps(
            dict(
                schema="mdanderson-stats.uaroet-study",
                version=1,
                dose_count=self.dose_count,
                efficacy_levels=self.efficacy_levels,
                toxicity_levels=self.toxicity_levels,
                prior_mean=self.prior_mean.tolist(),
                prior_sd=self.prior_sd.tolist(),
                utility=self.utility.tolist(),
                trial_parameters=dict(self.trial_parameters),
                simulation_parameters=dict(self.simulation_parameters),
                scenarios=[
                    dict(
                        identifier=s.identifier,
                        efficacy=s.efficacy.tolist(),
                        toxicity=s.toxicity.tolist(),
                        association=s.association,
                    )
                    for s in self.scenarios
                ],
                patients=self.patients,
                input_notes=self.input_notes,
                native_association_mean=self.native_association_mean,
            ),
            allow_nan=False,
            indent=2,
        )


def _validate_settings(
    trial: Mapping[str, float], simulation: Mapping[str, float], D: int, E: int, T: int
) -> None:
    if not isinstance(trial, Mapping) or not isinstance(simulation, Mapping):
        raise ValueError("settings must be mappings")
    if any(not isinstance(k, str) for k in (*trial, *simulation)):
        raise ValueError("setting names must be strings")
    integers = {
        "NumPatients": (1, 500),
        "CohortSize_First": (1, 500),
        "CohortSize_NewDose": (1, 500),
        "CohortSize_OldDose": (1, 500),
        "DoseStart": (1, D),
        "LowestUnacceptableToxLevel": (1, T - 1),
        "LowestAcceptableEffLevel": (1, E - 1),
        "UseHybridGreedyArandAssignment": (0, 1),
        "NumPatientSurplusBeforeRandomize": (0, 500),
        "RandomizeByMeanUtility": (0, 1),
        "NumTimesToShrinkUtilityCutoff": (0, 500),
    }
    probability = {
        "MaxProbTox",
        "UpperAcceptablePiToxCutoff",
        "MinProbEff",
        "UpperAcceptablePiEffCutoff",
        "MinProbUtilityIsHighest",
    }
    utility = {
        "MinUtilityGoodSet",
        "InitialUtilityCutoffToRandomize",
        "FinalUtilityCutoffToRandomize",
    }
    times = {
        "MinDaysToEfficacy",
        "MaxDaysToEfficacy",
        "MinDaysToToxicity",
        "MaxDaysToToxicity",
        "MeanDaysBetweenArrivals",
    }
    required = set(integers) | probability | utility | times
    if set(trial) != required:
        raise ValueError(
            f"trial settings require exactly the documented keys; "
            f"missing={sorted(required - set(trial))}, "
            f"unsupported={sorted(set(trial) - required)}"
        )
    for k, v in trial.items():
        v = _scalar(v, f"trial setting {k}")
        if k in integers:
            _whole(float(v), k, *integers[k])
        elif k in probability and not 0 <= v <= 1:
            raise ValueError(f"{k} must lie in[0,1]")
        elif k in utility and not 0 <= v <= 100:
            raise ValueError(f"{k} must lie in[0,100]")
        elif k in times and not 0.1 <= v <= 3600:
            raise ValueError(f"{k} must lie in[.1,3600]")
    N = trial["NumPatients"]
    for k in (
        "CohortSize_First",
        "CohortSize_NewDose",
        "CohortSize_OldDose",
        "NumPatientSurplusBeforeRandomize",
        "NumTimesToShrinkUtilityCutoff",
    ):
        if trial[k] > N:
            raise ValueError(f"{k} must not exceed NumPatients")
    for endpoint in ("Efficacy", "Toxicity"):
        if trial["MinDaysTo" + endpoint] > trial["MaxDaysTo" + endpoint]:
            raise ValueError("outcome window minimum exceeds maximum")
    if trial["FinalUtilityCutoffToRandomize"] > trial["InitialUtilityCutoffToRandomize"]:
        raise ValueError("shrinking utility cutoff cannot increase")
    allowed = {
        "OutputLevel",
        "DetailLevel",
        "TargetMCSERatio",
        "TargetMCSERatio_Treat",
        "TargetMCSERatio_Prior",
        "TargetMCSERatio_Sim",
        "BurnInSamples",
        "NumChains",
        "DistributionSaveOption",
    }
    if set(simulation) - allowed:
        raise ValueError("unsupported simulation settings")
    for k, v in simulation.items():
        v = _scalar(v, f"simulation setting {k}")
        if k.startswith("TargetMCSE"):
            if not 0 < v < 1:
                raise ValueError("target MCSE ratios must be strictly between 0 and 1")
        elif k == "NumChains":
            _whole(float(v), k, 2, 8)
        elif k == "BurnInSamples":
            _whole(float(v), k, 0, 100000)
        else:
            _whole(float(v), k, 0, 3)


def read_uaroet_desktop_bundle(
    files: Mapping[str, str],
    *,
    patient_text: str | None = None,
    repair_example_typo: bool = False,
    utility_identifier: int | None = None,
) -> UAROETDesktopStudy:
    """Read an explicit original-file text bundle; reject unsupported scientific settings.

    The bundled Algorithm spelling is rejected unless the caller explicitly
    requests repair to the guide's Assignment keyword. The repair is recorded.
    """
    if (
        not isinstance(files, Mapping)
        or len(files) > 128
        or not isinstance(repair_example_typo, bool)
    ):
        raise ValueError("files must be a bounded filename/text mapping")
    if (
        any(not isinstance(k, str) or not isinstance(v, str) for k, v in files.items())
        or sum(len(v.encode("utf-8")) for v in files.values()) > 10_000_000
    ):
        raise ValueError("bundle must contain filename/text pairs totaling at most 10MB")

    def required(name: str) -> str:
        if name not in files:
            raise ValueError(f"missing saved input {name}")
        return files[name]

    model = _keywords(required("UAROET_ModelParams.txt"))
    if set(model) != {"NumDoses", "NumEffLevels", "NumToxLevels"}:
        raise ValueError("only the documented ordinal saturated model settings are supported")
    D = _whole(model["NumDoses"], "NumDoses", 2, 5)
    E = _whole(model["NumEffLevels"], "NumEffLevels", 2, 4)
    T = _whole(model["NumToxLevels"], "NumToxLevels", 2, 4)
    n = len(uaroet_parameter_names(D, E, T))
    mean, association_metadata = uaroet_desktop_prior_coordinates(
        _numbers(required("UAROET_PriorMeans.txt")),
        dose_count=D,
        efficacy_levels=E,
        toxicity_levels=T,
    )
    if "UAROET_PriorStdDeviations.txt" in files:
        sd, _ = uaroet_desktop_prior_coordinates(
            _numbers(files["UAROET_PriorStdDeviations.txt"]),
            dose_count=D,
            efficacy_levels=E,
            toxicity_levels=T,
        )
    else:
        values = _numbers(required("UAROET_PriorStdDev.txt"))
        if len(values) != 1 or values[0] <= 0:
            raise ValueError("default prior SD file must contain one positive value")
        sd = np.full(n, values[0])
    selected = _numbers(required("UAROET_DefaultUtility.txt"))
    if len(selected) != 1:
        raise ValueError("default utility file must contain one identifier")
    identifier = _whole(selected[0], "utility identifier", 1, 99)
    notes = []
    if utility_identifier is not None:
        chosen = _integer(utility_identifier, "utility_identifier", 1, 99)
        if chosen != identifier:
            notes.append(
                f"Explicitly selected utility{chosen} instead of saved default utility{identifier}."
            )
        identifier = chosen
    utility = np.full((E, T), np.nan)
    for row in _lines(required(f"UAROET_Utility_{identifier}.txt")):
        if len(row) != 3:
            raise ValueError("utility records require efficacy, toxicity, value")
        e = _whole(float(row[0]), "utility efficacy", 0, E - 1)
        t = _whole(float(row[1]), "utility toxicity", 0, T - 1)
        if np.isfinite(utility[e, t]):
            raise ValueError("duplicate utility cell")
        utility[e, t] = float(row[2])
    trial = _keywords(required("UAROET_TrialParams.txt"))
    canonical = "UseHybridGreedyArandAssignment"
    for alias in ("UseHybridGreedyARandAssignment", "UseHybridGreedyArandAlgorithm"):
        if alias not in trial:
            continue
        if canonical in trial:
            raise ValueError("duplicate hybrid-assignment setting")
        if alias.endswith("Algorithm"):
            if not repair_example_typo:
                raise ValueError(
                    "bundled Algorithm spelling is not recognized by native 1.8; explicitly set "
                    "repair_example_typo=True or use the documented Assignment spelling"
                )
            notes.append(
                "Explicitly repaired bundled UseHybridGreedyArandAlgorithm typo "
                "to UseHybridGreedyArandAssignment."
            )
        trial[canonical] = trial.pop(alias)
    simulation = _keywords(required("UAROET_SimulationParams.txt"))
    _validate_settings(trial, simulation, D, E, T)
    scenarios = []
    for name, text in sorted(files.items()):
        match = re.fullmatch(r"UAROET_Scenario_(\d+)\.txt", name)
        if match is None:
            continue
        rows = _lines(text)
        rho = 0.0
        if rows and len(rows[0]) == 1:
            rho = float(rows.pop(0)[0])
        if len(rows) != D:
            raise ValueError("scenario must contain each active dose exactly once")
        eprob = np.zeros((D, E))
        tprob = np.zeros((D, T))
        seen = set()
        for row in rows:
            if len(row) != 1 + E + T:
                raise ValueError("scenario row must include only active marginal categories")
            d = _whole(float(row[0]), "scenario dose", 1, D) - 1
            if d in seen:
                raise ValueError("duplicate scenario dose")
            seen.add(d)
            eprob[d] = [float(v) for v in row[1 : 1 + E]]
            tprob[d] = [float(v) for v in row[1 + E :]]
        scenarios.append(UAROETDesktopScenario(int(match[1]), eprob, tprob, rho))
    patients: list[tuple[int, int, int, int]] = []
    if patient_text is not None:
        for row in _lines(patient_text):
            if len(row) != 4:
                raise ValueError("patient rows require identifier, dose, efficacy, toxicity")
            limits = ((1, 1000000), (1, D), (-1, E - 1), (-1, T - 1))
            patient_values = [
                _whole(float(v), "patient field", *bounds) for v, bounds in zip(row, limits)
            ]
            patients.append(
                (patient_values[0], patient_values[1] - 1, patient_values[2], patient_values[3])
            )
    return UAROETDesktopStudy(
        D,
        E,
        T,
        mean,
        sd,
        utility,
        trial,
        simulation,
        tuple(sorted(scenarios, key=lambda scenario: scenario.identifier)),
        tuple(patients),
        tuple(notes),
        association_metadata,
    )


def read_uaroet_desktop_study(text: str) -> UAROETDesktopStudy:
    """Reopen the versioned Python saved-study schema, rejecting duplicate keys."""
    _lines(text)

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate saved-study key")
            result[key] = value
        return result

    try:
        data = json.loads(text, object_pairs_hook=unique)
        if (
            not isinstance(data, dict)
            or data.pop("schema", None) != "mdanderson-stats.uaroet-study"
            or type(data.get("version")) is not int
            or data.pop("version", None) != 1
        ):
            raise ValueError("unsupported UAROET saved-study schema")
        data["scenarios"] = tuple(UAROETDesktopScenario(**s) for s in data["scenarios"])
        return UAROETDesktopStudy(**data)
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid UAROET saved-study structure") from exc
