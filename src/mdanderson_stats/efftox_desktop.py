"""Current EffTox 5.2.3 binary-model study and patient XML interchange.

The saved document is retained semantically, including inactive and unknown
settings. Historical four-coefficient files require a separate interpretation
and are deliberately not migrated by this reader.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .efftox_decision import EffToxContour
from .efftox_model import EffToxPrior, efftox_standardize

_GUID = "C893BCF6-6184-419a-8173-354C4BDE8728"
# Original PatientDlg.GetPatientData, not ordinary two-bit encoding.
_OUTCOMES = {-1: None, 0: (0, 1), 1: (0, 0), 2: (1, 1), 3: (1, 0)}


def _text(node: ET.Element, path: str) -> str:
    values = node.findall(path)
    if len(values) != 1 or values[0].text is None:
        raise ValueError(f"expected exactly one EffTox {path}")
    return values[0].text.strip()


def _number(node: ET.Element, path: str) -> float:
    value = float(_text(node, path))
    if not np.isfinite(value):
        raise ValueError(f"EffTox {path} must be finite")
    return value


def _integer(node: ET.Element, path: str, low: int, high: int) -> int:
    value = _number(node, path)
    if value != int(value) or not low <= value <= high:
        raise ValueError(f"EffTox {path} must be an integer in [{low},{high}]")
    return int(value)


def _vector(node: ET.Element, path: str, size: int) -> tuple[float, ...]:
    values = tuple(float(v.text or "") for v in node.findall(path))
    if len(values) != size or not np.isfinite(values).all():
        raise ValueError(f"EffTox {path} requires {size} finite values")
    return values


def _probabilities(node: ET.Element, path: str, size: int) -> tuple[float, ...]:
    values = _vector(node, path, size)
    if any(not 0 <= v <= 1 for v in values):
        raise ValueError(f"EffTox {path} must contain probabilities")
    return values


@dataclass(frozen=True)
class EffToxDesktopPatient:
    """Native zero-based calculated/administered dose indices and outcome code.

    ``outcome`` is -1 unknown, 0 toxicity only, 1 neither, 2 both, 3 efficacy
    only. Dates and IDs are identifiers, never time-to-event observations.
    """

    patient_id: int
    begin_date: str
    calculated_dose_index: int
    administered_dose_index: int
    outcome: int
    cohort: int

    @property
    def binary_outcome(self) -> tuple[int, int] | None:
        """Return (efficacy, toxicity), or None for an unspecified outcome."""
        return _OUTCOMES[self.outcome]


@dataclass(frozen=True)
class EffToxDesktopScenario:
    """Saved marginal and conditional truth, including inactive conditionals."""

    toxicity: tuple[float, ...]
    efficacy: tuple[float, ...]
    efficacy_given_no_toxicity: tuple[float, ...]
    dependent: bool

    def joint_probabilities(self) -> np.ndarray:
        """Return dose-by-efficacy-by-toxicity cells without truth approximation."""
        e, t = np.asarray(self.efficacy), np.asarray(self.toxicity)
        e_no_t = np.asarray(self.efficacy_given_no_toxicity) if self.dependent else e
        joint = np.empty((e.size, 2, 2))
        joint[:, 1, 0] = e_no_t * (1 - t)
        joint[:, 1, 1] = e - joint[:, 1, 0]
        joint[:, 0, 0] = (1 - e_no_t) * (1 - t)
        joint[:, 0, 1] = t - joint[:, 1, 1]
        if np.any(joint < -1e-14) or np.any(joint > 1 + 1e-14):
            raise ValueError("conditional efficacy probabilities imply impossible joint truth")
        # Only remove roundoff at exact probability boundaries.
        return np.clip(joint, 0, 1)


@dataclass(frozen=True)
class EffToxDesktopStudy:
    """Current native study with validated settings and preserved XML metadata."""

    xml: str
    doses: tuple[float, ...]
    prior_means: tuple[float, ...]
    prior_sds: tuple[float, ...]
    contour_points: tuple[float, ...]
    max_patients: int
    cohort_size: int
    starting_dose_index: int
    patients: tuple[EffToxDesktopPatient, ...]
    scenarios: tuple[EffToxDesktopScenario, ...]
    newest_hyperparameters: str

    def to_xml(self) -> str:
        """Export normalized XML, retaining unknown fields and inactive settings."""
        return self.xml

    def saved_prior(self, *, monotone_toxicity: bool = True) -> EffToxPrior:
        """Use saved output only; stale output must not override elicitation input."""
        if self.newest_hyperparameters != "output":
            raise ValueError("NewestHyperParams=input requires prior calibration before execution")
        return EffToxPrior(
            np.asarray(self.prior_means), np.asarray(self.prior_sds), monotone_toxicity
        )

    def contour(self) -> EffToxContour:
        """Solve the current Lp contour accurately, not at legacy bisection tolerance."""
        e0, em, tm, t0 = self.contour_points
        return EffToxContour.from_points(e0, t0, em, tm)

    def observed_counts(self) -> np.ndarray:
        """Count complete outcomes at administered doses; pending contributes zero."""
        counts = np.zeros((len(self.doses), 2, 2), dtype=np.int64)
        for patient in self.patients:
            if patient.binary_outcome is not None:
                efficacy, toxicity = patient.binary_outcome
                counts[patient.administered_dose_index, efficacy, toxicity] += 1
        return counts


def read_efftox_desktop_xml(document: str) -> EffToxDesktopStudy:
    """Read the current six-coefficient binary study without executing content.

    DTDs/entities, duplicate scalar settings, invalid truth, and historical
    model formats are rejected. Unknown metadata is retained. No inference or
    calibration is performed by the reader.
    """
    if not isinstance(document, str) or len(document) > 2_000_000:
        raise ValueError("EffTox document must be a string of at most 2000000 characters")
    if "<!DOCTYPE" in document.upper() or "<!ENTITY" in document.upper():
        raise ValueError("XML DTDs and entities are not supported")
    try:
        root = ET.fromstring(document)
    except ET.ParseError as exc:
        raise ValueError("invalid EffTox XML") from exc
    if root.tag != "EffToxTrial" or root.get("ProgID") != _GUID:
        raise ValueError("only current EffTox 5.2.3 study XML is supported")
    if root.get("TypeOfTrial") not in {"Simulation", "TrialConduct"}:
        raise ValueError("unknown EffTox TypeOfTrial")
    for path in (".", "DoseFinderParams", "SimulationParams"):
        nodes = root.findall(path)
        if len(nodes) != 1:
            raise ValueError(f"expected exactly one EffTox {path}")
        tags = [c.tag for c in nodes[0] if c.tag != "ScenarioParams"]
        if len(tags) != len(set(tags)):
            raise ValueError("duplicate EffTox settings are ambiguous")
    design = root.find("DoseFinderParams")
    sim = root.find("SimulationParams")
    assert design is not None and sim is not None
    if _integer(design, "Model", 0, 2) != 1:
        raise ValueError("only the current six-coefficient binary EffTox model is supported")
    size = len(design.findall("Doses/DVal"))
    doses = _vector(design, "Doses/DVal", size)
    efftox_standardize(doses)
    means = _vector(design, "PriorMeans/MVal", 6)
    sds = _vector(design, "PriorStdDevs/SVal", 6)
    # Parsing must not silently impose an unresolved slope-distribution profile.
    EffToxPrior(np.asarray(means), np.asarray(sds), monotone_toxicity=False)
    raw_contour = _text(design, "BivariateBinaryContourPoints")
    if not raw_contour.startswith("[") or not raw_contour.endswith("]"):
        raise ValueError("invalid EffTox contour vector")
    points = tuple(float(v.strip()) for v in raw_contour[1:-1].split(","))
    if len(points) != 4:
        raise ValueError("EffTox contour requires four values")
    EffToxContour.from_points(points[0], points[3], points[1], points[2])
    maximum = _integer(root, "MaxQtyPatients", 1, 500)
    cohort_size = _integer(root, "CohortSize", 1, maximum)
    start = _integer(design, "StartingDoseIndex", 0, size - 1)
    _integer(root, "TrialIsShutDown", 0, 1)
    for name in (
        "ResponseCutoff",
        "ToxicityCutoff",
        "AcceptableProbOfLowTox",
        "AcceptableProbOfHighRes",
    ):
        if not 0 <= _number(design, name) <= 1:
            raise ValueError(f"EffTox {name} must be a probability")
    patients = []
    for index, cohort in enumerate(root.findall("Cohorts/Cohort"), 1):
        for node in cohort.findall("Patient"):
            patients.append(
                EffToxDesktopPatient(
                    _integer(node, "ID", 0, 2**31 - 1),
                    _text(node, "BeginDate"),
                    _integer(node, "DoseCalcedIndex", 0, size - 1),
                    _integer(node, "DoseAdminIndex", 0, size - 1),
                    _integer(node, "Outcome", -1, 3),
                    index,
                )
            )
    if len(patients) > maximum:
        raise ValueError("EffTox patient history exceeds MaxQtyPatients")
    if len({p.patient_id for p in patients}) != len(patients):
        raise ValueError("EffTox patient IDs must be unique")
    newest_nodes = sim.findall("NewestHyperParams")
    if len(newest_nodes) != 1 or newest_nodes[0].get("value") not in {"input", "output"}:
        raise ValueError("NewestHyperParams must explicitly select input or output")
    newest = str(newest_nodes[0].get("value"))
    dependent = bool(_integer(sim, "ProbsAreDependent", 0, 1))
    scenarios = []
    for node in sim.findall("ScenarioParams"):
        scenario = EffToxDesktopScenario(
            _probabilities(node, "TrueToxProbs/TVal", size),
            _probabilities(node, "TrueResProbs/RVal", size),
            _probabilities(node, "TrueResGivenNoToxProbs/RNVal", size),
            dependent,
        )
        scenario.joint_probabilities()
        scenarios.append(scenario)
    if len(scenarios) > 100 or (root.get("TypeOfTrial") == "Simulation" and not scenarios):
        raise ValueError("EffTox simulation requires 1..100 scenarios")
    if newest == "input":
        for path in ("ElicitedToxProbs/TVal", "ElicitedResProbs/RVal"):
            if any(not 0 < p < 1 for p in _probabilities(sim, path, size)):
                raise ValueError("elicited EffTox probabilities must lie strictly in (0,1)")
        if _number(sim, "TargetESS") <= 0:
            raise ValueError("EffTox TargetESS must be positive")
    return EffToxDesktopStudy(
        ET.tostring(root, encoding="unicode"),
        doses,
        means,
        sds,
        points,
        maximum,
        cohort_size,
        start,
        tuple(patients),
        tuple(scenarios),
        newest,
    )
