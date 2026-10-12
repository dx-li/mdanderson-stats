"""BOIN desktop 1.1.0 saved-project interchange with retained inactive metadata.

Native projects embed XML between GUID markers in an HTML comment. Original
HTML/templates are reference inputs, not package assets. Dates are retained as
literal strings: native short-date serialization depends on machine culture.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

_GUID = "E45C3723-2D8F-40F8-8A47-AEF4DFD489E7"
_OUTCOMES = {"tslNO": 0, "tslYES": 1, "tslUNKNOWN": 2}


def _text(node: ET.Element, path: str) -> str:
    found = node.findall(path)
    if len(found) != 1 or found[0].text is None:
        raise ValueError(f"expected one BOIN {path}")
    return found[0].text.strip()


def _number(node: ET.Element, path: str) -> float:
    value = float(_text(node, path))
    if not np.isfinite(value):
        raise ValueError(f"BOIN {path} must be finite")
    return value


def _integer(node: ET.Element, path: str, low: int, high: int) -> int:
    value = _number(node, path)
    if value != int(value) or not low <= value <= high:
        raise ValueError(f"BOIN {path} must be an integer in [{low},{high}]")
    return int(value)


def _boolean(node: ET.Element, path: str) -> bool:
    value = _text(node, path)
    if value not in ("true", "false", "1", "0"):
        raise ValueError(f"BOIN {path} must be an XML boolean")
    return value in ("true", "1")


@dataclass(frozen=True)
class BOINDesktopPatient:
    """Native patient codes: 0 no DLT, 1 DLT, 2 pending; doses are one-based."""

    patient_id: int
    cohort: int
    dose: tuple[int, int]
    outcome: int
    date_entered: str
    date_updated: str
    fingerprint: int


@dataclass(frozen=True)
class BOINDesktopScenario:
    name: str
    probabilities: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class BOINDesktopStudy:
    """Validated active structure plus the complete semantically retained XML."""

    xml: str
    mode: str
    shape: tuple[int, int]
    cohort_size: int
    maximum_patients: int
    start_dose: tuple[int, int]
    target: float
    run_in: bool
    titration: bool
    titration_cap: int
    patients: tuple[BOINDesktopPatient, ...]
    scenarios: tuple[BOINDesktopScenario, ...]
    repetitions: int
    seed: int
    subtrial_maximums: tuple[int, ...] = ()

    def counts(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return enrolled, DLT and pending counts at actual administered cells."""
        n, y, pending = (np.zeros(self.shape, dtype=np.int64) for _ in range(3))
        for patient in self.patients:
            cell = patient.dose[0] - 1, patient.dose[1] - 1
            n[cell] += 1
            y[cell] += patient.outcome == 1
            pending[cell] += patient.outcome == 2
        for array in (n, y, pending):
            array.setflags(write=False)
        return n, y, pending

    def to_html(self, report_html: str = "<p>Saved BOIN study</p>") -> str:
        """Embed retained XML after a supplied report; no original template is used.

        The caller owns the report HTML. XML text is made comment-safe; this is
        semantic project interchange, not native layout or file-byte identity.
        """
        if not isinstance(report_html, str) or len(report_html) > 10_000_000:
            raise ValueError("report_html must be a string of at most 10000000 characters")
        if f"begin{_GUID}" in report_html or f"end{_GUID}" in report_html:
            raise ValueError("report already contains BOIN project markers")
        xml = self.xml.replace("--", "&#45;&#45;")
        return f"{report_html}\n<!--\nbegin{_GUID}\n{xml}\nend{_GUID}\n-->\n"


def read_boin_desktop_project(document: str) -> BOINDesktopStudy:
    """Read current project XML or its native HTML envelope, preserving metadata.

    Limits are Python resource bounds, not claims about the original GUI limits.
    Historical/foreign ProgIDs are rejected; no implicit model migration occurs.
    """
    if not isinstance(document, str) or len(document) > 10_000_000:
        raise ValueError("BOIN document must be a string of at most 10000000 characters")
    begin, end = f"begin{_GUID}", f"end{_GUID}"
    if begin in document or end in document:
        if document.count(begin) != 1 or document.count(end) != 1:
            raise ValueError("expected one BOIN project marker pair")
        marker_start, stop = document.index(begin) + len(begin), document.index(end)
        if stop <= marker_start:
            raise ValueError("reversed BOIN project markers")
        document = document[marker_start:stop].strip()
    if "<!DOCTYPE" in document.upper() or "<!ENTITY" in document.upper():
        raise ValueError("BOIN XML declarations/entities are unsupported")
    try:
        root = ET.fromstring(document.lstrip("\ufeff"))
    except ET.ParseError as exc:
        raise ValueError("invalid BOIN XML") from exc
    if root.tag != "Model" or root.get("ProgID") != _GUID:
        raise ValueError("expected current BOIN Model ProgID")
    sim_nodes = root.findall("SimDat")
    if len(sim_nodes) != 1 or len(sim_nodes[0].findall("TrialDesign")) != 1:
        raise ValueError("expected one BOIN SimDat/TrialDesign")
    sim = sim_nodes[0]
    design = sim.find("TrialDesign")
    assert design is not None
    single = _boolean(design, "SingleDrugTrial")
    mode = (
        ("tite" if _boolean(design, "TITE_BOINTrial") else "single")
        if single
        else ("contour" if _boolean(design, "FindContours") else "combination")
    )
    shape = (
        _integer(design, "NumberOfDoses_DrugA", 1, 100),
        _integer(design, "NumberOfDoses_DrugB", 1, 100),
    )
    if np.prod(shape) > 1000 or (not single and (min(shape) < 2 or shape[0] > shape[1])):
        raise ValueError("BOIN dose grid exceeds supported bounds")
    size = _integer(design, "CohortSize", 1, 1000)
    maximum = _integer(design, "MaximumSampleSize", 1, 1000)
    start = (
        _integer(design, "StartingDoseLevel_DrugA", 1, shape[0]),
        _integer(design, "StartingDoseLevel_DrugB", 1, shape[1]),
    )
    target = _number(design, "TargetToxicityProbability")
    if not 0.05 <= target <= 0.6:
        raise ValueError("BOIN target must be in [.05,.6]")
    # Original DoingThreePlusThreeRunIn applies the GUI eligibility gate.
    run_in = _boolean(design, "UseThreePlusThreeRunIn") and target == 0.25 and size == 3
    titration = _boolean(design, "UseAcceleratedTitration")
    cap = _integer(design, "AcceleratedTitrationCap_DrugA", 1, 1000)
    for name in ("Patients", "Scenarios"):
        if len(sim.findall(name)) != 1:
            raise ValueError(f"expected one BOIN {name} container")
    subtrial_maximums: tuple[int, ...] = ()
    if mode == "contour":
        values = tuple(
            float(node.text or "") for node in design.findall("MaximumSampleSizeForSubTrials/int")
        )
        if (
            len(values) != shape[0]
            or not np.isfinite(values).all()
            or any(v != int(v) or v < 1 or int(v) % size for v in values)
            or sum(values) > 1000
        ):
            raise ValueError(
                "contour subtrial maxima must be positive cohort multiples, "
                "one per row, total <=1000"
            )
        subtrial_maximums = tuple(int(v) for v in values)
    patients = []
    for item in sim.findall("Patients/Patient"):
        if _text(item, "OutcomeCode") not in _OUTCOMES:
            raise ValueError("BOIN OutcomeCode must be tslNO, tslYES or tslUNKNOWN")
        dose = (
            _integer(item, "DoseIndexA", 0, shape[0] - 1) + 1,
            _integer(item, "DoseIndexB", 0, shape[1] - 1) + 1,
        )
        if (
            item.find("DoseLevelB") is not None
            and _integer(item, "DoseLevelB", 1, shape[1]) != dose[1]
        ):
            raise ValueError("inconsistent BOIN DoseLevelB/DoseIndexB")
        patients.append(
            BOINDesktopPatient(
                _integer(item, "PID", 0, 2**31 - 1),
                _integer(item, "Cohort", 1, 1000),
                dose,
                _OUTCOMES[_text(item, "OutcomeCode")],
                _text(item, "DateEntered"),
                _text(item, "DateUpdated"),
                _integer(item, "Fingerprint", -(2**31), 2**31 - 1),
            )
        )
    if len(patients) > 1000 or len({p.patient_id for p in patients}) != len(patients):
        raise ValueError("BOIN patients exceed limit or have duplicate IDs")
    if any(b.cohort < a.cohort for a, b in zip(patients, patients[1:])):
        raise ValueError("BOIN patient cohorts must be ordered")
    scenarios = []
    for item in sim.findall("Scenarios/Scenario"):
        rows = _integer(item, "RowsPerScenario", 1, 100)
        values = tuple(float(x.text or "") for x in item.findall("TrueProbabilities/double"))
        if rows != shape[0] or len(values) != np.prod(shape) or not np.isfinite(values).all():
            raise ValueError("BOIN scenario shape/values must match the design")
        if any(not 0 <= v <= 1 for v in values):
            raise ValueError("BOIN truth probabilities must lie in [0,1]")
        scenarios.append(
            BOINDesktopScenario(
                _text(item, "Name"),
                tuple(tuple(values[i : i + shape[1]]) for i in range(0, len(values), shape[1])),
            )
        )
    if len(scenarios) > 100:
        raise ValueError("BOIN supports at most 100 saved scenarios")
    return BOINDesktopStudy(
        ET.tostring(root, encoding="unicode"),
        mode,
        shape,
        size,
        maximum,
        start,
        target,
        run_in,
        titration,
        cap,
        tuple(patients),
        tuple(scenarios),
        _integer(sim, "NumberOfSimulations", 1, 1_000_000),
        _integer(sim, "Seed", 0, 2**31 - 1),
        subtrial_maximums,
    )


def desktop_bound_probabilities(study: BOINDesktopStudy) -> tuple[float, float]:
    """Resolve the native flag before consulting retained inactive numeric bounds."""
    design = ET.fromstring(study.xml).find("SimDat/TrialDesign")
    assert design is not None
    if _boolean(design, "UseDefaultBounds"):
        return 0.6 * study.target, 1.4 * study.target
    return (
        _number(design, "AcceptableToxicity_LowerBound"),
        _number(design, "AcceptableToxicity_UpperBound"),
    )
