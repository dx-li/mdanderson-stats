"""Native standalone combination final-data arrays, separate from patient conduct."""

import xml.etree.ElementTree as ET

import numpy as np

from .boin import _owned
from .boin_combination import _validate_counts
from .boin_desktop import read_boin_desktop_project
from .boin_desktop_controller import desktop_combination_design
from .boin_desktop_selection import select_boin_desktop_mtd

_FIELDS = ("ComboTrialData_NEvaluable", "ComboTrialData_NDLT")


def _counts(study, patients, toxicities):
    desktop_combination_design(study)
    n, y = _validate_counts(patients, toxicities)
    if n.shape != study.shape or n.sum() > 1000:
        raise ValueError("combination counts must match saved grid with at most 1000 patients")
    return n, y


def boin_desktop_combination_counts(study):
    """Read drug-A rows, drug-B columns from the native jagged int arrays."""
    sim = ET.fromstring(study.xml).find("SimDat")
    assert sim is not None
    values = []
    for field in _FIELDS:
        nodes = sim.findall(field)
        if len(nodes) != 1 or any(row.tag != "ArrayOfInt" for row in nodes[0]):
            raise ValueError("expected one native jagged integer array for " + field)
        rows = []
        for row in nodes[0]:
            if any(cell.tag != "int" for cell in row):
                raise ValueError("native combination cells must be int elements")
            try:
                rows.append([int(cell.text or "") for cell in row])
            except ValueError as exc:
                raise ValueError("native combination values must be integers") from exc
        if len(rows) != study.shape[0] or any(len(row) != study.shape[1] for row in rows):
            raise ValueError("native combination arrays do not match saved grid")
        values.append(np.asarray(rows))
    return tuple(_owned(v) for v in _counts(study, *values))


def set_boin_desktop_combination_counts(study, patients, toxicities):
    """Replace standalone final counts while retaining unrelated native metadata."""
    n, y = _counts(study, patients, toxicities)
    root = ET.fromstring(study.xml)
    sim = root.find("SimDat")
    assert sim is not None
    for field, counts in zip(_FIELDS, (n, y), strict=True):
        nodes = sim.findall(field)
        if len(nodes) > 1:
            raise ValueError("duplicate native combination arrays")
        node = nodes[0] if nodes else ET.SubElement(sim, field)
        node.clear()
        for row in counts:
            array = ET.SubElement(node, "ArrayOfInt")
            for value in row:
                ET.SubElement(array, "int").text = str(int(value))
    return read_boin_desktop_project(ET.tostring(root, encoding="unicode"))


def select_boin_desktop_combination_counts(study, *, outcomes_complete, profile="safe"):
    """Analyze standalone final counts only after explicit completeness confirmation."""
    if outcomes_complete is not True or any(p.outcome == 2 for p in study.patients):
        raise ValueError("final selection requires complete outcomes without pending history")
    n, y = boin_desktop_combination_counts(study)
    return select_boin_desktop_mtd(
        desktop_combination_design(study), n, y, profile=profile, contour=study.mode == "contour"
    )
