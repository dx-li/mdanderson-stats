"""Saved single-agent/TITE final data, with explicit complete-outcome certification."""

import xml.etree.ElementTree as ET

import numpy as np

from .boin import BOINDesign, BOINSelection, _owned
from .boin_desktop import (
    BOINDesktopStudy,
    _boolean,
    _integer,
    _number,
    desktop_bound_probabilities,
    read_boin_desktop_project,
)


def desktop_single_design(study: BOINDesktopStudy) -> BOINDesign:
    """Resolve active single-agent bounds; final selection is unbounded natively."""
    if study.mode not in ("single", "tite") or study.shape[1] != 1:
        raise ValueError("single-agent/TITE saved study with one drug-B column required")
    design = ET.fromstring(study.xml).find("SimDat/TrialDesign")
    assert design is not None
    safe, toxic = desktop_bound_probabilities(study)
    return BOINDesign(
        target=study.target,
        safe_probability=safe,
        toxic_probability=toxic,
        elimination_probability=_number(design, "EliminateDoseThreshold"),
        extra_safe=_boolean(design, "ExtraSafe"),
        safety_offset=_number(design, "Offset"),
        early_stop_patients=_integer(design, "EarlyStop", 3, 1000),
        stay_at_one_of_three=study.run_in,
        bound_mtd=False,
    )


def boin_desktop_single_counts(study: BOINDesktopStudy) -> tuple[np.ndarray, np.ndarray]:
    """Read native row-major int[]: all patient counts, then all DLT counts.

    The original file does not encode outcome completeness in this array.
    Reading it alone does not establish eligibility for final MTD selection.
    """
    design = desktop_single_design(study)
    nodes = ET.fromstring(study.xml).findall("SimDat/TrialData")
    if len(nodes) != 1 or any(child.tag != "int" for child in nodes[0]):
        raise ValueError("expected one single-agent TrialData integer array")
    values = []
    for node in nodes[0]:
        try:
            value = int(node.text or "")
        except ValueError as exc:
            raise ValueError("TrialData values must be integers") from exc
        values.append(value)
    if len(values) != 2 * study.shape[0]:
        raise ValueError("TrialData must contain two rows for every dose")
    n, y = np.asarray(values).reshape(2, study.shape[0])
    design._state(n, y, None)
    if n.sum() > 1000:
        raise ValueError("at most 1000 retained patients supported")
    return _owned(n), _owned(y)


def set_boin_desktop_single_counts(
    study: BOINDesktopStudy, patients, toxicities
) -> BOINDesktopStudy:
    """Replace the native count array while retaining settings and other metadata."""
    design = desktop_single_design(study)
    n, y, _, _ = design._state(patients, toxicities, None)
    if len(n) != study.shape[0] or n.sum() > 1000:
        raise ValueError("counts must match saved doses with at most 1000 patients")
    root = ET.fromstring(study.xml)
    sim = root.find("SimDat")
    assert sim is not None
    existing = sim.findall("TrialData")
    if len(existing) > 1:
        raise ValueError("duplicate TrialData arrays")
    node = existing[0] if existing else ET.SubElement(sim, "TrialData")
    node.clear()
    for value in np.concatenate((n, y)):
        ET.SubElement(node, "int").text = str(int(value))
    return read_boin_desktop_project(ET.tostring(root, encoding="unicode"))


def select_boin_desktop_single(
    study: BOINDesktopStudy, *, outcomes_complete: bool
) -> BOINSelection:
    """Select from saved counts only after explicitly confirming no outcomes pending.

    Native 1e-10 perturbation is replaced by mathematical tie ordering in the
    shared selector; near-tie floating-point identity is not promised.
    """
    if outcomes_complete is not True:
        raise ValueError("final selection requires explicit complete-outcome confirmation")
    if any(patient.outcome == 2 for patient in study.patients):
        raise ValueError("saved patient history still contains pending outcomes")
    n, y = boin_desktop_single_counts(study)
    return desktop_single_design(study).select_mtd(n, y)
