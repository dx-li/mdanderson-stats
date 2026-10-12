"""Resolve current EffTox saved elicitation inputs with explicit Python calibration."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

from .efftox_calibration import EffToxCalibration, calibrate_efftox_prior
from .efftox_desktop import EffToxDesktopStudy, read_efftox_desktop_xml


@dataclass(frozen=True)
class EffToxDesktopCalibration:
    """Calibration evidence and executable output, only when numerical checks pass.

    A converged optimizer can have nonzero target residuals or active bounds;
    those remain in ``calibration`` and require scientific interpretation.
    No native optimizer identity or clinical validation is asserted.
    """

    study: EffToxDesktopStudy | None
    calibration: EffToxCalibration


def calibrate_efftox_desktop(
    study: EffToxDesktopStudy,
    *,
    monotone_toxicity: bool,
    max_iterations: int = 1500,
    max_evaluations: int = 2000,
) -> EffToxDesktopCalibration:
    """Calibrate newer elicitation inputs and preserve all other saved settings.

    The saved curvature and association priors remain explicit fixed inputs to
    calibration. A failed optimizer or quadrature check returns ``study=None``;
    stale hyperparameters are never relabeled as executable output.
    """
    if study.patients:
        raise ValueError("cannot change the prior after patient accrual; clear the history first")
    if study.newest_hyperparameters != "input":
        raise ValueError("calibration requires NewestHyperParams=input")
    root = ET.fromstring(study.xml)
    sim = root.find("SimulationParams")
    assert sim is not None
    efficacy = [float(n.text or "nan") for n in sim.findall("ElicitedResProbs/RVal")]
    toxicity = [float(n.text or "nan") for n in sim.findall("ElicitedToxProbs/TVal")]
    calibration = calibrate_efftox_prior(
        study.doses,
        efficacy,
        toxicity,
        target_ess=float(sim.findtext("TargetESS", "nan")),
        curvature_mean=study.prior_means[4],
        curvature_sd=study.prior_sds[4],
        association_mean=study.prior_means[5],
        association_sd=study.prior_sds[5],
        monotone_toxicity=monotone_toxicity,
        max_iterations=max_iterations,
        max_evaluations=max_evaluations,
    )
    if not (
        calibration.efficacy_optimizer_success
        and calibration.toxicity_optimizer_success
        and calibration.quadrature_converged
    ):
        return EffToxDesktopCalibration(None, calibration)
    for path, values in [
        ("DoseFinderParams/PriorMeans/MVal", calibration.prior.mean),
        ("DoseFinderParams/PriorStdDevs/SVal", calibration.prior.sd),
    ]:
        for node, value in zip(root.findall(path), values, strict=True):
            node.text = repr(float(value))
    newest = sim.find("NewestHyperParams")
    assert newest is not None
    newest.set("value", "output")
    metadata = root.find("PythonEffToxCalibration")
    if metadata is None:
        metadata = ET.SubElement(root, "PythonEffToxCalibration")
    metadata.attrib.update(
        {
            "monotone_toxicity": str(bool(monotone_toxicity)).lower(),
            "efficacy_objective": repr(calibration.efficacy_objective),
            "toxicity_objective": repr(calibration.toxicity_objective),
            "quadrature_error": repr(calibration.quadrature_error),
            "efficacy_boundary_hits": ",".join(calibration.efficacy_boundary_hits),
            "toxicity_boundary_hits": ",".join(calibration.toxicity_boundary_hits),
            "method": "Python deterministic quadrature; not native optimizer identity",
        }
    )
    return EffToxDesktopCalibration(
        read_efftox_desktop_xml(ET.tostring(root, encoding="unicode")), calibration
    )
