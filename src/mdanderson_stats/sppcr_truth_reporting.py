"""Bounded truth-parameter reports with explicit DNA units and choices."""

import numpy as np

from .sppcr_generate import sppcr_detection_probabilities
from .sppcr_reporting import _Report
from .sppcr_truth import sppcr_truth
from .sppcr_truth_console import SPPCRSimulationRequest


def format_sppcr_truth(
    request: SPPCRSimulationRequest,
    *,
    precision: int = 10,
    max_characters: int = 1000000,
) -> str:
    """Format native parameter-report quantities plus DNA, means and probabilities.

    Allele and progenitor indices in the report are one-based. Model DNA is used
    directly. Every run's wells are listed, including unequal-well designs made
    through sppcr_truth. No streams, files or random state are touched.
    """
    report = _Report(precision, max_characters)
    if not isinstance(request.bootstrap_from_truth, bool) or not isinstance(
        request.write_simulations, bool
    ):
        raise ValueError("Simulation choices must be boolean")
    t = request.truth
    # Validate public dataclass contents without silently replacing reported values.
    sppcr_truth(t.dna, t.wells, t.frequency, t.calibration, progenitor=t.progenitor)
    if (
        not np.isclose(np.sum(t.frequency), 1, rtol=0, atol=8 * np.finfo(float).eps)
        or not np.array_equal(t.mu, t.calibration * t.frequency)
        or not np.array_equal(t.probability, sppcr_detection_probabilities(t.dna, t.mu))
    ):
        raise ValueError("Truth design has inconsistent frequencies, means or probabilities")
    report.row("SPPCR TRUTH PARAMETERS")
    report.row("dna_levels", str(t.dna.size))
    report.row("alleles", str(t.frequency.size))
    report.row("calibration", report.number(t.calibration))
    report.row("progenitor_indices_1_based", *(str(x + 1) for x in t.progenitor))
    report.row("bootstrap_model", "truth" if request.bootstrap_from_truth else "observed_fractions")
    report.row("write_simulations", str(request.write_simulations).lower())
    report.row("ALLELES")
    report.row("index_1_based", "frequency", "mu")
    for j, (frequency, mu) in enumerate(zip(t.frequency, t.mu, strict=True), 1):
        report.row(str(j), report.number(frequency), report.number(mu))
    report.row("DESIGN")
    report.row(
        "run_1_based",
        "model_dna",
        "wells",
        *(f"probability_{j + 1}" for j in range(t.frequency.size)),
    )
    for i, (dna, wells, probability) in enumerate(
        zip(t.dna, t.wells, t.probability, strict=True), 1
    ):
        report.row(
            str(i),
            report.number(dna),
            report.number(wells),
            *(report.number(x) for x in probability),
        )
    return report.finish()
