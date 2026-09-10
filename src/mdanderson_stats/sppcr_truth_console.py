"""SPPCR truth-parameter entry and explicit simulation choices."""

from dataclasses import dataclass
from typing import TextIO

import numpy as np

from .cdflib_console import CDFConsole, CDFConsoleError
from .sppcr_truth import SPPCRTruth, sppcr_truth


@dataclass(frozen=True)
class SPPCRSimulationRequest:
    """Validated truth design and choices, without sampling or opening files."""

    truth: SPPCRTruth
    bootstrap_from_truth: bool
    write_simulations: bool


def read_sppcr_truth(
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    *,
    max_attempts: int = 3,
    max_records: int = 10000,
    max_line_length: int = 10000,
) -> SPPCRSimulationRequest:
    """Read native-order truth parameters over caller-owned streams.

    Dimensions are bounded to 1..50; wells 1..1000, DNA .001..10000 and
    calibration .001..1000 follow native prompts. Weights must be nonnegative
    with positive total. Input progenitors are one-based, returned zero-based.
    Invalid numeric vectors restart; all-zero weight lists have bounded retries.
    EOF, exhausted limits, stream errors and numerical underflow propagate.
    """
    console = CDFConsole(
        input_stream,
        output_stream,
        max_attempts=max_attempts,
        max_records=max_records,
        max_line_length=max_line_length,
    )
    dimensions = console.get_numbers(
        2,
        dtype="int32",
        lo=1,
        hi=50,
        message="Enter DNA-level count and allele count (each 1..50):",
    )
    runs, alleles = map(int, dimensions)
    wells = int(
        console.get_numbers(
            dtype="int32",
            lo=1,
            hi=1000,
            message="Common wells per run (1..1000):",
        )
    )
    for _ in range(max_attempts):
        weights = console.get_numbers(
            alleles,
            lo=0,
            message=f"Enter {alleles} nonnegative allele weights to normalize:",
        )
        if np.any(weights > 0):
            break
        console.write_message("At least one weight must be positive. Re-enter all weights.")
    else:
        raise CDFConsoleError("Too many all-zero allele-weight lists")
    dna = console.get_numbers(
        runs,
        lo=0.001,
        hi=10000,
        message=f"Enter {runs} model DNA amounts (.001..10000; used directly):",
    )
    calibration = float(
        console.get_numbers(
            lo=0.001,
            hi=1000,
            message="Calibration (.001..1000):",
        )
    )
    parents = console.get_numbers(
        2,
        dtype="int32",
        lo=1,
        hi=alleles,
        message="Enter two one-based progenitor indices; repeat one for a homozygote:",
    )
    truth = sppcr_truth(
        dna,
        wells,
        weights,
        calibration,
        progenitor=(int(parents[0]) - 1, int(parents[1]) - 1),
    )
    from_truth = console.get_yn(
        "Bootstrap from supplied truth instead of observed fractions? (y/n)"
    )
    simulations = console.get_yn("Request replicate estimates for output? (y/n)")
    return SPPCRSimulationRequest(truth, from_truth, simulations)
