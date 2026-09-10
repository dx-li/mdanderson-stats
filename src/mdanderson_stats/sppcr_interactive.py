"""Bounded interactive SPPCR data entry over caller-owned streams."""

from typing import TextIO

import numpy as np

from .cdflib_console import CDFConsole, CDFConsoleError
from .sppcr_data import SPPCRData, sppcr_data


def read_sppcr_interactive(
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    *,
    unseen_alleles: str = "drop",
    max_attempts: int = 3,
    max_records: int = 10000,
    max_line_length: int = 10000,
) -> SPPCRData:
    """Read one experiment in native field order with bounded corrections.

    Uses native limits: 1..50 runs/alleles, sizes 1..999, genome DNA .001..1000,
    wells 1..500, and detections 0..wells. Invalid numeric vectors restart; duplicate
    labels or absent progenitors may be corrected. EOF/stream failures propagate.
    Numeric record limits apply per vector; supplied streams remain caller-owned.
    """
    if unseen_alleles not in ("drop", "retain"):
        raise ValueError("unseen_alleles must be 'drop' or 'retain'")
    console = CDFConsole(
        input_stream,
        output_stream,
        max_attempts=max_attempts,
        max_records=max_records,
        max_line_length=max_line_length,
    )
    console.write_message(
        "SPPCR data entry: each run has one genome-equivalent DNA amount.\n"
        "Alleles have distinct integer size labels. Lists may continue on later lines."
    )
    runs = int(console.get_numbers(dtype="int32", lo=1, hi=50, message="Number of runs (1..50):"))
    alleles = int(
        console.get_numbers(dtype="int32", lo=1, hi=50, message="Number of alleles (1..50):")
    )
    for _ in range(max_attempts):
        sizes = console.get_numbers(
            alleles,
            dtype="int32",
            lo=1,
            hi=999,
            message=f"Enter {alleles} distinct allele sizes (1..999):",
        )
        if len(np.unique(sizes)) == alleles:
            break
        console.write_message("Allele sizes must be distinct. Re-enter the complete size list.")
    else:
        raise CDFConsoleError("Too many duplicate allele-size lists")
    for _ in range(max_attempts):
        parents = console.get_numbers(
            2,
            dtype="int32",
            lo=1,
            hi=999,
            message="Enter two progenitor sizes; repeat one for a homozygote:",
        )
        if np.all(np.isin(parents, sizes)):
            break
        console.write_message("Both progenitor sizes must appear in the allele list. Try again.")
    else:
        raise CDFConsoleError("Too many invalid progenitor pairs")
    dna = console.get_numbers(
        runs,
        lo=0.001,
        hi=1000,
        message=f"Enter {runs} DNA amounts in genome equivalents (.001..1000):",
    )
    wells = console.get_numbers(
        runs, dtype="int32", lo=1, hi=500, message=f"Enter {runs} well counts (1..500):"
    )
    seen = []
    labels = " ".join(str(int(x)) for x in sizes)
    for i, n in enumerate(wells):
        seen.append(
            console.get_numbers(
                alleles,
                dtype="int32",
                lo=0,
                hi=int(n),
                message=(
                    f"Run {i + 1}: DNA {dna[i]:g} genome equivalents, {int(n)} wells.\n"
                    f"Allele order: {labels}\nEnter {alleles} detection counts (0..{int(n)}):"
                ),
            )
        )
    data = sppcr_data(
        dna, seen, wells, sizes, (int(parents[0]), int(parents[1])), unseen_alleles=unseen_alleles
    )
    if data.omitted_allele_sizes:
        console.write_message(
            "Omitted never-seen alleles: " + " ".join(map(str, data.omitted_allele_sizes))
        )
    return data
