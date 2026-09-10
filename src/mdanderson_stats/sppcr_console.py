"""Repeated SPPCR menu analyses over caller-owned streams."""

from dataclasses import dataclass
from typing import Literal, TextIO

import numpy as np

from ._validation import scalar
from .cdflib_console import CDFConsole, CDFConsoleError, _positive
from .randlib import RandlibGenerator
from .sppcr_analysis import SPPCRAnalysis, format_sppcr_analysis, sppcr_analyze, sppcr_simulate
from .sppcr_bootstrap import _bootstrap_options
from .sppcr_files import read_sppcr_file
from .sppcr_interactive import read_sppcr_interactive
from .sppcr_truth_console import read_sppcr_truth


@dataclass(frozen=True)
class SPPCRRun:
    """Menu outcome and last completed analysis; partial input is never analyzed."""

    reason: Literal["exit", "eof"]
    completed: int
    rejected: int
    last_analysis: SPPCRAnalysis | None


def run_sppcr(
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    *,
    rng: np.random.Generator | RandlibGenerator,
    report_stream: TextIO | None = None,
    simulation_stream: TextIO | None = None,
    write_simulations: bool | None = None,
    replicates: int = 1000,
    saturation: str = "half",
    unseen_alleles: str = "drop",
    precision: int = 10,
    multiplier: float = 1.959964,
    max_steps: int = 1000,
    max_attempts: int = 3,
    max_records: int = 10000,
    max_line_length: int = 10000,
    max_file_characters: int = 1000000,
    max_report_characters: int = 1000000,
    max_simulation_characters: int = 10000000,
) -> SPPCRRun:
    """Run the native 0..4 menu with bounded retries and explicit random state.

    1 reads FileMaker input, 2 batch input, 3 interactive data, 4 truth generation.
    Each completed report goes to output and optionally report_stream. Requested
    replicate text goes to simulation_stream or output if no separate stream was
    supplied. Output streams contain labeled experiment sections across sessions.
    EOF ends cleanly with prior results retained. Invalid data/numerical analyses
    return to the menu; exhausted console limits and output failures propagate.
    Streams remain caller-owned; only input files opened here are closed here.
    """
    _bootstrap_options(rng, replicates, saturation)
    if unseen_alleles not in ("drop", "retain"):
        raise ValueError("unseen_alleles must be 'drop' or 'retain'")
    if write_simulations is not None and not isinstance(write_simulations, bool):
        raise ValueError("write_simulations must be boolean or None")
    for value, name in [
        (precision, "precision"),
        (max_steps, "max_steps"),
        (max_file_characters, "max_file_characters"),
        (max_report_characters, "max_report_characters"),
        (max_simulation_characters, "max_simulation_characters"),
    ]:
        _positive(value, name)
    if precision > 17 or scalar(multiplier, "multiplier") <= 0:
        raise ValueError("precision must not exceed 17 and multiplier must be positive")
    console = CDFConsole(
        input_stream,
        output_stream,
        max_attempts=max_attempts,
        max_records=max_records,
        max_line_length=max_line_length,
    )
    if any(sink is console.input for sink in (console.output, report_stream, simulation_stream)):
        raise ValueError("input stream must differ from output streams")
    completed = rejected = 0
    last: SPPCRAnalysis | None = None
    console.write_message("SPPCR: small-pool PCR allele-frequency analysis")
    for _ in range(max_steps):
        try:
            action = int(
                console.get_numbers(
                    dtype="int32",
                    lo=0,
                    hi=4,
                    message=(
                        "0 exit; 1 FileMaker file; 2 batch file; "
                        "3 enter data; 4 simulate from truth"
                    ),
                )
            )
            if action == 0:
                return SPPCRRun("exit", completed, rejected, last)
            try:
                if action in (1, 2):
                    name = console.get_string(
                        "Enter input file path (back returns to menu):"
                    ).strip()
                    if name.lower() == "quit":
                        return SPPCRRun("exit", completed, rejected, last)
                    if name.lower() == "back":
                        continue
                    try:
                        data = read_sppcr_file(
                            name,
                            file_format="filemaker" if action == 1 else "batch",
                            unseen_alleles=unseen_alleles,
                            max_characters=max_file_characters,
                        )
                    except OSError as error:
                        rejected += 1
                        console.write_message("Cannot read input file: " + str(error))
                        continue
                elif action == 3:
                    data = read_sppcr_interactive(
                        console.input,
                        console.output,
                        unseen_alleles=unseen_alleles,
                        max_attempts=max_attempts,
                        max_records=max_records,
                        max_line_length=max_line_length,
                    )
                else:
                    request = read_sppcr_truth(
                        console.input,
                        console.output,
                        max_attempts=max_attempts,
                        max_records=max_records,
                        max_line_length=max_line_length,
                    )
                if action == 4:
                    analysis = sppcr_simulate(
                        request, rng=rng, replicates=replicates, saturation=saturation
                    )
                else:
                    analysis = sppcr_analyze(
                        data, rng=rng, replicates=replicates, saturation=saturation
                    )
            except (ValueError, ArithmeticError) as error:
                rejected += 1
                console.write_message("Analysis rejected: " + str(error))
                continue
        except EOFError:
            console.write_message("End of input.")
            return SPPCRRun("eof", completed, rejected, last)
        # Formatting and output failures are not retried as input errors.
        reports = format_sppcr_analysis(
            analysis,
            write_simulations=write_simulations,
            precision=precision,
            multiplier=multiplier,
            max_report_characters=max_report_characters,
            max_simulation_characters=max_simulation_characters,
        )
        heading = f"SPPCR experiment {completed + 1}\n"
        console.output.write(heading + reports.report)
        if report_stream is not None and report_stream is not console.output:
            report_stream.write(heading + reports.report)
        if reports.simulations is not None:
            sink = simulation_stream if simulation_stream is not None else console.output
            sink.write(heading + reports.simulations)
        completed += 1
        last = analysis
    raise CDFConsoleError("SPPCR exhausted max_steps")
