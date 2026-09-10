"""Bounded SPPCR input reads and staged report-file publication."""

import os
import tempfile
from pathlib import Path
from typing import Literal

from .cdflib_console import _positive
from .sppcr_analysis import SPPCRReports
from .sppcr_batch import parse_sppcr_batch
from .sppcr_data import SPPCRData
from .sppcr_filemaker import parse_sppcr_filemaker


def read_sppcr_file(
    path: str | Path,
    *,
    file_format: Literal["batch", "filemaker"] = "batch",
    unseen_alleles: str = "drop",
    max_characters: int = 1000000,
    max_runs: int = 50,
    max_alleles: int | None = None,
) -> SPPCRData:
    """Read and close one UTF-8 input file, then apply the native-format parser.

    At most max_characters+1 characters are read; oversize input is rejected.
    Default allele limits are 50 for batch and 25 for FileMaker. Limits can be
    explicitly increased as with the parsers. Decode, IO and parsing errors
    propagate; no files are modified or output files opened.
    """
    if file_format not in ("batch", "filemaker"):
        raise ValueError("file_format must be 'batch' or 'filemaker'")
    if unseen_alleles not in ("drop", "retain"):
        raise ValueError("unseen_alleles must be 'drop' or 'retain'")
    if max_alleles is None:
        max_alleles = 50 if file_format == "batch" else 25
    for value, name in [
        (max_characters, "max_characters"),
        (max_runs, "max_runs"),
        (max_alleles, "max_alleles"),
    ]:
        _positive(value, name)
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        text = stream.read(max_characters + 1)
    if len(text) > max_characters:
        raise ValueError("SPPCR input exceeds max_characters")
    parse = parse_sppcr_batch if file_format == "batch" else parse_sppcr_filemaker
    return parse(
        text,
        unseen_alleles=unseen_alleles,
        max_runs=max_runs,
        max_alleles=max_alleles,
        max_characters=max_characters,
    )


def _stage(destination: Path, payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=".sppcr-", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def write_sppcr_reports(
    reports: SPPCRReports,
    report_path: str | Path,
    simulation_path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path | None]:
    """Publish UTF-8 reports from complete staged files; return absolute paths.

    Simulation text and a simulation path must either both be present or absent.
    Destinations must differ, including aliases. Parents must already exist.
    Existing destinations are preserved unless overwrite=True. All text is encoded
    and all files staged before publishing either. Each publication is atomic;
    the pair is NOT a transaction. A later race/IO failure may leave an earlier
    complete output published. Existing files are not backed up. Concurrent writers
    are not coordinated. Replacement replaces a symlink itself, not its target.
    """
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be boolean")
    if not isinstance(reports, SPPCRReports) or not isinstance(reports.report, str):
        raise TypeError("reports must contain report text")
    if (reports.simulations is None) != (simulation_path is None):
        raise ValueError("simulation text and simulation_path must both be present or absent")
    destination = Path(report_path).absolute()
    simulation = Path(simulation_path).absolute() if simulation_path is not None else None
    paths = [destination]
    payloads = [reports.report.encode("utf-8")]
    if simulation is not None:
        if not isinstance(reports.simulations, str):
            raise TypeError("simulations must contain text")
        if destination.resolve() == simulation.resolve() or (
            destination.exists() and simulation.exists() and destination.samefile(simulation)
        ):
            raise ValueError("report and simulation destinations must differ")
        paths.append(simulation)
        payloads.append(reports.simulations.encode("utf-8"))
    for path in paths:
        if path.is_dir():
            raise IsADirectoryError(str(path))
        if not overwrite and os.path.lexists(path):
            raise FileExistsError(str(path))
    staged: list[Path] = []
    try:
        for path, payload in zip(paths, payloads, strict=True):
            staged.append(_stage(path, payload))
        for temporary, path in zip(staged, paths, strict=True):
            if overwrite:
                os.replace(temporary, path)
            else:
                # Atomic exclusive publication: a competing new file is never truncated.
                os.link(temporary, path)
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)
    return destination, simulation
