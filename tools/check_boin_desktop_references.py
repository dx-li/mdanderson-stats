"""Regenerate synthetic desktop references against hash-pinned external R files."""

import argparse
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("original_root", type=Path, help="extracted original directory containing R/")
parser.add_argument("--r-executable", default="R")
parser.add_argument("--r-home")
parser.add_argument("--library-path")
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
environment = os.environ.copy()
if args.r_home:
    environment["R_HOME"] = args.r_home
if args.library_path:
    environment["LD_LIBRARY_PATH"] = args.library_path
references = (
    ("selection", "selectMTD_BOINComb_updated.R", "selection", True),
    ("simulation_extended", "Functions for BOIN_Comb.R", "simulation-extended", True),
    ("single_selection", None, "single-selection", False),
    ("edited_history", None, "edited-history", False),
    ("tite_calendar", None, "tite-calendar", False),
    ("single_simulation", None, "single-simulation", False),
    ("tite_calendar_extended", None, "tite-calendar-extended", False),
    ("rolling_calendar", None, "rolling-calendar", False),
    ("three_plus_three", None, "three-plus-three", False),
    ("single_edited", None, "single-edited", False),
    ("tite_table", None, "tite-table", False),
    ("single_reports", None, "single-reports", False),
    ("simulation_summaries", "Functions for BOIN_Comb.R", "simulation-summaries", True),
)
with tempfile.TemporaryDirectory(prefix="boin-reference-") as temp:
    for driver, original, fixture, needs_repo in references:
        output = Path(temp) / (fixture + ".csv")
        source = args.original_root.resolve() / "R"
        if original:
            source /= original
        command = [
            args.r_executable,
            "--vanilla",
            "--slave",
            "-f",
            str(repo / "tools" / ("reference_boin_desktop_" + driver + ".R")),
            "--args",
            str(source),
            str(output),
        ]
        if needs_repo:
            command.append(str(repo))
        subprocess.run(command, env=environment, check=True, timeout=120, capture_output=True)
        expected = repo / "research/fixtures/boin-desktop" / (fixture + ".csv")
        assert output.read_bytes() == expected.read_bytes(), fixture
        print(fixture, hashlib.sha256(output.read_bytes()).hexdigest(), "identical", flush=True)
