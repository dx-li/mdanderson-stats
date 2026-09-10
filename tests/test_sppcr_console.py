"""Four-mode menu integration, repeated sessions, EOF and CLI execution."""

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    CDFConsoleError,
    RandlibGenerator,
    format_sppcr_batch,
    format_sppcr_filemaker,
    run_sppcr,
    sppcr_data,
)

FIXTURES = Path(__file__).parent / "fixtures"
INTERACTIVE = json.loads((FIXTURES / "sppcr_interactive.json").read_text())["cases"]["basic"][
    "input"
]
TRUTH = json.loads((FIXTURES / "sppcr_truth_console.json").read_text())["cases"]["basic"]["input"]


@pytest.mark.parametrize("legacy", [False, True])
def test_all_four_modes_and_owned_streams(tmp_path, legacy):
    d = sppcr_data([0.25, 0.5], [[4, 10, 2], [10, 20, 5]], [50, 100], [100, 102, 104], (100, 102))
    batch, filemaker = tmp_path / "batch data", tmp_path / "export"
    batch.write_text(format_sppcr_batch(d))
    filemaker.write_text(format_sppcr_filemaker(d))
    source = io.StringIO(f"1\n{filemaker}\n2\n{batch}\n3\n{INTERACTIVE}4\n{TRUTH}0\n")
    output, report, simulations = io.StringIO(), io.StringIO(), io.StringIO()
    rng = RandlibGenerator() if legacy else np.random.default_rng(42)
    result = run_sppcr(
        source,
        output,
        rng=rng,
        report_stream=report,
        simulation_stream=simulations,
        write_simulations=True,
        replicates=4,
    )
    assert result.reason == "exit" and result.completed == 4 and result.rejected == 0
    assert result.last_analysis.request is not None
    assert report.getvalue().count("SPPCR experiment ") == 4
    assert simulations.getvalue().count("SPPCR experiment ") == 4
    assert output.getvalue().count("SPPCR analysis") == 4
    for stream in (source, output, report, simulations):
        assert not stream.closed


def test_truth_output_choice_does_not_leak_into_observed_analysis():
    source = io.StringIO("4\n" + TRUTH.replace("y\nn\n", "y\ny\n") + "3\n" + INTERACTIVE + "0\n")
    simulations = io.StringIO()
    result = run_sppcr(
        source,
        io.StringIO(),
        rng=np.random.default_rng(1),
        simulation_stream=simulations,
        replicates=3,
    )
    assert result.completed == 2 and result.last_analysis.request is None
    assert simulations.getvalue().count("SPPCR experiment ") == 1


@pytest.mark.parametrize("suffix", ["", "3\n1\n", "4\n2 3\n"])
def test_eof_retains_previous_result(suffix):
    result = run_sppcr(
        io.StringIO("3\n" + INTERACTIVE + suffix),
        io.StringIO(),
        rng=np.random.default_rng(1),
        replicates=3,
    )
    assert result.reason == "eof" and result.completed == 1 and result.last_analysis is not None


def test_bad_file_and_missing_file_can_return_to_menu(tmp_path):
    bad = tmp_path / "bad"
    bad.write_text("invalid data")
    text = f"2\n{tmp_path / 'missing'}\n1\n{bad}\n3\n{INTERACTIVE}0\n"
    result = run_sppcr(io.StringIO(text), io.StringIO(), rng=np.random.default_rng(1), replicates=3)
    assert result.completed == 1 and result.rejected == 2


def test_back_and_quit_file_commands_do_not_sample():
    rng = RandlibGenerator()
    before = rng.get_seeds()
    result = run_sppcr(io.StringIO("2\nback\n1\nquit\n"), io.StringIO(), rng=rng)
    assert result.reason == "exit" and result.completed == result.rejected == 0
    assert rng.get_seeds() == before


@pytest.mark.parametrize(
    "options",
    [
        {"replicates": 0},
        {"precision": 18},
        {"multiplier": 0},
        {"max_steps": 0},
        {"write_simulations": "no"},
    ],
)
def test_bad_configuration_before_input(options):
    source = io.StringIO("0\n")
    with pytest.raises(ValueError):
        run_sppcr(source, io.StringIO(), rng=RandlibGenerator(), **options)
    assert source.tell() == 0


def test_input_output_identity_rejected():
    stream = io.StringIO("0\n")
    with pytest.raises(ValueError, match="differ"):
        run_sppcr(stream, stream, rng=RandlibGenerator())
    assert stream.tell() == 0


def test_output_failure_is_not_retried():
    class Broken(io.StringIO):
        def write(self, text):
            raise OSError("broken report")

    with pytest.raises(OSError, match="broken report"):
        run_sppcr(
            io.StringIO("3\n" + INTERACTIVE + "0\n"),
            io.StringIO(),
            report_stream=Broken(),
            rng=np.random.default_rng(1),
            replicates=3,
        )


def test_output_limits_propagate_after_analysis():
    with pytest.raises(ValueError, match="max_characters"):
        run_sppcr(
            io.StringIO("3\n" + INTERACTIVE + "0\n"),
            io.StringIO(),
            rng=np.random.default_rng(1),
            replicates=3,
            max_report_characters=10,
        )


def test_menu_limit_propagates():
    with pytest.raises(CDFConsoleError, match="max_steps"):
        run_sppcr(io.StringIO("2\nback\n0\n"), io.StringIO(), rng=RandlibGenerator(), max_steps=1)


@pytest.mark.parametrize("legacy", [False, True])
def test_cli_real_process_is_reproducible(legacy):
    command = [
        sys.executable,
        "-m",
        "mdanderson_stats.sppcr",
        "--seed",
        "12345",
        "--replicates",
        "3",
    ] + (["--legacy"] if legacy else [])
    runs = [
        subprocess.run(
            command, input="4\n" + TRUTH + "0\n", text=True, capture_output=True, timeout=30
        )
        for _ in range(2)
    ]
    assert runs[0].returncode == runs[1].returncode == 0
    assert runs[0].stdout == runs[1].stdout and not runs[0].stderr
    assert "SPPCR analysis" in runs[0].stdout


def test_cli_failure_status():
    result = subprocess.run(
        [sys.executable, "-m", "mdanderson_stats.sppcr", "--seed", "1", "--replicates", "0"],
        input="0\n",
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2 and "replicates" in result.stderr


def test_cli_rejected_input_status(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "mdanderson_stats.sppcr", "--seed", "1"],
        input=f"2\n{tmp_path / 'missing'}\n0\n",
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 1 and "Cannot read input file" in result.stdout
    assert not result.stderr
