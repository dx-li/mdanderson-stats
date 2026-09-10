"""Output decisions, atomic staging, cancellation and integrated file saving."""

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    CDFConsole,
    CDFConsoleError,
    SPPCRReports,
    run_sppcr,
    sppcr_output_dialogue,
)

TRUTH = json.loads((Path(__file__).parent / "fixtures/sppcr_truth_console.json").read_text())[
    "cases"
]["basic"]["input"]


def console(text):
    return CDFConsole(io.StringIO(text), io.StringIO())


def test_default_paths_and_both_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = console("y\n\n\n")
    result = sppcr_output_dialogue(
        c, SPPCRReports("answer\n", "sim\n"), default_name="dir/input.dat"
    )
    assert result.status == "saved"
    assert result.report == tmp_path / "input.ans" and result.report.read_text() == "answer\n"
    assert (
        result.simulations == tmp_path / "input.sim" and result.simulations.read_text() == "sim\n"
    )
    assert not c.input.closed and not c.output.closed


@pytest.mark.parametrize("choice,expected", [("o", "new\n"), ("a", "old\nnew\n")])
def test_existing_output_choices(tmp_path, choice, expected):
    path = tmp_path / "answer"
    path.write_text("old\n")
    result = sppcr_output_dialogue(console(f"y\n{path}\n{choice}\n"), SPPCRReports("new\n", None))
    assert result.status == "saved" and path.read_text() == expected
    assert not list(tmp_path.glob(".sppcr-*"))


@pytest.mark.parametrize("ending", ["quit\n", "back\n"])
def test_second_file_cancel_preserves_first(tmp_path, ending):
    a = tmp_path / "a"
    a.write_text("old")
    result = sppcr_output_dialogue(console(f"y\n{a}\no\n" + ending), SPPCRReports("new", "sim"))
    assert result.status == "declined" and a.read_text() == "old"


def test_second_file_eof_preserves_first(tmp_path):
    a = tmp_path / "a"
    a.write_text("old")
    with pytest.raises(EOFError):
        sppcr_output_dialogue(console(f"y\n{a}\no\n"), SPPCRReports("new", "sim"))
    assert a.read_text() == "old"


def test_retry_and_protected_input(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    source.write_text("input data")
    result = sppcr_output_dialogue(
        console(f"y\n{source}\n{output}\n"), SPPCRReports("answer", None), protected_paths=(source,)
    )
    assert result.report == output and source.read_text() == "input data"


def test_output_alias_retry_and_independent_modes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("old a\n")
    b.write_text("old b\n")
    result = sppcr_output_dialogue(console(f"y\n{a}\na\n{a}\n{b}\no\n"), SPPCRReports("a\n", "b\n"))
    assert result.status == "saved"
    assert a.read_text() == "old a\na\n" and b.read_text() == "b\n"


def test_decline_and_existing_file_retry(tmp_path):
    assert sppcr_output_dialogue(console("n\n"), SPPCRReports("a", None)).status == "declined"
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("old")
    result = sppcr_output_dialogue(console(f"y\n{a}\nr\n{b}\n"), SPPCRReports("new", None))
    assert result.report == b and a.read_text() == "old"


def test_append_bound_before_either_file_changes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("old a")
    b.write_text("too long")
    with pytest.raises(ValueError, match="max_append_characters"):
        sppcr_output_dialogue(
            console(f"y\n{a}\no\n{b}\na\n"), SPPCRReports("a", "b"), max_append_characters=3
        )
    assert a.read_text() == "old a" and b.read_text() == "too long"


def test_exhausted_path_attempts(tmp_path):
    with pytest.raises(CDFConsoleError, match="max_attempts"):
        sppcr_output_dialogue(
            console(f"y\n{tmp_path}\n{tmp_path}\n"), SPPCRReports("a", None), max_attempts=2
        )


def test_menu_saves_after_completed_analysis_and_preserves_result_on_save_eof(tmp_path):
    path = tmp_path / "answer"
    run = run_sppcr(
        io.StringIO("4\n" + TRUTH + f"y\n{path}\n0\n"),
        io.StringIO(),
        rng=np.random.default_rng(1),
        replicates=3,
        ask_save=True,
    )
    assert run.completed == 1 and run.last_files.report == path and path.exists()
    run = run_sppcr(
        io.StringIO("4\n" + TRUTH),
        io.StringIO(),
        rng=np.random.default_rng(1),
        replicates=3,
        ask_save=True,
    )
    assert run.reason == "eof" and run.completed == 1 and run.last_analysis is not None
    assert run.last_files is None


def test_cli_save_option(tmp_path):
    path = tmp_path / "answer"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mdanderson_stats.sppcr",
            "--seed",
            "1",
            "--replicates",
            "3",
            "--save-reports",
        ],
        input="4\n" + TRUTH + f"y\n{path}\n0\n",
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0 and "SPPCR analysis" in path.read_text()


def test_bad_configuration_before_prompt():
    c = console("y\n")
    with pytest.raises(ValueError):
        sppcr_output_dialogue(c, SPPCRReports("a", None), max_append_characters=0)
    assert c.input.tell() == 0


def test_existing_file_quit_preserves_contents(tmp_path):
    path = tmp_path / "report"
    path.write_text("original")
    result = sppcr_output_dialogue(console(f"y\n{path}\nq\n"), SPPCRReports("new", None))
    assert result.status == "declined" and path.read_text() == "original"


def test_active_named_stream_cannot_be_selected(tmp_path):
    active, answer = tmp_path / "active", tmp_path / "answer"
    with active.open("w+", encoding="utf-8") as output:
        c = CDFConsole(io.StringIO(f"y\n{active}\n{answer}\n"), output)
        result = sppcr_output_dialogue(c, SPPCRReports("new", None))
        assert not output.closed
    assert result.report == answer and answer.read_text() == "new"
    assert "conflicts" in active.read_text()
