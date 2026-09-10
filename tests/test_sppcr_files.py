"""Real filesystem integration, collision protection and failed publication semantics."""

import os
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats import (
    SPPCRReports,
    format_sppcr_analysis,
    format_sppcr_batch,
    format_sppcr_filemaker,
    read_sppcr_file,
    sppcr_analyze,
    sppcr_data,
    write_sppcr_reports,
)


def data():
    return sppcr_data(
        [0.25, 0.5], [[4, 10, 2], [10, 20, 5]], [50, 100], [100, 102, 104], (100, 102)
    )


@pytest.mark.parametrize(
    "file_format,formatter", [("batch", format_sppcr_batch), ("filemaker", format_sppcr_filemaker)]
)
def test_file_to_analysis_to_saved_reports(tmp_path, file_format, formatter):
    original = data()
    source = tmp_path / "input data.txt"
    text = formatter(original)
    source.write_text(text, encoding="utf-8")
    source.chmod(0o444)
    d = read_sppcr_file(source, file_format=file_format, max_characters=len(text))
    assert_array_equal(d.seen, original.seen)
    assert_array_equal(d.dna, original.dna)
    analysis = sppcr_analyze(d, rng=np.random.default_rng(42), replicates=5)
    reports = format_sppcr_analysis(analysis, write_simulations=True)
    a, s = write_sppcr_reports(reports, tmp_path / "results.ans", tmp_path / "results.sim")
    assert a.is_absolute() and s.is_absolute()
    assert a.read_text() == reports.report
    assert s.read_text() == reports.simulations
    assert source.read_text() == text
    assert not list(tmp_path.glob(".sppcr-*"))


def test_read_limits_and_invalid_encoding(tmp_path):
    source = tmp_path / "input"
    text = format_sppcr_batch(data())
    source.write_text(text)
    with pytest.raises(ValueError, match="max_characters"):
        read_sppcr_file(source, max_characters=len(text) - 1)
    with pytest.raises(ValueError):
        read_sppcr_file(source, max_runs=1)
    source.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        read_sppcr_file(source)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"file_format": "auto"},
        {"unseen_alleles": "bad"},
        {"max_characters": 0},
        {"max_runs": 0},
        {"max_alleles": 0},
    ],
)
def test_invalid_options_before_open(tmp_path, kwargs):
    with pytest.raises(ValueError):
        read_sppcr_file(tmp_path / "missing", **kwargs)


def test_missing_input_propagates(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_sppcr_file(tmp_path / "missing")


def test_report_only_utf8_and_explicit_replacement(tmp_path):
    path = tmp_path / "report"
    reports = SPPCRReports("Unicode μ\n", None)
    assert write_sppcr_reports(reports, path) == (path, None)
    assert path.read_bytes() == reports.report.encode("utf-8")
    with pytest.raises(FileExistsError):
        write_sppcr_reports(SPPCRReports("replacement", None), path)
    assert path.read_text() == reports.report
    write_sppcr_reports(SPPCRReports("replacement", None), path, overwrite=True)
    assert path.read_text() == "replacement"


@pytest.mark.parametrize("same", ["path", "relative", "hardlink", "symlink"])
def test_output_aliases_rejected_before_modification(tmp_path, monkeypatch, same):
    monkeypatch.chdir(tmp_path)
    a = tmp_path / "a"
    a.write_text("original")
    if same == "path":
        b = a
    elif same == "relative":
        b = Path("a")
    elif same == "hardlink":
        b = tmp_path / "b"
        os.link(a, b)
    else:
        b = tmp_path / "b"
        b.symlink_to(a)
    with pytest.raises(ValueError, match="differ"):
        write_sppcr_reports(SPPCRReports("new a", "new b"), a, b, overwrite=True)
    assert a.read_text() == "original"
    assert not list(tmp_path.glob(".sppcr-*"))


def test_conflicting_second_destination_preserves_first(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    b.write_text("original")
    with pytest.raises(FileExistsError):
        write_sppcr_reports(SPPCRReports("new a", "new b"), a, b)
    assert not a.exists() and b.read_text() == "original"


def test_failed_second_staging_cleans_temporary_and_preserves_existing(tmp_path):
    a, b = tmp_path / "a", tmp_path / "missing" / "b"
    a.write_text("original")
    with pytest.raises(FileNotFoundError):
        write_sppcr_reports(SPPCRReports("new a", "new b"), a, b, overwrite=True)
    assert a.read_text() == "original"
    assert not list(tmp_path.glob(".sppcr-*"))


def test_unicode_encoding_failure_before_any_write(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    with pytest.raises(UnicodeEncodeError):
        write_sppcr_reports(SPPCRReports("valid", "\ud800"), a, b)
    assert not list(tmp_path.iterdir())


def test_racing_new_file_is_preserved_and_earlier_publication_is_complete(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    real_link = os.link

    def racing_link(source, target):
        if target == b:
            b.write_text("competing writer")
        return real_link(source, target)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(FileExistsError):
        write_sppcr_reports(SPPCRReports("complete a", "complete b"), a, b)
    assert a.read_text() == "complete a" and b.read_text() == "competing writer"
    assert not list(tmp_path.glob(".sppcr-*"))


def test_replacement_of_symlink_preserves_target(tmp_path):
    target, link = tmp_path / "target", tmp_path / "link"
    target.write_text("original")
    link.symlink_to(target)
    with pytest.raises(FileExistsError):
        write_sppcr_reports(SPPCRReports("new", None), link)
    write_sppcr_reports(SPPCRReports("new", None), link, overwrite=True)
    assert not link.is_symlink() and link.read_text() == "new"
    assert target.read_text() == "original"


@pytest.mark.parametrize(
    "reports,path", [(SPPCRReports("a", "b"), None), (SPPCRReports("a", None), "sim")]
)
def test_simulation_pairing_required(tmp_path, reports, path):
    with pytest.raises(ValueError, match="both"):
        write_sppcr_reports(reports, tmp_path / "a", None if path is None else tmp_path / path)
    assert not list(tmp_path.iterdir())


def test_directory_and_invalid_overwrite_rejected(tmp_path):
    with pytest.raises(IsADirectoryError):
        write_sppcr_reports(SPPCRReports("a", None), tmp_path, overwrite=True)
    with pytest.raises(ValueError, match="boolean"):
        write_sppcr_reports(SPPCRReports("a", None), tmp_path / "a", overwrite="yes")


def test_failed_sync_removes_partial_stage_and_preserves_destination(tmp_path, monkeypatch):
    destination = tmp_path / "report"
    destination.write_text("original")

    def fail_sync(descriptor):
        raise OSError("injected disk sync failure")

    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(OSError, match="disk sync"):
        write_sppcr_reports(SPPCRReports("replacement", None), destination, overwrite=True)
    assert destination.read_text() == "original"
    assert not list(tmp_path.glob(".sppcr-*"))
