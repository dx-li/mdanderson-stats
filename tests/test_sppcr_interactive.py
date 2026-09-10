"""Native interactive input agreement and bounded, caller-owned stream behavior."""

import io
import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import read_sppcr_interactive, sppcr_bootstrap, sppcr_bootstrap_intervals
from mdanderson_stats.cdflib_console import CDFConsoleError

FIXTURE = json.loads((Path(__file__).parent / "fixtures/sppcr_interactive.json").read_text())
BASE = FIXTURE["cases"]["basic"]["input"]


def native(case, tag):
    return np.array(
        [
            [float(x) for x in line.split()[1:]]
            for line in case["stdout"].splitlines()
            if line.split() and line.split()[0] == tag
        ]
    )


@pytest.mark.parametrize("name", list(FIXTURE["cases"]))
def test_native_interactive_data_agreement(name):
    case = FIXTURE["cases"][name]
    assert case["returncode"] == 0
    output = io.StringIO()
    d = read_sppcr_interactive(io.StringIO(case["input"]), output)
    np.testing.assert_array_equal(d.dna, native(case, "dna")[0])
    np.testing.assert_array_equal(d.wells, native(case, "wells")[0])
    np.testing.assert_array_equal(d.seen, native(case, "seen"))
    np.testing.assert_array_equal(d.allele_sizes, native(case, "sizes")[0])
    np.testing.assert_array_equal(np.array(d.progenitor) + 1, native(case, "parents")[0])
    assert "genome equivalents" in output.getvalue()


def test_duplicate_labels_and_missing_parents_can_be_corrected():
    text = BASE.replace("100 102 104\n", "100 100 104\n100 102 104\n").replace(
        "100 102\n", "100 999\n100 102\n"
    )
    output = io.StringIO()
    d = read_sppcr_interactive(io.StringIO(text), output)
    assert d.progenitor == (0, 1)
    assert "must be distinct" in output.getvalue()
    assert "must appear" in output.getvalue()


@pytest.mark.parametrize(
    "original,bad",
    [
        (".5 1", ".0001 1"),
        ("20 30", "20 501"),
        ("4 8 2", "4 21 2"),
        ("100 102 104", "100 102 1000"),
    ],
)
def test_out_of_range_vectors_restart_entire_vector(original, bad):
    d = read_sppcr_interactive(
        io.StringIO(BASE.replace(original, bad + "\n" + original)), io.StringIO()
    )
    np.testing.assert_array_equal(d.seen, [[4, 8, 2], [10, 15, 5]])
    np.testing.assert_array_equal(d.dna, [1, 2])


def test_disclosed_omission_and_explicit_unseen_parent_policy():
    text = BASE.replace("4 8 2", "4 8 0").replace("10 15 5", "10 15 0")
    output = io.StringIO()
    d = read_sppcr_interactive(io.StringIO(text), output)
    assert d.omitted_allele_sizes == (104,)
    assert "Omitted never-seen alleles: 104" in output.getvalue()
    text = BASE.replace("4 8 2", "0 8 2").replace("10 15 5", "0 15 5")
    with pytest.raises(ValueError, match="unseen progenitor"):
        read_sppcr_interactive(io.StringIO(text), io.StringIO())
    d = read_sppcr_interactive(io.StringIO(text), io.StringIO(), unseen_alleles="retain")
    assert d.progenitor == (0, 1) and np.all(d.seen[:, 0] == 0)


def test_successive_experiments_and_stream_ownership():
    source, output = io.StringIO(BASE + BASE), io.StringIO()
    a = read_sppcr_interactive(source, output)
    b = read_sppcr_interactive(source, output)
    np.testing.assert_array_equal(a.seen, b.seen)
    assert not source.closed and not output.closed
    assert not a.seen.flags.writeable
    assert a.seen is not b.seen


@pytest.mark.parametrize("text", ["", "2\n", "2\n3\n100 102\n", BASE.rsplit("10 15 5", 1)[0]])
def test_eof_returns_no_partial_experiment(text):
    source = io.StringIO(text)
    with pytest.raises(EOFError):
        read_sppcr_interactive(source, io.StringIO())
    assert not source.closed


@pytest.mark.parametrize(
    "text",
    [
        "0\n0\n0\n",
        "2\n3\n" + ("100 100 104\n" * 3),
        "2\n3\n100 102 104\n" + ("100 999\n" * 3),
    ],
)
def test_numeric_and_identity_retry_limits(text):
    with pytest.raises(CDFConsoleError):
        read_sppcr_interactive(io.StringIO(text), io.StringIO())


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(max_attempts=0),
        dict(max_records=0),
        dict(max_line_length=True),
        dict(unseen_alleles="unknown"),
    ],
)
def test_invalid_configuration_consumes_no_input(kwargs):
    source = io.StringIO(BASE)
    with pytest.raises(ValueError):
        read_sppcr_interactive(source, io.StringIO(), **kwargs)
    assert source.tell() == 0


def test_record_and_line_limits():
    with pytest.raises(CDFConsoleError, match="max_records"):
        read_sppcr_interactive(
            io.StringIO(BASE.replace(".5 1", ".5\n1")), io.StringIO(), max_records=1
        )
    with pytest.raises(CDFConsoleError, match="max_line_length"):
        read_sppcr_interactive(io.StringIO(BASE), io.StringIO(), max_line_length=5)


def test_output_failure_propagates():
    class Broken(io.StringIO):
        def write(self, value):
            raise OSError("output unavailable")

    with pytest.raises(OSError, match="output unavailable"):
        read_sppcr_interactive(io.StringIO(BASE), Broken())


def test_interactive_to_bootstrap_intervals():
    d = read_sppcr_interactive(io.StringIO(BASE), io.StringIO())
    b = sppcr_bootstrap(
        d.dna, d.seen, d.wells, progenitor=d.progenitor, rng=np.random.default_rng(8), replicates=20
    )
    assert np.all(sppcr_bootstrap_intervals(b).frequency.available)
