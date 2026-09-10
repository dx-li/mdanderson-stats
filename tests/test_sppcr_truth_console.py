"""Native truth dialogue agreement, corrections, ownership and useful reports."""

import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import (
    CDFConsoleError,
    SPPCRSimulationRequest,
    format_sppcr_truth,
    read_sppcr_truth,
    sppcr_truth,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/sppcr_truth_console.json").read_text())
BASE = FIXTURE["cases"]["basic"]["input"]


def tagged(text, tag):
    return next(line.split()[1:] for line in text.splitlines() if line.startswith(tag + " "))


@pytest.mark.parametrize("case", FIXTURE["cases"].values())
def test_native_parameters_and_report(case):
    request = read_sppcr_truth(io.StringIO(case["input"]), io.StringIO())
    truth, native = request.truth, case["stdout"]
    for name, field in [
        ("DNA", truth.dna),
        ("WELLS", truth.wells),
        ("FREQUENCY", truth.frequency),
        ("CALIBRATION", [truth.calibration]),
    ]:
        assert_allclose(field, list(map(float, tagged(native, name))), rtol=3e-16)
    assert_array_equal(np.array(truth.progenitor) + 1, list(map(int, tagged(native, "PARENTS"))))
    assert [request.bootstrap_from_truth, request.write_simulations] == [
        x == "T" for x in tagged(native, "CHOICES")
    ]
    assert "PARAMETERS ENTERED FOR GENERATING DATA" in native
    report = format_sppcr_truth(request, precision=17)
    assert "progenitor_indices_1_based\t" + "\t".join(tagged(native, "PARENTS")) in report
    rows = report.splitlines()
    start = rows.index("ALLELES") + 2
    frequencies = [float(rows[start + j].split("\t")[1]) for j in range(truth.frequency.size)]
    assert_array_equal(frequencies, truth.frequency)
    start = rows.index("DESIGN") + 2
    design = np.array([[float(x) for x in row.split("\t")[1:]] for row in rows[start:]])
    assert_array_equal(design[:, 0], truth.dna)
    assert_array_equal(design[:, 1], truth.wells)
    assert_array_equal(design[:, 2:], truth.probability)


@pytest.mark.parametrize(
    "bad,original",
    [
        ("0 3", "2 3"),
        ("51 3", "2 3"),
        ("-1 5 3", "2 5 3"),
        ("0 0 0", "2 5 3"),
        ("0 1", ".25 1"),
        ("10001 1", ".25 1"),
        ("0", "1.7"),
        ("0 2", "1 2"),
        ("1 4", "1 2"),
    ],
)
def test_invalid_input_can_be_corrected(bad, original):
    source = BASE.replace(original + "\n", bad + "\n" + original + "\n", 1)
    request = read_sppcr_truth(io.StringIO(source), io.StringIO())
    assert_allclose(request.truth.frequency, [0.2, 0.5, 0.3])
    assert_array_equal(request.truth.dna, [0.25, 1])


def test_false_choice_and_repeated_sessions_preserve_streams():
    source = io.StringIO(BASE.replace("y\nn\n", "n\ny\n") + BASE)
    output = io.StringIO()
    a = read_sppcr_truth(source, output)
    b = read_sppcr_truth(source, output)
    assert not a.bootstrap_from_truth and a.write_simulations
    assert b.bootstrap_from_truth and not b.write_simulations
    assert not source.closed and not output.closed
    assert "bootstrap_model\tobserved_fractions" in format_sppcr_truth(a)


@pytest.mark.parametrize("lines", range(8))
def test_eof_returns_no_partial_request(lines):
    with pytest.raises(EOFError):
        read_sppcr_truth(io.StringIO("\n".join(BASE.splitlines()[:lines]) + "\n"), io.StringIO())


def test_zero_weights_exhaustion():
    with pytest.raises(CDFConsoleError, match="all-zero"):
        read_sppcr_truth(io.StringIO("2 3\n100\n0 0 0\n0 0 0\n"), io.StringIO(), max_attempts=2)


@pytest.mark.parametrize(
    "kwargs", [{"max_attempts": 0}, {"max_records": 0}, {"max_line_length": 0}]
)
def test_invalid_limits_before_input(kwargs):
    source = io.StringIO(BASE)
    with pytest.raises(ValueError):
        read_sppcr_truth(source, io.StringIO(), **kwargs)
    assert source.tell() == 0


def test_report_variable_wells_and_size_limits():
    request = SPPCRSimulationRequest(
        sppcr_truth([0.5, 1], [20, 40], [1, 3], 2, progenitor=(0, 1)), True, False
    )
    report = format_sppcr_truth(request)
    assert "1\t0.5\t20\t" in report and "2\t1\t40\t" in report
    assert format_sppcr_truth(request, max_characters=len(report)) == report
    with pytest.raises(ValueError, match="max_characters"):
        format_sppcr_truth(request, max_characters=len(report) - 1)
    with pytest.raises(ValueError):
        format_sppcr_truth(request, precision=18)


@pytest.mark.parametrize("field", ["mu", "probability", "frequency"])
def test_report_rejects_inconsistent_public_record(field):
    request = read_sppcr_truth(io.StringIO(BASE), io.StringIO())
    bad = replace(request.truth, **{field: getattr(request.truth, field) * 2})
    with pytest.raises(ValueError, match="inconsistent"):
        format_sppcr_truth(replace(request, truth=bad))


def test_report_requires_boolean_choices():
    request = read_sppcr_truth(io.StringIO(BASE), io.StringIO())
    with pytest.raises(ValueError, match="boolean"):
        format_sppcr_truth(replace(request, bootstrap_from_truth="n"))


def test_stream_failure_propagates():
    class Broken(io.StringIO):
        def write(self, text):
            raise OSError("broken output")

    with pytest.raises(OSError, match="broken output"):
        read_sppcr_truth(io.StringIO(BASE), Broken())
