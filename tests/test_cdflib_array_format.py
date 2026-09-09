"""Native layouts, repaired errors, and independent decimal rounding checks."""

import io
import json
import math
import re
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import CDFConsole, format_cdflib_array

REFERENCE = json.loads((Path(__file__).parent / "fixtures/cdflib_array_format.json").read_text())
ARRAY_CASES = [c for c in REFERENCE["cases"] if c["mode"] == "array"]


@pytest.mark.parametrize("case", ARRAY_CASES)
def test_native_array_layouts_and_documented_repairs(case):
    values = [float(v) for v in case["input"].split()]
    if case["format"] == "5F11.0" and values[0] == 0:
        assert case["returncode"] == 2 and "Scale factor" in case["stderr"]
        with pytest.raises(ValueError, match="Scientific"):
            format_cdflib_array(values, case["format"])
        return
    actual = format_cdflib_array(values, case["format"])
    if case["format"] == "5E11.3":
        assert "INTERNAL ERROR" in case["stdout"]
        assert actual == "  0.000E+00  0.100E+01  0.100E-03  0.100E+05 -0.100E+01"
    elif case["format"] == "5f11.6":
        assert "INTERNAL ERROR" in case["stdout"]
        assert actual == format_cdflib_array(values, "5F11.6")
    elif case["format"] == "5F18.6":
        assert case["returncode"] == 2
        assert len(actual) == 90
        assert actual[:18].strip() == "0.000E+00" and actual[-18:].strip() == "-1.000000"
    else:
        assert case["returncode"] == 0 and not case["stderr"]
        # Drop only the native list-directed leading blank and unused buffer padding.
        assert actual.rstrip() == case["stdout"][1:].rstrip()


def test_native_message_routing_is_audited_but_separate_from_array_port():
    cases = {c["flag"]: c for c in REFERENCE["cases"] if c["mode"] == "message"}
    assert cases["default"]["stdout"] == " Hello\n" and cases["default"]["report"] == ""
    assert cases["unit"]["stdout"] == cases["unit"]["report"] == " Hello\n"
    assert cases["unit_false"]["stdout"] == cases["unit_true"]["stdout"] == ""
    assert cases["unit_false"]["report"] == cases["unit_true"]["report"] == " Hello\n"
    assert cases["off"]["stdout"] == "" and cases["force"]["stdout"] == " Hello\n"


@pytest.mark.parametrize("precision", [1, 2, 3, 6, 15, 17])
@pytest.mark.parametrize(
    "value", [0, 5e-324, -5e-324, 1e-300, -1e-100, 0.125, 9.9999999, 1000, 1e100, -1e308, 1.7e308]
)
def test_scientific_fields_meet_independent_decimal_rounding_bound(value, precision):
    result = format_cdflib_array([value], f"E{precision + 10}.{precision}").strip()
    match = re.fullmatch(r"([+-]?0\.[0-9]+)E?([+-][0-9]+)", result)
    assert match
    with localcontext() as ctx:
        ctx.prec = 800
        exponent = int(match[2])
        printed = Decimal(match[1]) * (Decimal(10) ** exponent)
        error = abs(Decimal.from_float(float(value)) - printed)
        half_unit = Decimal(10) ** (exponent - precision) / 2
        assert error <= half_unit


def test_switch_threshold_width_overflow_and_signed_zero():
    threshold = float(np.float32(1e-3))
    assert "E" in format_cdflib_array([np.nextafter(threshold, 0)], "F11.6")
    assert "E" not in format_cdflib_array([threshold], "F11.6")
    assert "E" in format_cdflib_array([1e-3], "F11.6")
    assert format_cdflib_array([1000], "F10.6") == "*" * 10
    assert "E" in format_cdflib_array([np.nextafter(1000, np.inf)], "F10.6")
    assert format_cdflib_array([-0.0], "F11.6").strip() == "-0.000E+00"
    assert math.copysign(1, float("-0.000E+00")) < 0


def test_layout_extensions_preserve_explicit_spaces_and_field_counts():
    result = format_cdflib_array([1, 2, 3], "(2f8.3, 3x, e10.3, X)")
    assert len(result) == 30 and result[16:19] == "   " and result.endswith(" ")
    assert result[:16] == "   1.000   2.000"
    assert result[19:29] == " 0.300E+01"
    assert format_cdflib_array([], "3X") == "   "
    assert format_cdflib_array([], "") == ""
    assert format_cdflib_array([1], "F2.1000000000") == "**"


def test_console_records_and_report_ownership():
    output, report = io.StringIO(), io.StringIO()
    console = CDFConsole(io.StringIO(), output, report_stream=report)
    text = console.write_array([1, 2], "2F11.6 2X")
    assert output.getvalue() == report.getvalue() == text + "\n"
    assert text.endswith("  ")
    output = io.StringIO()
    console = CDFConsole(io.StringIO(), output, report_stream=output)
    text = console.write_array([1], "F11.6")
    assert output.getvalue() == text + "\n"
    assert not output.closed and not report.closed


@pytest.mark.parametrize(
    "values,spec",
    [
        ([1], "2F11.6"),
        ([1, 2], "F11.6"),
        ([1], "0F11.6"),
        ([1], "F0.6"),
        ([1], "E11.0"),
        ([1], "F11"),
        ([1], "bad"),
        ([1], "(F11.6"),
        ([[1]], "F11.6"),
        ([np.nan], "F11.6"),
        ([np.inf], "F11.6"),
    ],
)
def test_invalid_formats_do_not_write_partial_records(values, spec):
    output, report = io.StringIO(), io.StringIO()
    with pytest.raises(ValueError):
        CDFConsole(io.StringIO(), output, report_stream=report).write_array(values, spec)
    assert output.getvalue() == report.getvalue() == ""


def test_output_budget_checked_before_formatting_or_allocation():
    with pytest.raises(ValueError, match="max_output"):
        format_cdflib_array([1], "F1000000000.6")
    with pytest.raises(ValueError, match="max_output"):
        format_cdflib_array([1], "F11.6", max_output=10)
    assert len(format_cdflib_array([1], "F11.6", max_output=11)) == 11
    with pytest.raises(ValueError):
        format_cdflib_array([1], "F11.6", max_output=True)
