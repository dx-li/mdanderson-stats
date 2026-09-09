"""Console contracts use real streams and independently classified native runs."""

import io
import json
import re
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import CDFConsole, CDFConsoleError

REFERENCE = json.loads((Path(__file__).parent / "fixtures/cdflib_console.json").read_text())
INPUT_CASES = [c for c in REFERENCE["cases"] if c["mode"] not in ("array", "list")]


@pytest.mark.parametrize("case", INPUT_CASES)
def test_native_input_transcripts_with_defined_python_errors(case):
    console = CDFConsole(io.StringIO(case["input"]), io.StringIO())
    mode = case["mode"]

    def run():
        if mode == "character":
            return console.get_character("abc")
        if mode == "string":
            return console.get_string()
        dtype = (
            "float32"
            if mode.startswith("real")
            else "int32"
            if mode.startswith("integer")
            else "float64"
        )
        bounds = dict(lo=0, hi=1) if mode in ("number", "double_array") else {}
        return console.get_numbers(3 if mode.endswith("array") else None, dtype=dtype, **bounds)

    if case["input"] in ("", "nan\n"):
        with pytest.raises(EOFError):
            run()
        return
    match = re.search(r"RESULT\s+([TF])\s*\n(.*)", case["stdout"], re.S)
    assert case["returncode"] == 0 and match
    if match[1] == "T":
        with pytest.raises(CDFConsoleError):
            run()
    else:
        result = run()
        if mode == "string":
            assert result == match[2].strip()
        elif mode == "character":
            assert result == int(match[2])
        else:
            expected = np.array([float(x) for x in match[2].split()]).reshape(result.shape)
            np.testing.assert_array_equal(result, expected)


def test_audit_records_native_list_and_input_defects():
    cases = REFERENCE["cases"]
    huge = next(c for c in cases if "1e308 1.5e308" in c["input"])
    # Native relative comparison overflows its denominator and discards 1.5e308.
    assert re.search(r"RESULT\s+F\s+1\s*\n", huge["stdout"])
    assert 1e308 != 1.5e308
    eof = next(c for c in cases if c["mode"] == "string" and c["input"] == "")
    assert eof["returncode"] == 2 and "End of file" in eof["stderr"]
    nan = next(c for c in cases if c["input"] == "nan\n")
    assert re.search(r"RESULT\s+F", nan["stdout"]) and "NaN" in nan["stdout"]


@pytest.mark.parametrize("dtype", ["float64", "float32", "int32"])
def test_vector_retry_is_atomic_bounds_and_owned_dtype(dtype):
    source = io.StringIO("0 2 3\n1 2 4\n1 2 3\n")
    console = CDFConsole(source, io.StringIO())
    result = console.get_numbers(3, dtype=dtype, lo=[1, 2, 3], hi=[1, 2, 3])
    np.testing.assert_array_equal(result, [1, 2, 3])
    assert result.dtype == np.dtype(dtype)
    with pytest.raises(ValueError):
        result.setflags(write=True)


@pytest.mark.parametrize("flag", ["lo_eq_ok", "hi_eq_ok"])
def test_strict_bounds_retry(flag):
    values = "0\n0.5\n" if flag == "lo_eq_ok" else "1\n0.5\n"
    c = CDFConsole(io.StringIO(values), io.StringIO())
    assert c.get_numbers(lo=0, hi=1, **{flag: False}) == 0.5


def test_repeats_continuation_and_discarded_extra_fields():
    c = CDFConsole(io.StringIO("2*1D-2,\n\n3*2.5 ignored\n7\n"), io.StringIO())
    np.testing.assert_array_equal(c.get_numbers(5), [0.01, 0.01, 2.5, 2.5, 2.5])
    assert c.get_numbers() == 7
    c = CDFConsole(io.StringIO("9" * 5000 + "*1\n"), io.StringIO())
    np.testing.assert_array_equal(c.get_numbers(2), [1, 1])


@pytest.mark.parametrize("bad", ["NaN", "Inf", "1e400", "1,,2", "/", "2*", "0*1", "1e", "1++2"])
def test_invalid_or_undefined_numeric_fields_are_retried(bad):
    c = CDFConsole(io.StringIO(bad + "\n1 2\n"), io.StringIO())
    np.testing.assert_array_equal(c.get_numbers(2), [1, 2])


def test_float32_overflow_and_integer_fraction_do_not_silently_convert():
    c = CDFConsole(io.StringIO("1e40\n1e30\n1.5\n2\n"), io.StringIO())
    assert c.get_numbers(dtype="float32") == np.float32(1e30)
    assert c.get_numbers(dtype="int32") == 2


def test_strings_comments_spaces_choices_and_yes_no():
    c = CDFConsole(io.StringIO("#skip\n\n  hello # tail\n B extra\nY\nn\n"), io.StringIO())
    assert c.get_string() == "  hello"
    assert c.get_character("abc") == 2
    assert c.get_yn() is True
    assert c.get_yn() is False
    assert CDFConsole(io.StringIO(" #comment\n"), io.StringIO()).get_string(allow_blank=True) == ""


def test_output_report_pause_and_independent_consoles():
    output, report = io.StringIO(), io.StringIO()
    first = CDFConsole(io.StringIO("\n1\n"), output, report_stream=report)
    second = CDFConsole(io.StringIO("2\n"), io.StringIO())
    first.write_message("hello  ")
    assert output.getvalue() == report.getvalue() == "hello\n"
    first.clear_screen(2)
    first.hold()
    assert first.get_numbers() == 1 and second.get_numbers() == 2
    with pytest.raises(CDFConsoleError, match="failed"):
        first.write_error("failed")
    assert report.getvalue() == "hello\nfailed\n"
    assert not output.closed and not report.closed


def test_limits_and_stream_errors_propagate():
    with pytest.raises(CDFConsoleError, match="max_records"):
        CDFConsole(io.StringIO("#x\n#y\n"), io.StringIO(), max_records=1).get_string()
    with pytest.raises(CDFConsoleError, match="max_records"):
        CDFConsole(io.StringIO("\n\n"), io.StringIO(), max_records=1).get_numbers()
    with pytest.raises(CDFConsoleError, match="max_line_length"):
        CDFConsole(io.StringIO("12345\n"), io.StringIO(), max_line_length=4).get_string()
    stream = io.StringIO()
    stream.close()
    with pytest.raises(ValueError, match="closed"):
        CDFConsole(stream, io.StringIO()).get_numbers()
    with pytest.raises(EOFError):
        CDFConsole(io.StringIO(), io.StringIO()).hold()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(size=-1),
        dict(dtype="int64"),
        dict(lo=2, hi=1),
        dict(size=3, lo=[1, 2]),
        dict(dtype="int32", lo=0.5),
        dict(lo=np.nan),
        dict(lo=1, hi=1, lo_eq_ok=False),
    ],
)
def test_invalid_configuration_does_not_consume_input(kwargs):
    source = io.StringIO("1\n")
    with pytest.raises(ValueError):
        CDFConsole(source, io.StringIO()).get_numbers(**kwargs)
    assert source.tell() == 0


def test_zero_size_and_constructor_validation():
    source = io.StringIO("1\n")
    assert CDFConsole(source, io.StringIO()).get_numbers(0).shape == (0,)
    assert source.tell() == 0
    for kw in ({"max_attempts": 0}, {"max_records": True}, {"max_line_length": -1}):
        with pytest.raises(ValueError):
            CDFConsole(**kw)
