"""Native output routing with explicit, repeatable Python substitutions."""

from io import StringIO

import pytest

from mdanderson_stats import CDFConsole, CDFConsoleError


@pytest.mark.parametrize("flag", ["default", "unit", "unit_false", "unit_true", "off", "force"])
def test_native_message_controls_with_unit_only_false_repaired(flag):
    out, report = StringIO(), StringIO()
    c = CDFConsole(StringIO(), out)
    # Native (1X,'Hello') is expressed as a plain Python message template.
    c.message_format = " Hello"
    kwargs = {}
    if flag.startswith("unit"):
        kwargs["unit"] = report
    if flag == "unit_false":
        kwargs["unit_only"] = False
    if flag == "unit_true":
        kwargs["unit_only"] = True
    if flag in ("off", "force"):
        c.print_off = True
    if flag == "force":
        kwargs["force"] = True
    result = c.print_message_format(**kwargs)
    assert out.getvalue() == ("" if flag in ("unit_true", "off") else " Hello\n")
    assert report.getvalue() == (" Hello\n" if flag.startswith("unit") else "")
    assert c.format_printed == (flag != "off")
    assert result == (None if flag == "off" else " Hello")


def test_substitutions_are_initialized_literal_and_repeatable():
    out = StringIO()
    c = CDFConsole(StringIO(), out)
    c.message_format = "Name: {0}; note: {1}; literal: {{}}"
    c.substitutions = ("O'Brien", "{unparsed}")
    assert c.num_subs == 2
    expected = "Name: O'Brien; note: {unparsed}; literal: {}"
    assert c.print_message_format() == c.print_message_format() == expected
    assert out.getvalue() == (expected + "\n") * 2
    assert c.message_format == "Name: {0}; note: {1}; literal: {{}}"
    with pytest.raises(AttributeError):
        c.num_subs = 3


@pytest.mark.parametrize(
    "answer,shown", [("y\n", True), ("n\n", False), ("x\nx\nx\n", True), ("", True)]
)
def test_optional_help_and_failed_response_default(answer, shown):
    out = StringIO()
    c = CDFConsole(StringIO(answer), out)
    c.message_format = "Helpful details"
    c.always_print = False
    c.print_level = 2
    assert c.print_message_format() == ("Helpful details" if shown else None)
    assert c.format_printed == shown
    assert ("Helpful details" in out.getvalue()) == shown


def test_suppression_precedence_force_and_completed_state():
    source, out = StringIO(), StringIO()
    c = CDFConsole(source, out)
    c.message_format = "details"
    c.always_print = False
    c.print_level = 3
    assert c.print_message_format() is None
    c.always_print = True
    assert c.print_message_format() == "details"
    c.print_off = True
    assert c.print_message_format() is None and not c.format_printed
    c.always_print = False
    c.print_level = 2
    assert c.print_message_format(force=True) == "details"
    assert source.tell() == 0 and "Want" not in out.getvalue()


def test_explicit_unit_is_separate_from_default_report_and_not_closed():
    out, report, unit = StringIO(), StringIO(), StringIO()
    c = CDFConsole(StringIO(), out, report_stream=report)
    c.message_format = "record"
    c.print_message_format()
    assert report.getvalue() == ""
    c.print_message_format(unit=unit, unit_only=True)
    assert unit.getvalue() == "record\n" and out.getvalue() == "record\n"
    c.print_message_format(unit=out)
    assert out.getvalue() == "record\nrecord\n"
    assert not unit.closed and not report.closed


@pytest.mark.parametrize("template", ["{", "{1}", "{missing}", "{0:.2f}", "{0.not_an_attribute}"])
def test_invalid_templates_do_not_write_partial_messages(template):
    out, unit = StringIO(), StringIO()
    c = CDFConsole(StringIO(), out)
    c.message_format = template
    c.substitutions = ("text",)
    with pytest.raises(ValueError, match="Invalid message template"):
        c.print_message_format(unit=unit)
    assert out.getvalue() == unit.getvalue() == "" and not c.format_printed


def test_invalid_controls_and_substitutions():
    c = CDFConsole(StringIO(), StringIO())
    with pytest.raises(ValueError):
        c.substitutions = ["x"]
    with pytest.raises(ValueError):
        c.substitutions = (1,)
    with pytest.raises(ValueError):
        c.print_message_format(unit_only=True)
    c.print_level = 0
    with pytest.raises(ValueError):
        c.print_message_format()
    c.print_level = 1
    c.print_off = "yes"
    with pytest.raises(ValueError):
        c.print_message_format()


def test_failed_attempt_clears_completion_and_line_errors_are_not_hidden():
    out = StringIO()
    c = CDFConsole(StringIO("too long\n"), out, max_line_length=1)
    c.message_format = "details"
    assert c.print_message_format() == "details" and c.format_printed
    c.message_format = 123
    with pytest.raises(ValueError):
        c.print_message_format()
    assert not c.format_printed
    c.message_format = "new details"
    c.always_print = False
    c.print_level = 2
    with pytest.raises(CDFConsoleError, match="max_line_length"):
        c.print_message_format()
    assert "new details" not in out.getvalue() and not c.format_printed
