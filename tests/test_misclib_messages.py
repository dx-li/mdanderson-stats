import json
from io import StringIO
from pathlib import Path

import pytest

from mdanderson_stats import CDFConsole, compile_misclib_messages, print_misclib_message


def test_repaired_native_generator_and_unchanged_native_renderer():
    case = json.loads((Path(__file__).parent / "fixtures/misclib-messages-native.json").read_text())
    pages = compile_misclib_messages(case["source"])
    assert [p.name for p in pages] == ["message", "message", "detail"]
    for page, values, expected in zip(pages, case["substitutions"], case["expected"], strict=True):
        assert page.render(values) == expected
        assert page.render(values) == expected  # rendering does not edit the template
    assert pages[0].substitution_widths == (8,)
    assert pages[1].substitution_widths == (3,)


def test_stream_controls_literal_substitutions_and_parser_errors():
    page = compile_misclib_messages(">>BEGIN Named\nValue: %%%%% and {braces}\n>>END")[0]
    out, report = StringIO(), StringIO()
    console = CDFConsole(StringIO(), out)
    console.message_format, console.substitutions = "Previous {0}", ("state",)
    assert print_misclib_message(page, ["{'x'}extra"], console=console, unit=report) == (
        "     Value: {'x'} and {braces}"
    )
    assert out.getvalue() == report.getvalue()
    assert (console.message_format, console.substitutions) == ("Previous {0}", ("state",))
    console.print_off = True
    assert print_misclib_message(page, ["text"], console=console) is None
    assert not console.format_printed
    before = out.getvalue()
    print_misclib_message(page, ["x"], console=console, unit=report, unit_only=True, force=True)
    assert console.format_printed and out.getvalue() == before
    assert not out.closed and not report.closed
    with pytest.raises(ValueError):
        print_misclib_message(page, [], console=console)
    assert not console.format_printed
    for text in (">>BEGIN\nunclosed", ">>END", ">>BEGIN\n>>BEGIN", ">>BEGIN\n>>END"):
        with pytest.raises(ValueError):
            compile_misclib_messages(text)
    assert compile_misclib_messages("ignored outside a block") == ()
    page = compile_misclib_messages(">>BEGIN\n\n  indented  \n\n>>END")[0]
    assert page.render() == "\n       indented\n"
