"""Interactive prompts stay off the report; explicit messages are mirrored."""

from io import StringIO

import pytest

from mdanderson_stats import CDFConsole


@pytest.mark.parametrize(
    "operation,answer", [("get_numbers", "1\n"), ("get_string", "text\n"), ("get_character", "a\n")]
)
def test_prompt_messages_do_not_pollute_report(operation, answer):
    output, report = StringIO(), StringIO()
    console = CDFConsole(StringIO(answer), output, report_stream=report)
    if operation == "get_character":
        console.get_character("abc", message="Choose a value")
    else:
        getattr(console, operation)(message="Choose a value")
    assert "Choose a value" in output.getvalue()
    assert report.getvalue() == ""
    console.write_message("Recorded result")
    assert report.getvalue() == "Recorded result\n"
