from io import StringIO

import pytest

from mdanderson_stats import CDFConsole, compile_misclib_messages, print_misclib_message


def test_native_pause_and_clear_before_display_policy():
    # Unchanged native PAUSE body compiled with gfortran -fcheck=all.
    output, report = StringIO(), StringIO()
    console = CDFConsole(StringIO("\n"), output)
    console.pause()
    assert output.getvalue() == '\n Hit the "Enter" or "Return" key to continue\n\n\n'
    output.seek(0)
    output.truncate()
    console.clear_screen_before_print = True
    console.window_size = 3
    page = compile_misclib_messages(">>BEGIN\nHello\n>>END")[0]
    print_misclib_message(page, console=console, unit=report, unit_only=True)
    assert output.getvalue() == "\n\n\n"  # Source clears stdout even for report-only output.
    assert report.getvalue() == "     Hello\n"
    console.print_off = True
    before = output.getvalue()
    assert print_misclib_message(page, console=console) is None
    assert output.getvalue() == before and not console.format_printed
    console.window_size = -1
    with pytest.raises(ValueError):
        print_misclib_message(page, console=console, force=True)
