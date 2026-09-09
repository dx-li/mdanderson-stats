"""Bounded STATTAB console orchestration over caller-owned streams."""

from dataclasses import dataclass
from typing import Literal, TextIO

from ._validation import FloatArray
from .cdflib_console import CDFConsole, CDFConsoleError, _positive
from .cdflib_number_list import CDFNumberList
from .stattab_files import stattab_report_file_dialogue
from .stattab_reporting import format_stattab_result, stattab_help
from .stattab_results import STATTAB_DISTRIBUTIONS, STATTABResult
from .stattab_session import STATTABSession, parse_stattab_request


@dataclass(frozen=True)
class STATTABRun:
    """Console outcome, last successful solve and any list interrupted by EOF."""

    reason: Literal["exit", "eof"]
    completed: int
    rejected: int
    last_result: STATTABResult | None
    selected_distribution: str | None
    pending_table: FloatArray | None = None


def run_stattab(
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    *,
    report_stream: TextIO | None = None,
    ask_report: bool = False,
    max_steps: int = 1000,
    max_table_size: int = 100,
    max_line_length: int = 4096,
    max_list_actions: int = 1000,
    page_size: int = 21,
    precision: int = 12,
    max_output: int = 1000000,
) -> STATTABRun:
    """Run the twelve-distribution menu, requests, list editor, help and reports.

    Enter 0 at the menu to exit; a blank request returns to the menu. Invalid
    requests are reported and may be corrected. EOF ends cleanly, preserving an
    interrupted list in the outcome. Stream failures and exhausted limits raise.
    Supplied streams remain caller-owned. ask_report prompts for a report file
    which this function owns and closes. max_steps counts menu/request iterations;
    max_output bounds each formatted numerical result, not the whole transcript.
    """
    if not isinstance(ask_report, bool) or (ask_report and report_stream is not None):
        raise ValueError("ask_report must be boolean and cannot accompany report_stream")
    for key, value in [
        ("max_steps", max_steps),
        ("max_table_size", max_table_size),
        ("max_list_actions", max_list_actions),
        ("precision", precision),
        ("max_output", max_output),
    ]:
        _positive(value, key)
    _positive(page_size, "page_size", allow_zero=True)
    if precision > 17:
        raise ValueError("precision must not exceed 17")
    console = CDFConsole(
        input_stream, output_stream, report_stream=report_stream, max_line_length=max_line_length
    )
    session: STATTABSession | None = None
    last: STATTABResult | None = None
    pending: CDFNumberList | None = None
    completed = rejected = 0
    menu = tuple(STATTAB_DISTRIBUTIONS)
    owned_report: TextIO | None = None
    try:
        if ask_report:
            selected = stattab_report_file_dialogue(console)
            if selected.status in ("quit", "back"):
                return STATTABRun("exit", 0, 0, None, None)
            if selected.status == "opened":
                assert selected.stream is not None
                owned_report = selected.stream
                console.report = owned_report
        console.write_message(stattab_help())
        for _ in range(max_steps):
            if session is None:
                choice = int(
                    console.get_numbers(
                        dtype="int32", lo=0, hi=12, message="Choose distribution (0 exits)"
                    )
                )
                if choice == 0:
                    return STATTABRun("exit", completed, rejected, last, None)
                session = STATTABSession(
                    menu[choice - 1], max_table_size=max_table_size, max_length=max_line_length
                )
                console.write_message(stattab_help(session.distribution))
                continue
            console.prompt()
            # Preserve comment-only/menu semantics and raw request length checks.
            line = console._read()
            if console.report is not None and console.report is not console.output:
                console.report.write("Request: " + line + "\n")
            try:
                request = parse_stattab_request(
                    session.distribution, line, max_length=max_line_length
                )
                if request.action != "solve":
                    session.execute(line)
                    if request.action == "menu":
                        session = None
                        console.write_message(stattab_help())
                    else:
                        console.write_message(stattab_help(session.distribution))
                    continue
                table = None
                if request.table_parameter is not None:
                    pending = CDFNumberList(max_size=max_table_size)
                    table = console.get_list_double(
                        pending,
                        message="List for " + request.table_parameter,
                        max_actions=max_list_actions,
                        page_size=page_size,
                    )
                    pending = None
                solved = session.execute(line, table=table)
            except (ValueError, ArithmeticError) as error:
                rejected += 1
                console.write_message("Invalid request: " + str(error))
                continue
            assert isinstance(solved, STATTABResult)
            # Presentation/stream failures propagate instead of being counted as
            # input errors or attempting to print another partial report.
            text = format_stattab_result(
                solved, precision=precision, max_rows=max_table_size * 3, max_output=max_output
            )
            console.write_message(text)
            completed += 1
            last = solved
    except EOFError:
        console.write_message("End of input.")
        return STATTABRun(
            "eof",
            completed,
            rejected,
            last,
            None if session is None else session.distribution,
            None if pending is None else pending.values,
        )
    finally:
        if owned_report is not None:
            owned_report.close()
    raise CDFConsoleError("STATTAB exhausted max_steps")
