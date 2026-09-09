# STATTAB console, help and reports

The STATTAB application can now run through a terminal or caller-owned streams.
It uses the [checked session layer](stattab-sessions.md), all twelve distributions
and the existing eight-action CDFLIB list editor. The full catalog entry remains
partial until its final source/version and documentation reconciliation is complete.

Run the installed module with:

```text
python -m mdanderson_stats.stattab
```

It first asks whether to create a report file. Select a distribution by its menu
number; enter positional parameter requests; enter `HELP` for the formula and
parameter meanings; enter a blank line to return to the menu; select `0` to exit.
EOF ends cleanly at any input stage. Ctrl-C exits with status 130. Exhausted input
limits and I/O errors fail explicitly, without a successful-looking numerical row.
The prompts and text layout are modern Python output, not byte-for-byte Fortran.

## Example session

For a Poisson calculation, decline the report initially and enter:

```text
n
10
2 3 ? .
? = = .

0
```

The first request computes the tails and individual term at two events with mean
three. The lower tail is approximately 0.423190081127 and the term approximately
0.224041807655. The second request reuses the mean and **that original lower tail**,
returning a continuous count near two plus floor/floor+1 rows. This repairs the
native program's reuse of an overwritten CDF at count one. A continuous inverse
can lie a few floating-point units from an integer; neighboring rows follow its
actual floor, without claiming exact integer root recovery.

For a normal table, choose menu 9 and enter `T 0 1 ? .`. The list editor offers:

1. Add values.
2. Append a linear sequence.
3. Append a logarithmic sequence.
4. Print the list, with pagination.
5. Delete one position.
6. Delete a range.
7. Sort and remove duplicates.
8. Finish editing and calculate.

Positions are one-based. A sequence with N intervals contributes N+1 points.
The default list capacity is 100, matching STATTAB. Successful edits survive a
rejected action. EOF while editing returns the partial list in `STATTABRun.pending_table`
without replacing the last successful numerical result. Empty lists produce an
explicit empty table. A subsequent scalar request does not reopen the editor.

## Python streams and outcomes

```python
from io import StringIO
from mdanderson_stats import run_stattab

source = StringIO("9\n1 0 1 ? .\n\n0\n")
output, report = StringIO(), StringIO()
outcome = run_stattab(source, output, report_stream=report)
assert outcome.reason == "exit"
assert outcome.completed == 1
print(report.getvalue())
```

`run_stattab` leaves supplied streams open. The report receives help/menu messages,
raw parameter requests, accepted numerical tables and rejection messages. Console
prompts and internal editor prompts are not all duplicated into the report. If
output and report are the same stream, results are written once. Each run keeps
only its last successful result and any list interrupted by EOF; counters record
successful solves and rejected parameter requests, excluding help/menu commands
and internal list-editor retries. On EOF the selected distribution is also returned.

Invalid requests are reported and may be corrected without losing saved session
values. I/O failures, exhausted list/menu/input limits and formatting limits
propagate. Defaults are 1,000 menu/request steps, 1,000 list actions per editor,
4,096 characters per line and pages of 21 list entries. `page_size=0` suppresses
list pagination. `max_output` bounds each numerical table, not the whole transcript.

## File selection and ownership

`run_stattab(..., ask_report=True)` enables the report-file dialogue. It cannot be
combined with `report_stream`. A report opened this way is owned and closed by the
run on normal exit, EOF and exceptions. Quit/back during startup file selection
ends the run without opening a report. Declining a report continues normally.

The public `stattab_open_file(console, ..., read=True, confirm=False, append_ok=True)`
and `stattab_report_file_dialogue(console)` expose the archived helper responsibilities.
They return `STATTABFile`, with status `opened`, `quit`, `back`, or (for the report
question) `declined`. An opened result contains its path and a stream that **the
caller must close**. They do not mutate global report state.

Read requests require an existing file and open read-only. New write files use
exclusive creation. Existing write targets require an explicit quit/retry/overwrite/
append choice; `append_ok=False` removes append. Optional confirmation uses quit/
retry/proceed **before** opening or truncation, repairing the source's destructive
post-open confirmation. File specification has a configurable three-attempt default.
Filesystem failures are shown with context and retain their cause if attempts run out.
EOF propagates from the standalone file helpers and is a clean outcome in the run.

Paths may contain Unicode and spaces. `#` starts a filename comment, as in the
source; control characters and empty names are rejected. Exact case-insensitive
`quit`/`back` names are commands; use a path such as `./quit` for a file of that name.
Files currently named by the console's input/output/report streams are rejected,
including hard links. There is no process-wide registry of unrelated open files
or a fixed Fortran unit-number pool. Python/OS stream errors replace unit exhaustion.
The source's Fortran `delim` setting is replaced by explicit UTF-8 plain-text
formatting; list-directed Fortran quoting is not a Python stream option.

## Numerical text format

`format_stattab_result` prints source-ordered column names, one-based flattened row
indices and `result`, `floor`, or `floor_plus_one` labels. Unavailable neighbors are
marked explicitly without fabricated tails. Scientific notation retains small
probabilities. Default precision is 12 significant digits; choose 17 for float64
round-trip text. The underlying arrays retain their full precision regardless.
Default limits are 10,000 displayed rows and 1,000,000 output characters; unavailable
neighbor rows count toward the row limit. Formatting completes before a numerical
table is sent to the streams. Individual stream writes can still fail; reports are
not transactional filesystem snapshots.

Tests cover every family, all eight list actions and pagination, reuse/error
recovery, empty tables, EOF at several stages, report aliases, resource limits,
actual temporary-file reading/creation/append/overwrite, confirmation cancellation,
active-file protection, stream closure and the installed module entry point.
The numerical batching measurements remain those of the [session layer](stattab-sessions-benchmark.json);
text formatting necessarily visits each displayed row and makes no Fortran speed claim.
