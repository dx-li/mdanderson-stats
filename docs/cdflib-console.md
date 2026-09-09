# CDFLIB console input

`CDFConsole` implements CDFLIB's character, yes/no, string and numeric input,
plus basic output, prompt, pause and clear-screen operations. Each instance owns
its interaction state and references caller-owned streams; it never closes them.
Default streams are the current `sys.stdin` and `sys.stdout` at construction.
There is no shared Fortran logical-unit or saved input state.

```python
from io import StringIO
from mdanderson_stats import CDFConsole

console = CDFConsole(StringIO("bad\n0.25\n"), StringIO())
value = console.get_numbers(lo=0, hi=1)
assert value.shape == () and value == 0.25

console = CDFConsole(StringIO("2*0.5,\n1D-2\n"), StringIO())
values = console.get_numbers(3)
assert values.tolist() == [0.5, 0.5, 0.01]
```

`get_numbers(size=None, dtype="float64", message="", lo=None, hi=None,
lo_eq_ok=True, hi_eq_ok=True)` represents all six native numeric overloads.
The supported types are `float64`, `float32` and `int32`; `size=None` returns a
zero-dimensional array and explicit nonnegative `size` returns a vector. Results
own immutable storage and retain their selected dtype. Explicit zero size returns
an empty vector without consuming input. Floating parsing follows NumPy's dtype
conversion; float32 rounding and subnormals are preserved. Integer parsing does
not round through floating point or silently truncate fractional input.

Bounds can be scalar, length one or a vector matching `size`. They are interpreted
in the selected dtype. Bounds must be finite, and integer bounds must be integral
and representable. Inverted or empty intervals are rejected before reading input.
Each valid vector is checked as a whole. A malformed or out-of-bound entry restarts
the entire requested vector, with three attempts by default. Partially initialized
values are never returned.

Numeric records support whitespace/comma separators, decimal exponents (`e`, `E`,
`d`, `D`, or implicit exponents such as `1-2`), positive repeat counts such as `3*0.5`, and continuation across records.
Blank continuation records are allowed. Unused fields on the last record are
discarded, matching native list-directed reads. Null fields and slash termination
are rejected when a value is still required because they can leave native output
undefined. Repeat expansion is limited to the requested output size. Nonfinite
values and dtype overflow are rejected; underflow follows the dtype's rounding.

`get_character(chars, message="")` returns a one-based index into nonblank,
lowercase ASCII choices. It checks the first non-space input character and folds
ASCII uppercase to lowercase. Extra text is ignored. `get_yn(message="")` selects
`yn` and returns a boolean. Invalid responses have the same default three-attempt
limit. Strings and choice lines are not silently truncated to the native 80-byte
buffer.

`get_string(message="", allow_blank=False)` skips records beginning with `#`,
removes inline comments and trailing spaces, and retains leading spaces. A line
with spaces before `#` counts as a blank response. Skipped comment records do not
consume retry attempts. Three disallowed blank responses exhaust the default
limit. With `allow_blank=True`, a blank response returns the empty string.

EOF consistently raises `EOFError`. Exhausted attempts, record budgets or line
limits raise `CDFConsoleError`, replacing undefined outputs and process-level STOP
behavior. Constructor settings `max_attempts=3`, `max_records=10000` and
`max_line_length=10000` bound retries, numeric/string records per operation and
record length. Underlying stream errors propagate. The record cap includes blank
continuations, comment-only string records and failed numeric attempts.

Messages are plain Python strings, including ordinary Python formatting; they are
not interpreted as Fortran FORMAT expressions. `write_message` trims trailing
spaces and writes to output and optional `report_stream`. Supplying the same stream
for both writes only once. `write_error` writes a message then raises
`CDFConsoleError`. `prompt` writes and flushes `" > "`; `hold` displays a pause
message and consumes a record; `clear_screen(lines=24)` writes blank lines without
terminal escape sequences. Interactive prompts/retry text are not byte-for-byte
Fortran transcript reproductions and are not copied to the report stream.

The source's `report_unit` maps to `report_stream`. These methods implement
`clear_screen`, `get_character`, `get_string`, `get_yn`, `hold`, `prompt`,
`write_error`, `write_message` and the six `get_numbers` overloads.
[Message templates and controls](cdflib-message-format.md) complete the console
module with documented Python formatting semantics. The
[list editor](cdflib-number-list.md) uses these input methods.

The [reference generator](../tools/reference_cdflib_console.py) compiles four
unchanged source modules from the hash-verified archive. The [31 transcripts](../tests/fixtures/cdflib_console.json)
include input, return codes, output and diagnostics; compiler, driver and source
hashes are recorded. Backtrace addresses and temporary build paths are normalized.
All runs finish within the three-second per-call limit. Tests use defined native
results and independently exercise repaired error handling and output ownership.

The fixture also establishes repaired source defects: string EOF
causes a native runtime exit, numeric bounds accept NaN, and the list editor's
relative comparison merges distinct `1e308` and `1.5e308` values when its denominator
overflows. The [Python list editor](cdflib-number-list.md) now implements all eight menu
actions with explicit state and corrected numerical behavior.

Parsing and interactive I/O are scalar control flow; conversion and vector bounds
use NumPy. No speedup over native console I/O is claimed. CDFLIB90 and the full
catalog conversion remain partial.

[Numeric array formatting](cdflib-array-format.md) is available through
`write_array`, with checked field counts and support for records longer than 79 characters.
