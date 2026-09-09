# CDFLIB message templates and controls

`CDFConsole.print_message_format` completes the F95 console module using explicit
Python templates and per-console state. It returns the rendered message without
its terminating newline, or `None` when suppressed.

```python
from io import StringIO
from mdanderson_stats import CDFConsole

output, report = StringIO(), StringIO()
console = CDFConsole(StringIO(), output)
console.message_format = "Result for {0}: {1}"
console.substitutions = ("O'Brien", "converged")
assert console.num_subs == 2
assert console.print_message_format(unit=report) == "Result for O'Brien: converged"
assert output.getvalue() == report.getvalue()
```

## Templates and source mapping

`message_format` is a Python `str.format` template, initially empty. For example,
the native `(1X,'Hello')` becomes `" Hello"`. This is a documented replacement for
Fortran FORMAT expressions, not an interpreter for their full syntax. Use `{0}`,
`{1}`, etc. for positional replacements and `{{`/`}}` for literal braces.
`substitutions` must be a tuple of strings. Inserted quotes and braces are literal;
values are not reparsed, and repeated calls do not mutate the template.

The native public `num_subs` depends on private substitution-position/string
arrays with no public initialization API. Python replaces those arrays with
`substitutions`; read-only `num_subs` is always its length. There is no independent
count that can expose uninitialized storage. Long templates are not truncated.
Invalid templates raise `ValueError` before message output.

## Display policy

- `always_print=True` and `print_off=False` are the defaults.
- `print_off=True` suppresses messages unless `force=True`.
- When `always_print=False`, `print_level=1` displays, `2` asks whether to show
  help, and `3` suppresses. Python exposes the source's private help-level policy
  as a validated per-console setting.
- `force=True` overrides suppression and skips the optional question.
- At level 2, a valid `n` suppresses and `y` displays. EOF or exhausted invalid
  choice attempts defaults to displaying help, matching the source's fallback.
  Line-limit and stream failures propagate; direct `get_yn` still raises on EOF
  or exhausted attempts.

`format_printed` resets before each attempt and becomes true only after all
requested writes finish. Malformed controls raise `ValueError`, including an
invalid level, nonboolean controls, or `unit_only=True` without a stream.

## Output routing and repairs

Without `unit`, the message goes to the console output. An explicit `unit`
receives the same record; `unit_only=True` suppresses the console copy. The source
incorrectly suppresses console output whenever `unit_only` is present, even when
false. Python honors its value. Supplying the console output itself as `unit`
writes once. Streams remain caller-owned and are never closed.

This method does not automatically use `report_stream`: the source distinguishes
its explicit message unit from the default report unit used by `write_message`
and `write_array`. Optional-help prompts use console output. Message whitespace
is retained and one newline appended. Underlying write errors propagate; writes
to separate streams are not transactional.

## Validation and coverage

The [native fixture](../tests/fixtures/cdflib_array_format.json) records six
message routing/suppression cases compiled from unchanged, hash-verified modules.
Tests reproduce the defined routing and independently verify the false-flag
repair, substitutions, repeated calls, help answers/fallbacks, forced display,
state reset, invalid templates and stream ownership.

All seven F95 support modules now have validated mappings with documented Python
semantics.
See the [CDFLIB90 completion audit](cdflib90-completion.md) for the
complete library scope. No speedup over native interactive I/O is claimed.
