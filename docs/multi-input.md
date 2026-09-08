# MULTI p-value text input

```python
from mdanderson_stats import parse_multi_data, read_multi_data

observations = read_multi_data("pvalues.txt")
print(observations.pvalues)  # sorted ascending
print(observations.order)  # zero-based indices into accepted input
print(observations.entered_values)  # reconstruct original accepted order
for diagnostic in observations.warnings:
    print(diagnostic.line, diagnostic.column, diagnostic.token, diagnostic.reason)

observations = parse_multi_data(".1 .2 .3 .4 q", terminal=True)
```

These functions implement the desktop RDDATA input contract using the original
QLEX token state machine. Inputs need 4–10,000 accepted p-values, inclusive of
0 and 1. Out-of-range numbers and nonnumeric tokens are excluded and returned as
structured diagnostics with one-based line/column positions and the lexer token
text. Quoted token text excludes its quotes and undoubles matching quotes.
Warnings are returned explicitly, rather than printed or suppressed. Too few or
too many accepted values raises ValueError; no successful partial result is
returned. Sorted values and index arrays are read-only. Ties retain input order,
which is a defined stable convention rather than a promise about native DSORT's
tie ordering.

File input uses UTF-8 and propagates I/O failures. File names are explicit, not
prompted for and retried. `terminal=True` parses supplied terminal text: any
nonnumeric token beginning q or Q ends input immediately, including quoted strings
and names such as Quit. In file mode those tokens produce diagnostics. End of the
supplied text also ends input; no process-level EOF failure is reproduced. This
is the text ingestion layer, not yet the desktop change-data/menu session.

## Syntax and compatibility

QLEX syntax is deliberately distinct from CSV or a generic Python float parser:

- Blank spaces separate tokens. Commas, parentheses and other delimiters each
  produce a nonnumeric diagnostic while adjacent numeric tokens remain available.
- Tabs and unsupported characters are diagnosed, rather than silently treated as
  spaces. CRLF line endings are normalized; internal control characters are tokens.
- Decimal real tokens allow E/e/D/d exponents. A decimal point is required:
  `1.e-2` is numeric, while `1e-2` is a nonnumeric token under the source rules.
- Quotes, doubled quotes, identifiers, operators, malformed strings and unary
  signs follow QLEX's state/action tables. Quoted numbers are not p-values.
- Fractional digits and decimal exponent shifts follow the native accumulation
  algorithm, allowing small differences from correctly rounded Python float
  conversion. Very large exponent processing is bounded after binary64 saturation.
  Exponent magnitudes at least 2,147,483,647 are rejected as in native QLEX.

Two source defects are corrected without a compatibility mode: QLEX fails to apply
integer minus signs (so native RDDATA accepts `-1` as +1); the port correctly rejects
it. Native RDDATA reads fixed 212-character records and can silently lose the rest
of a line or its final unterminated token; the port processes complete physical
lines and terminates the final token explicitly. Non-ASCII characters are diagnosed
instead of indexing outside the native 128-element character table. These changes
prevent incorrectly accepted negative values and silently lost observations.

## Validation and provenance

`tools/reference_multi_input.py` extracts original QLEX unchanged into an ignored
private build and compiles it with IEEE binary64/single and 32-bit integer machine
maxima. `tests/fixtures/multi_input.json` records the source hash and 18 original
lexer cases. Tests compare token classes/text and numeric values, then independently
apply RDDATA's acceptance rules to those native tokens and compare accepted values
and diagnostics. The signed-integer correction is explicit in those comparisons.
Additional tests exercise sorted/input-order reconstruction, ties, CRLF file reads,
terminal quit behavior, missing files, long lines, limits, overflow/underflow,
negative integers, tabs and malformed arguments. Original MULTI redistribution
terms cover the adapted lexer tables; see THIRD_PARTY_NOTICES.md.
