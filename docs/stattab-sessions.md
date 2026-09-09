# STATTAB requests and sessions

`parse_stattab_request` and `STATTABSession` implement the application's positional
request grammar and saved-parameter state. They connect requests to the validated
[result layer](stattab-results.md). Console menus, help text, interactive list
editing, file dialogs and formatted reports remain pending.

```python
from mdanderson_stats import CDFNumberList, STATTABSession

session = STATTABSession("poisson")
forward = session.execute("2 3 ? .")
inverse = session.execute("? = = .")  # Reuses mean and the original computed CDF.

normal = STATTABSession("normal")
table = normal.execute("T 0 1 ? .", table=CDFNumberList([0, 1, 2]))
next_row = normal.execute("3 = = ? .")  # Scalar request; no stale table selection.
assert normal.previous["x"] == 3
```

## Request grammar

Requests use the input order in `STATTAB_DISTRIBUTIONS`, including gamma's explicit
`x, rate, shape, cum, ccum` order. A solve contains exactly one `?`, at most one
`T`, and exactly as many fields as the distribution descriptor requires.

- A finite number supplies an input. Decimal and E/D exponents, unary signs,
  spaces, tabs and comma separators are accepted. Numbers use float64, including
  integer tokens larger than int32; arbitrary adjacent integers above 2**53 are
  not promised. Nonzero literals that overflow or underflow float64 are rejected.
- `?` selects the parameter to compute. Either member selects its complementary
  group. F and noncentral F degree-of-freedom inversions remain unsupported.
- `.` omits one complementary member. Every complementary pair must have exactly
  one dot, including a computed pair. Required scalar parameters cannot be omitted.
  This positional grammar is stricter than `stattab_solve`, which also accepts
  explicit consistent pairs.
- `T` selects the input position receiving the separately supplied table.
- `=` reuses that named parameter from the session's last completed row.

A `#` starts a comment. `HELP`, ignoring case and surrounding whitespace, returns
a help request; it is matched as a whole command, so `SHELPER` is invalid. Blank
or comment-only input returns a menu request. Quoted values/placeholders, other
delimiters and embedded newlines are rejected. Commas/tabs and longer requests
are deliberate usability extensions; the source reads a 70-character record.

`parse_stattab_request(distribution, line)` is pure: it returns an immutable
`STATTABRequest` containing the action, computed parameter, literal inputs, reuse
names and optional table parameter. Parsing checks grammar, not numerical domains
or the existence of saved values. Execution performs those checks through the
result API. Invalid requests raise `ValueError`; numerical failures propagate.

## Session state and tables

A session belongs to one selected distribution. `execute` returns a
`STATTABResult` for a solve, or `STATTABRequest` for a help/menu command. It performs
no stream or file I/O. The caller can render command responses later.

`previous` is an immutable mapping of completed scalar values in input order.
A successful nonempty table saves its final row. A successful empty table returns
an empty result and leaves `previous` unchanged. `last_result` records the most
recent successful solve, including an empty table. Failed parsing, input checks,
root searches or numerical evaluations leave both properties unchanged.

Reuse before a value has been defined is an error. When `=` selects a member of
a complementary pair, execution retains **both saved members** internally. Thus a
stored lower tail rounded to one does not erase its small stored upper tail.
The same rule applies to beta coordinates and binomial success chances. Extra
probability columns and neighboring inverse rows never replace saved parameters.

`execute("HELP")` preserves state. A blank/menu request clears saved values and
`last_result`; `select(distribution)` does the same, even when reselecting the same
distribution. Invalid selection leaves the existing session intact. Independent
sessions share no mutable state. A session is intended for sequential use.

Supply `table` exactly when the request contains `T`. It must be a finite,
one-dimensional array or `CDFNumberList`; list objects are snapshotted without
editing them. Only the selected input varies; other fields are scalar literals or
saved scalar values. Table selection is local to each request. Defaults are
`max_table_size=100`, matching the source, and `max_length=4096`; both limits are
configurable positive integers. Length checks include comments and reject excess
input rather than truncating it. Empty tables are valid.

A noncentral-t DF request may pass `df_bracket=(low, high)` with scalar endpoints
to select a root. Use `stattab_solve` directly for broadcasting bracket arrays;
the session API does not let a bracket silently add table rows. Help/menu commands
do not accept table or bracket arguments.

## Validation

Tests cover native forward and table requests for all twelve families, positional
requests for all 42 computed groups, every table-input position, complementary
queries and omissions, numerical formats, comments, explicit limits, invalid
requests and state preservation on failure. Regression checks exercise undefined
reuse, the native Poisson CDF overwrite, gamma rate/shape swapping, stale table
selection and small-complement preservation. List snapshots, independent sessions,
empty-table behavior and immutable results are also checked.

[Benchmarks](stattab-sessions-benchmark.json) compare one table request against
repeated scalar requests through this same session API, checking equal output
rows and final saved state. Results are three-run medians and measure Python
batching, not speed relative to native Fortran. Numeric calculations remain
vectorized; Python parsing loops visit only the fixed number of input fields.
