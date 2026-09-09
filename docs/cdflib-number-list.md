# CDFLIB numeric list editing

`CDFNumberList` holds a bounded numeric list, and `CDFConsole.get_list_double`
implements all eight actions of the native list editor. Explicit state preserves
successful edits after EOF or an error. Each rejected action leaves the previous
list intact; callers can inspect or resume the same state.

```python
from io import StringIO
from mdanderson_stats import CDFConsole, CDFNumberList

state = CDFNumberList([3, 1, 1], max_size=5)
console = CDFConsole(StringIO("7\n8\n"), StringIO())
result = console.get_list_double(state)
assert result.tolist() == [1, 3]
```

| Action | Behavior |
| --- | --- |
| 1 | Read a count, then append that many individually supplied values; zero is a no-op |
| 2 | Append a linearly spaced sequence from start to stop |
| 3 | Append a logarithmically spaced sequence with positive endpoints |
| 4 | Print positions and values, pausing after each page and the final nonempty page |
| 5 | Delete one one-based position |
| 6 | Delete an inclusive range; either order of endpoints is accepted |
| 7 | Sort ascending and remove approximate duplicates against the last retained value |
| 8 | Return an immutable snapshot and end editing |

Spacing takes a positive integer number of intervals and adds one more point than
intervals, including both endpoints. Ascending, descending and equal endpoints
are supported. Fractional interval counts are rejected rather than silently
truncated as in the source. Supplied endpoints are reproduced exactly.

`CDFNumberList(values=(), max_size=1000, lo=None, hi=None, lo_eq_ok=True,
hi_eq_ok=True)` configures capacity and inclusive/exclusive bounds. The default
1000-value capacity is a Python convenience; native capacity comes from the
caller's allocated array. Capacity cannot exceed the console's int32 index range.
Initial and newly supplied values must be finite, one-dimensional and satisfy the
configured bounds. This validates initial state that the source leaves unchecked.

The same operations are available without interactive I/O:

```python
state = CDFNumberList(max_size=6)
state.append([3, 1])
state.append_spaced(2, 4, intervals=2)
state.delete(1)
state.sort_unique()
assert state.values.tolist() == [1, 2, 3, 4]
```

`values` returns an owned, immutable snapshot. Inputs are copied into preallocated
NumPy storage, and snapshots remain unchanged after later edits. Read-only
properties expose `size`, `max_size`, `lo`, `hi` and the inclusion flags. Appends,
spacing, deletion and duplicate removal validate before changing retained state.
The input array supplied by the caller is never modified. Independent states and
consoles can be used independently; concurrent mutation of one state is unsupported.

Linear spacing uses power-of-two normalization and convex weights, avoiding
`stop-start` overflow and premature underflow of weighted subnormal terms. Log
spacing uses relative `log1p` for close endpoints and logarithms for widely
separated endpoints, avoiding overflow or underflow of `stop/start`. Small rounding
excursions are confined to the endpoint range, and endpoints are set exactly.
Storage and spacing use NumPy; sorting reuses the existing `sort_list` implementation.

The source duplicate criterion is
`abs(x-y)/max(abs(x)+abs(y), 1e-100) < 1e-14`. The implementation scales its terms
to avoid overflow and compares each candidate with the **last retained** value.
The relation is not transitive: using only adjacent differences would change the
source's grouping rule. The native routine incorrectly merges `1e308` and
`1.5e308`; Python retains both. This approximate grouping is intentionally distinct
from exact `unique` semantics.

The console uses its ordinary three-attempt numeric-input limit within each
action. By default it tolerates three failed actions and aborts on the fourth,
matching the native cumulative failure count. Exhausted menu input terminates
immediately. `max_failures` can override the action allowance; `max_actions=10000`
adds a bound on the whole editing session. These limits raise `CDFConsoleError`.
EOF and stream errors propagate, while the explicit list state retains earlier
successful edits. For example, after EOF during a later action, `state.values`
still contains previously appended values.

`page_size=21` matches the source's 24-line window minus three hold lines. Printing
consumes one input record per pause. The final page is paused exactly once,
including when its length is an exact page multiple. Empty lists do not pause;
`page_size=0` disables pauses. Menu and numeric output text use Python formatting,
not byte-identical Fortran console layout. Optional `message` text is shown at
each menu iteration.

Tests replay all recorded native menu actions from the [console fixture](../tests/fixtures/cdflib_console.json),
with explicit repairs for overflowed duplicate comparison and exact log endpoints.
Independent Decimal calculations check spacing and duplicate groups. Further
regressions cover extreme/subnormal inputs, atomic rejection, retained edits after
EOF, distinct retry budgets, pagination and fractional interval rejection.

[Array formatting](cdflib-array-format.md) and [message controls](cdflib-message-format.md)
complete the F95 console support module with documented Python semantics.
See the [CDFLIB90 completion audit](cdflib90-completion.md) for the
complete library scope.
