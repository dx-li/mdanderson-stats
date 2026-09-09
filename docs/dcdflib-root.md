# Legacy reverse-communication root finders

`dcdflib_support` exposes all four archived entry points: `dstinv`, `dinvr`,
`dstzr` and `dzror`. Configuration returns independent Python state instead of
mutating shared C/F77 continuation variables. The caller supplies scalar residuals
only when requested.

```python
from mdanderson_stats import dcdflib_support as legacy

state = legacy.dstzr(0, 2, 1e-12, 0)
result = legacy.dzror(state)
while result.status == 1:
    result = legacy.dzror(state, result.x * result.x - 2)
assert result.status == 0
assert abs(result.x * result.x - 2) < 3e-12

state = legacy.dstinv(0, 2, 0.5, 0.5, 5, 1e-12, 0)
result = legacy.dinvr(state, initial=1)
while result.status == 1:
    result = legacy.dinvr(state, result.x * result.x - 2)
assert result.status == 0
```

## Configuration and protocol

| Entry point | Python call |
|---|---|
| DSTINV | `dstinv(small,big,absstp,relstp,stpmul,abstol,reltol,*,max_evaluations=4096)` |
| DINVR | `dinvr(state,fx=None,*,initial=None)` |
| DSTZR | `dstzr(xlo,xhi,abstol,reltol,*,max_evaluations=4096)` |
| DZROR | `dzror(state,fx=None)` |

Bounds are finite and ascending, allowing a single-point interval. Tolerances
are finite and nonnegative with at least one positive. Stepping parameters are
finite and nonnegative, the multiplier exceeds one, and the initial step
`max(absstp,relstp*abs(initial))` must be positive for a nonsingleton search.
`max_evaluations` is a positive Python integer, excluding booleans.

The first DINVR call requires a finite initial value inside the configured
bounds. DZROR does not take an initial guess. Neither first call accepts fx.
Subsequent calls require the finite scalar residual at the outstanding x and
must not resupply initial. Supply `f(x)-target` to solve a nonzero target.
Invalid input raises `ValueError` without consuming a pending request. In
particular, an out-of-bounds initial value raises rather than invoking native STOP.

Each state runs one search. Construct another to restart after success, failure
or exhaustion. Mixing configuration and advance modes is rejected. Independent
states can be interleaved or used by nested callbacks; simultaneous access to
the same state is unsupported. `state.result` is `None` before starting and then
holds the latest immutable `LegacyRootResult` snapshot.

## Results, brackets and failure flags

| Status | Meaning |
|---|---|
| 1 | Evaluate the residual at x and resume |
| 0 | Success; x and the bracket are available |
| -1 | Endpoint residuals do not bracket a zero; qleft/qhi are available |
| -2 | Evaluation budget exhausted; the terminal snapshot is retained and `ArithmeticError` is raised |

Every snapshot includes status, x and the number of supplied residuals. On
success, `xlo=x` is the endpoint with the smaller absolute residual and `xhi` is
the other endpoint. **They are not necessarily in numerical order.** This follows
the executable source and DSTZR's best-residual contract, despite the DZROR
header describing them as lower/upper bounds. An exact evaluated zero collapses
the bracket. Bracket fields are `None` on requests and failures, replacing native
undefined values.

The legacy stopping rule is

```text
abs(xlo-xhi) <= max(abstol,reltol*abs(xlo))
```

The bracket also permits termination when its endpoints are adjacent float64
values and further subdivision is impossible. The implementation compares half
widths and computes the halved tolerance without overflowing an intermediate
product. The F95 interface retains its different midpoint/sum-tolerance rule.

For a continuous function with opposite endpoint signs, the bracket contains a
root and bounds the coordinate error. DINVR's stepping interpretation additionally
assumes monotonicity. Equal-sign endpoints do not prove the absence of interior
roots for a nonmonotone function. A discontinuity or a residual rounded to zero
can invalidate a mathematical root guarantee; the solver can only use the
residuals supplied by the caller.

On status -1, qhi records whether the endpoint residuals are positive. qleft
indicates the native inferred direction of unsuccessful search. The reported x
remains the configured upper bound, as in the native endpoint checks, even if
qleft is true. A constant negative residual gives qleft=true for DINVR and false
for DZROR; that distinction is retained. Failure flags are `None` on success and
pending requests instead of retaining stale values.

## Algorithm, validation and performance

The implementation reuses the [F95 search engine](cdflib-root.md) with a distinct
legacy stopping policy and result adapter. It uses safeguarded secant/inverse
quadratic proposals and bisection, not an exact replay of the native Algorithm R
trajectory. Callback counts and intermediate requests can therefore differ;
an exact endpoint root may finish earlier. Geometric step expansion, bracketing,
best-residual output and the legacy tolerance contract are preserved.

The [reference generator](../tools/reference_dcdflib_root.py) verifies the archive
SHA256 and compiles unchanged C and F77 sources. The
[fixture](../tests/fixtures/dcdflib_root.json) records 63 calls per language with
source/driver hashes, compiler options, outputs, statuses and process termination.
A three-second timeout and 10,000-evaluation driver cap bound native execution.
Cases cover increasing/decreasing functions, constant residuals, endpoints,
single-point intervals, square/exponential/cubic equations, multiple tolerance
settings, extreme coordinate/residual scales, discontinuities and native STOP.

The [tests](../tests/test_dcdflib_root.py) verify roots independently with Decimal
arithmetic, bracket signs without multiplying tiny residuals, best-endpoint and
tolerance invariants, protocol errors, independent states, immutable snapshots
and exact work caps. Both native versions falsely report failure for the tiny
linear residual `1e-300*(x-0.23456789)` in DZROR; the shared Python engine avoids
the underflowing sign product and returns the root. Existing F95 root tests also
run to verify that its defaults and behavior remain intact.

The [benchmark](dcdflib-root-benchmark.json) reports elapsed time and evaluation
counts for independent scalar searches. It does not claim NumPy batching or a
speedup over native code. Reverse communication advances one caller-supplied
scalar residual per request; distribution-specific batched inversions remain
separate package APIs.

All 49 legacy support names now have validated Python mappings. CDFLIB90 remains
partial pending the final archive/documentation audit.
