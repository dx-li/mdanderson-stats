# CDFLIB root finding

`interval_zf` solves a continuous scalar equation within a sign-changing bracket.
`step_zf` assumes a continuous monotone equation and expands geometrically from
an initial guess before refinement. Both return `ZeroFinderResult`; inspect
`status` before using `x` as a root.

```python
from mdanderson_stats import interval_zf, set_zero_finder

state = set_zero_finder(low_limit=0, hi_limit=2, abs_tol=1e-12, rel_tol=0)
result = interval_zf(lambda x: x * x, y=2, local=state)
assert result.status == 0
assert abs(result.x - 2**0.5) <= 1e-12
```

Reverse communication requests one residual at a time. The first call has no
`fx`; each later call supplies `f(request.x) - y`, including the target subtraction.
The direct wrapper drives this same continuation, so requests and results agree.

```python
from mdanderson_stats import rc_step_zf, final_zf_state

state = set_zero_finder(low_limit=0, hi_limit=2, abs_tol=1e-12, rel_tol=0)
request = rc_step_zf(state, initial=1)
while request.status == 1:
    request = rc_step_zf(state, request.x**2 - 2)
assert request.status == 0
assert final_zf_state(state) == request
```

Each state supports one search. Create another with `set_zero_finder` to restart.
`initial` is required only on the first step-search call. Supplying an invalid
residual leaves the outstanding request available for correction. Switching modes,
resuming a finished search, or starting a direct search on a used state raises
`ValueError`. Callback exceptions propagate. Independent searches can be nested
or interleaved; concurrent mutation of one state is unsupported.

| Source interface | Python representation |
| --- | --- |
| `set_zero_finder` | Construct a configured independent `ZeroFinder` |
| `zf_locals` | `ZeroFinder`; its continuation is private |
| `interval_zf`, `step_zf` | Direct callbacks returning `ZeroFinderResult` |
| `rc_interval_zf`, `rc_step_zf` | Request/resume operations returning the same result type |
| `final_zf_state` | Latest immutable snapshot, or `None` before startup |
| `zf_current_status` | Snapshot `status` |
| `zf_bound_low`, `zf_bound_high` | Snapshot `bound_low`, `bound_high` |
| `zf_crash_left`, `zf_crash_hi` | Snapshot `crash_left`, `crash_hi` |

Status `1` requests evaluation; `evaluations` counts accepted residuals, excluding
the outstanding request. Status `0` succeeds. Exact roots collapse both bounds to
that point. Otherwise the midpoint and retained bracket are returned when its
half-width is at most `abs_tol + rel_tol*min(abs(left), abs(right))`, or no interior
binary64 point exists. Continuity and a sign change give the positional guarantee;
a discontinuity can also produce a small bracket without a root. A callback that
rounds a nonzero residual to zero limits the achievable accuracy.

Status `-1` means both endpoint residuals have the same strict sign. Bounds then
remain the configured limits; `x` identifies the indicated failed bound.
`crash_hi` means the residual is positive, and `crash_left` identifies the left
bound under the source endpoint-direction convention. Equal endpoint residuals
select the right bound. This does not prove absence of interior roots, particularly
for nonmonotone equations. Failure flags are `None` for every other status.

All settings, initial values, targets and residuals must be finite scalars. Limits
must be increasing. Tolerances are nonnegative with at least one positive; steps
are nonnegative and the multiplier exceeds one. A step search requires a positive
initial step. Defaults match CDFLIB: limits ±1e35, absolute/relative tolerances
1e-6, absolute step 1e-4, relative step 1, multiplier 2. The added positive integer
`max_evaluations` defaults to 4096. Exhaustion raises `ArithmeticError` and retains
a terminal status `-2` snapshot; no callback beyond the budget is invoked.

Refinement uses inverse quadratic/secant interpolation, normalized to reduce
intermediate overflow, with bisection safeguards ensuring bracket reduction.
This deliberately replaces the archived TOMS 748 implementation. Traces and
iteration counts are not promised to match Fortran. Endpoint values are reused
in step searches. Scalar callbacks and per-search control flow do not benefit
from NumPy broadcasting; no new compilation or concurrency dependency is added.

The [native audit](cdflib-root-reference.md) records 60 source calls. Python tests
require corrected results for all of them, including the sixteen incorrect native
successes and four stale completion states. Further tests cover direct/reverse
trace agreement, exact roots, effective tolerances, flat odd polynomials,
interleaved/nested searches, extreme residuals and bounds, adjacent floats,
immutable snapshots, input validation and exact work-budget behavior.
CDFLIB90 remains partial because other archive interfaces remain unimplemented.

For a reproducible evaluation-count check, use limits [0,2], absolute tolerance
1e-12, relative tolerance 0 and initial 1 for step search. With default step
settings, observed callback counts were:

| Residual | Interval | Step |
| --- | ---: | ---: |
| `x*x - 2` | 11 | 10 |
| `exp(x) - 2` | 9 | 9 |
| `(x - 0.123456789)**3` | 92 | 90 |

All returned roots were within 1e-12 of their analytic values. Flat equations
require more work; the safeguard bounds their progress. These are callback
counts for this implementation, not wall-clock speedups over Fortran.
