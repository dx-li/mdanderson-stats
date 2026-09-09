# CDFLIB root-finder reference audit

The [reference generator](../tools/reference_cdflib_root.py) verifies the pinned
CDFLIB90 archive hash and compiles unchanged `zero_finder` and
`biomath_constants_mod` source bytes. The [fixture](../tests/fixtures/cdflib_root.json)
records compiler flags, source and driver hashes, input configurations, evaluation
requests, answers where defined, and completion/failure state.

All 60 calls completed within the three-second per-call limit. They cover
`interval_zf`, `rc_interval_zf`, `step_zf` and `rc_step_zf`, using explicit
`zf_locals` state initialized by `set_zero_finder`. `final_zf_state` and the public
current-status variable are inspected on completion. The driver never records
undefined answer or failure fields as meaningful values.

## Independent findings

[Tests](../tests/test_cdflib_root_reference.py) solve the linear, decreasing-linear
and quadratic equations independently, using 80-digit decimal square roots for
the quadratic case. Native success is not treated as proof of correctness.

- Sixteen calls report success with a wrong answer. On [0,2], solving f(x)=1 for
  f(x)=x returns 1.5 through the interval interfaces. Solving f(x)=0 for f(x)=x
  or x*x returns near 2 for interval search and near 1 for step search starting
  at 1. Step search for f(x)=x, target 0.5, returns 0.75.
- The source fails to terminate immediately at a zero endpoint. Its candidate
  adjustment moves away from that root. For an exact interior candidate,
  `bracket` sets only the lower bound to the root; termination then returns the
  midpoint of the remaining interval rather than that exact root.
- Four step-search calls find the correct root at their initial point and return
  status 0, but `final_zf_state` reports local status 1. The public global status
  is 0. The early return leaves the local completion state stale.
- The square-root-of-two request traces are identical with configured absolute
  and relative tolerances of 1e-2 and 1e-12, for all four interfaces. The source
  interval tolerance routine uses machine epsilon; the stored common absolute
  and relative tolerance fields are not read by the solver.
- Same-sign endpoint failures do not prove that no root exists. On [-2,2],
  x*x=1 fails bracketing even though the equation has roots at -1 and 1.

The remaining successful numerical results agree with the independent roots to
within 1e-12. Every paired direct/reverse run uses the same evaluation requests
and reports the same solver and local states, including the documented defects.

## Failure outputs and remaining scope

On a direct interval failure, the INTENT(OUT) answer is undefined and the fixture
records null. Direct step search preserves its initial answer on failure.
Reverse-communication routines return the indicated bound. Failure flags and
bounds are recorded only for status -1; their sign and boundary contracts are
checked against the endpoint function values.

The [Python implementation](cdflib-root.md) uses this audit as evidence, with
corrected exact roots and completion state, effective tolerances and bounded
work. It replaces native shared default state with independent searches and
validates inputs instead of invoking the source STOP paths. Python tests cover
interleaving and invalid configuration; the native fixture itself covers explicit
local state and completed runs. CDFLIB90 and the full catalog remain partial.
