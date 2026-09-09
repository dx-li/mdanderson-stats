# CDFLIB remaining beta-helper audit

The unchanged F95 `basym`, `bfrac`, `bgrat` and `bratio` routines were compiled
and called 116 times: 111 calls completed and five exceeded the three-second
per-call timeout. Completion does not imply a valid numerical result.

[The fixture](../tests/fixtures/cdflib_beta_remaining.json) retains archive and
source hashes, compiler/version/options, driver text, inputs, results, native
status codes and execution outcomes. No source adaptations were made.
[The generator](../tools/reference_cdflib_beta_remaining.py) uses `-O0`,
`-ffp-contract=off` and `-fcheck=all`, matching the prior numerical audits.

## Independent expectations

[Tests](../tests/test_cdflib_beta_remaining_reference.py) use beta integrals,
exact endpoints, symmetry, and the seven documented bratio input-error statuses.
The existing 800-digit integral oracle now accepts a Decimal coordinate, so
basym's coordinate is reconstructed as (a-lambda)/(a+b) without an intermediate
float64 rounding. Other calls preserve the smaller supplied complementary
coordinate.

The oracle's signed-series domain extends to x<=0.5 and b*x<=100. In this bounded
region, cancellation consumes far fewer digits than the 800-digit working
precision. Independent positive binomial sums for integer beta shapes validate
the extension, including shapes (100,100). Reflection computes small upper tails
directly. For extreme shapes outside that series domain, an upper bound obtained
by bounding (1-t)**(b-1) over [0,x] proves underflow; an unsupported case raises
an assertion rather than silently substituting a value. Symmetric midpoints are
exactly one half. Near symmetric midpoints, a separate binomial expansion of the
density about x=0.5 retains displacements too small for float64 coordinates. Its
integral is independently cross-checked against positive binomial sums for shapes
15, 100 and 1000.

## Findings

- **basym:** lambda=a represents x=0 and returns NaN in all four ordinary-shape
  endpoint probes. At (a,b,lambda)=(50,15,25), the relative error is about 1.06e-8,
  despite eps=5e-15. This is outside the near-center region selected by bratio's
  0.03-shape checks, but satisfies basym's stated a,b>=15 and nonnegative-lambda
  assumptions. The equal-shape midpoint remains finite even at a=b=1e308, but
  lambda=1e146 incorrectly produces 0.5 instead of approximately 0.49999999435810416.
  An underflowing squared intermediate loses the displacement. A Python port must
  retain lambda's information rather than reconstructing a rounded x=0.5.
- **bfrac:** x=1 returns zero instead of one. Negative-lambda inputs expose
  inaccurate values, a value greater than one, and nontermination. Its header
  states a,b>1 and the lambda relation without a nonnegative-lambda restriction;
  bratio internally reflects parameters before calling it. A separate call at
  a=b=1e308,x=y=0.5,lambda=0 also times out, so reflection alone is insufficient.
- **bgrat:** x=0 returns NaN with ierr=0; x=1 leaves the initial accumulator
  unchanged with ierr=1, although the mathematical increment is one. Some huge-a
  interior calls return NaN or ierr=1 where a bound proves that the increment
  underflows. Ordinary initial accumulators 0, 0.25, -1 and 1 demonstrate that
  the result is added to the supplied value; it is not merely a standalone CDF.
- **bratio:** either individual zero shape is accepted in the interior, returning
  its degenerate probability pair. Both zero shapes, invalid coordinates,
  inconsistent complements and the two zero-shape endpoint corners produce the
  documented statuses 1 through 7. At a=b=1e308, all three interior probes
  x=0.1,0.5,0.9 time out; their rounded pairs are respectively (0,1), (0.5,0.5),
  and (1,0), established by bounds and symmetry.

The fixture records failures explicitly instead of treating native values as
reference truth. The later [bgrat](cdflib-bgrat.md) and [basym](cdflib-basym.md) ports implement
the accumulated increment and displacement-preserving tail; bfrac and bratio
remain pending. CDFLIB90 remains partial, with 33 of 35
F95 mathematical procedures implemented and other support interfaces and catalog
entries still outstanding.
