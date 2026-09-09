# CDFLIB beta factors and shape-shift audit

The unchanged F95 `brcomp`, `brcmp1` and `bup` routines now have 149 recorded
native cases and independent mathematical checks. These three public support
interfaces remain pending implementation. CDFLIB90 remains partial.

| Procedure | Source operation |
|---|---|
| `brcomp(a,b,x,y)` | x^a y^b / B(a,b) |
| `brcmp1(mu,a,b,x,y)` | exp(mu) x^a y^b / B(a,b), with integer mu |
| `bup(a,b,x,y,n,eps)` | I_x(a,b) − I_x(a+n,b), with positive integer n; eps controls series termination |

The helpers use complementary coordinates x,y. In small-coordinate branches,
the source reconstructs the larger coordinate through log1p of the smaller one.
The independent checks therefore preserve the supplied smaller coordinate and
form its complement at high precision. This includes x=1 with a tiny positive y
and y=1 with a tiny positive x.

## Findings relevant to the port

* At a=b=1e308 and x=y=1/2, `brcomp` returns NaN because a+b overflows in its
  internal offset calculation. The mathematical factor is finite,
  approximately 2.8209479177e153. `bup` also returns NaN there for n=1,2,10,100,
  although all four differences are finite and positive.
* `brcmp1` can overflow an intermediate exponential before a small multiplier
  restores a finite result. With mu=1000, a=b=1 and x=1e-300, the native result
  is infinity; the correct result is about 1.9700711140e134. For
  a=b=1e-309,x=y=1/2, the same selector gives infinity instead of about
  9.8503555701e124.
* With mu=−1000 and a=b=1e300,x=y=1/2, the exponential underflows before its
  large multiplier is applied, discarding a representable positive result.
* With mu=−744 and a=b=1e100 or 1e300 at x=y=1/2, the exponential is rounded to
  a subnormal first and then multiplied by a large factor. The resulting
  relative error is about 28.8%. Tests reproduce this specific rounding order
  and independently evaluate the correctly combined exponent.
* The valid `bup` case a=b=1, y=1e-12 and n=2^31−1 exceeds the driver's
  three-second limit. Its exact form x(1−x^n) is finite and readily evaluated
  using log1p/expm1. The port needs to account for very large shifts without
  requiring one iteration per shift in such cases.
* `bup` does not reject n=0 or n=−1; it returns a nonzero factor. Those values
  violate its stated positive-integer shift contract. Zero and negative eps
  still produce a finite full sum for the recorded bounded shift.
* eps tests the current term against the accumulated sum. It is not a bound
  on the complete uncomputed remainder: for a=2,b=3,x=0.9,n=100, eps=1e-3
  gives about 0.936% relative error. The default-tolerance case agrees with
  the independent difference.
* Inconsistent x+y inputs have branch-dependent behavior. With x=0.1,y=0.8,
  the a=2,b=3 branch effectively uses y=0.9, while a=b=8 uses the supplied
  pair. Such inputs need an explicit complementary-coordinate contract.
* a=0 and a=−1/2 examples are retained in the fixture. For b=1 and x=y=1/2,
  they agree with the continuation a*x^a*y. These individual checks do not
  establish general negative-shape accuracy of the source approximations.

## Independent evidence

[Tests](../tests/test_cdflib_beta_factors_reference.py) use 800-digit Decimal
arithmetic and independent log-gamma recurrences/Stirling expansions for beta
factors. A local log-gamma expansion preserves tiny shapes near reciprocal-gamma
zeros, where a generic oracle approximation error could change subnormal rounding.

Shape-shift differences use finite binomial sums for integer shapes, defining
beta-integral series for the recorded fractional shapes, and the exact power
identity for b=1. At equal a=b≥1e100 and fixed n≤100, the leading density
expression has relative corrections O(n²/a), below float64 resolution. Unit-beta,
half-shape integral and large-shift power identities check the oracle separately.

[The fixture](../tests/fixtures/cdflib_beta_factors.json) stores all 148 completed
calls and one timeout, with archive/source hashes, compiler version, compiler
arguments and the driver. Nonfinite results are JSON strings. Native library and
constants sources are compiled unchanged with `-O0 -ffp-contract=off -fcheck=all`.

```sh
uv run python tools/reference_cdflib_beta_factors.py
uv run pytest tests/test_cdflib_beta_factors_reference.py
```

The archive SHA-256 remains
`2f5dd397b93546222a3b31e02073abeee1fc213cea75768c34b17e06c8264a3b`.
The existing CDFLIB legal notice remains packaged. The wheel gains no native
source files or executables.
