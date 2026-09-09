# Legacy normal and t quantile helpers

`dcdflib_support` exposes `stvaln`, `dinvnr` and `dt1`. The first and third are
historical starting approximations; `dinvnr` is the refined normal inverse.
All three support NumPy batches and return owned immutable float64 arrays.

```python
from mdanderson_stats import dcdflib_support as legacy

initial = legacy.stvaln([0.25, 0.5, 0.75])
refined = legacy.dinvnr([0.25, 0.5, 0.75])
assert initial[1] != 0 and refined[1] == 0
upper_extreme = legacy.dinvnr(None, 5e-324)
t_initial = legacy.dt1([0.25, 0.75], None, 10)
```

## Normal starting approximation

`stvaln(p)` preserves the archived Kennedy–Gentle rational expression in
`sqrt(-2*log(min(p,1-p)))`, including its original coefficients. The sign follows
the selected tail. It requires finite 0<p<1 and rejects endpoints instead of
returning native NaNs. The approximation at p=0.5 is approximately -1.49e-8;
this is retained because the routine supplies a starting value, not the refined
inverse. Independent Decimal polynomial evaluation checks its mathematical formula.

## Refined normal inverse

`dinvnr(p, q=None)` reuses the validated [legacy normal inverse](dcdflib-normal.md),
including its forward check. Pass `p=None` to supply only q. Both completed tails
must be positive; explicitly supplied pairs must sum to one within three machine
epsilons. The smaller supplied probability is preserved while the larger is
reconstructed. This supports tiny tails whose complements round to one.

The implementation uses the existing compiled quantile kernel rather than
reproducing the native Newton loop's fallback to an inaccurate starting value.
The median is exactly zero. Minimum-subnormal tail probabilities are supported
and independently verified in the log domain. Although the native header mentions
a machine-epsilon clamp, the executable source does not apply that clamp.
No endpoint infinity is returned: zero tails have no finite quantile and raise
`ValueError`. Expected subnormal arithmetic during forward verification is allowed;
nonfinite kernel results and failed verification remain explicit errors.

## Student t starting approximation

`dt1(p, q, df)` preserves the source's four-term expansion in inverse df, with
positive finite df and the same probability contract as `dinvnr`. Either p or q
can be `None`. For signed normal deviate z and w=z², it evaluates

```text
z * [1 + (1+w)/(4*df)
       + (3+16*w+5*w²)/(96*df²)
       + (-15+17*w+19*w²+3*w³)/(384*df³)
       + (-945-1920*w+1482*w²+776*w³+79*w⁴)/(92160*df⁴)]
```

It is not an exact t quantile and does not have a uniform accuracy guarantee over
all positive df. For example, at p=0.25, df=1 it returns about -0.98345 rather than
the exact Cauchy quantile -1. At p=0.49, df=0.1 it returns about +3.386, even though
the true quantile is negative. These are properties of the specified starting
expansion, and the tests preserve them. Use the [legacy Student t distribution
inversion](dcdflib-t.md) when a refined quantile is needed.

The Python calculation starts each correction with its z factor, divides by df
successively, and uses compensated addition across terms. It does not form
`df**4`, which can underflow while the final correction still fits float64.
For p just above 0.5 and df=1e-80, the finite expansion is retained. At the exact
median every correction is zero, even for minimum-positive df. Unrepresentable
results raise `ArithmeticError` instead of returning NaN or infinity.

This rearrangement retains the formula, not native intermediate rounding.
Relative accuracy can deteriorate near cancellation zeros of the expansion.
Inputs are not mutated. Invalid domains or nonbroadcastable inputs raise
`ValueError`, and a failure in any batch member fails the call explicitly.

## Validation and performance

The [reference generator](../tools/reference_dcdflib_quantile_helpers.py) compiles
unchanged, SHA256-verified C and F77 sources. The [fixture](../tests/fixtures/dcdflib_quantile_helpers.json)
records 85 calls per language, source/driver hashes and compiler settings. Each
call has a three-second limit. Cases cover ordinary probabilities, both subnormal
tail orientations, values adjacent to the median, endpoints, fractional df and
extreme df. Nonfinite native outputs are retained as strings for audit.

The [tests](../tests/test_dcdflib_quantile_helpers.py) compare the native ordinary
contracts, evaluate the normal rational formula and t expansion independently
with Decimal arithmetic, and verify the refined normal inverse using an
independent erfcx series and log-tail identity. Further checks cover exactly zero
medians, finite results despite underflowing df powers, explicit overflow,
broadcasting, complementary inputs and output ownership. The t polynomial oracle
uses the separately verified normal deviate as its input.

The [benchmark](dcdflib-quantile-helpers-benchmark.json) compares one batch with
repeated scalar calls to this same Python API, requiring identical results.
It does not claim a speedup over C or Fortran. The implementation uses batched
NumPy operations, compiled normal kernels and fixed loops over polynomial terms.

These helpers reconcile 44 of 49 legacy support names. The subsequent
[incomplete-gamma inverse](dcdflib-gamma-inverse.md) brings the total to 45.
The [legacy root finders](dcdflib-root.md) complete all 49 mappings.
See the [CDFLIB90 completion audit](cdflib90-completion.md) for the
complete library scope.
