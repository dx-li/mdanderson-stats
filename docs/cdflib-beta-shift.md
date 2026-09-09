# CDFLIB incomplete-beta shape shifts

`bup(a,b,x,y=None,n=1,eps=5e-15)` returns
I_x(a,b) − I_x(a+n,b). Shapes must be finite and positive, and n must be a
positive signed int32 integer. Inputs broadcast; output float64 arrays are
independently owned and immutable. The existing complementary-coordinate
validator preserves the smaller supplied coordinate. Exact endpoints return zero.

```python
from mdanderson_stats import bup

ordinary = bup(2, 3, 0.9, n=100)
large_shift = bup(1, 1, 1 - 1e-12, 1e-12, n=2**31 - 1)
huge_center = bup(1e308, 1e308, 0.5, n=100)
slowly_varying = bup(1e12, 8, 1 - 1e-12, 1e-12, n=2**31 - 1)
```

## Positive finite sums

The difference is the sum of n positive terms
`t[k] = x**(a+k) * y**b / ((a+k)*Beta(a+k,b))`.
Successive terms have ratio `x*(a+b+k)/(a+1+k)`. The implementation locates the
largest term and sums outward in NumPy blocks of 128 terms. Compensated
coordinate products retain ratios near one without overflowing a+b.

A finite-count bound and a geometric bound control the uncomputed remainder.
For b<1, increasing forward ratios are bounded by their limit x. For b>1,
ratios decrease away from the peak on either side. This replaces the source's
current-term stopping test, which can substantially underestimate the remainder.

The peak is evaluated using the stable beta-factor foundations. Its division
by the shape and multiplication by the complete sum are combined in logarithms
when the intermediate quotient is subnormal. An early zero requires the upper
bound n times the largest term to underflow too.

## Large shifts

For b=1, the exact expression `x**a * (1-x**n)` uses log1p/expm1. This repairs
the archived valid int32-maximum shift that exceeded the native time limit.

For larger shifts, existing compiled beta tails provide the difference only
when it is at least 5% of the smaller available subtraction scale. The lower
or upper tail is selected to reduce cancellation. This path is used for
`a+n <= 1e14` and `b <= 1e10`; these are evaluation-path limits, not input-domain
limits. Tests extend through tiny positive first shapes and large, disparate
shapes using independent finite beta-integral formulas.

When recurrence ratios change slowly, a geometric sum replaces either the
whole interval or bounded blocks. The absolute derivative of the log ratio is
`abs(b-1)/((a+1+k)*(a+b+k))`, maximized at k=0. Block lengths bound the resulting
log-weight error using that curvature and the square of the block length.
Blocks use vectorized beta factors and logarithmic accumulation, retaining
results even when individual terms would underflow. This also handles huge
shapes for which adding n does not change the rounded shape.

## Accuracy and work limits

`eps` must be positive. Recurrence tolerance is capped at 5e-15 for loose
requests and floored at four machine epsilons. It controls truncation and
geometric approximation, not a guarantee of correctly rounded output or a
bound on all floating-point/kernel errors. Unlike the source, zero/negative
n or eps are rejected.

Each outward recurrence has a limit of 524,288 terms; geometric evaluation
allows up to one million blocks, processed in groups of 8,192. Failure to
satisfy the remainder bound within those work limits raises `ArithmeticError`.
The routine does not return a partial sum as a converged result. Arbitrary
extreme combinations outside the tested cases can still fail numerically.

## Validation

[Tests](../tests/test_cdflib_beta_shift.py) compare the archived shift cases with
independent high-precision beta integrals, including the native timeout,
overflow and loose-tolerance failures. A finite polynomial obtained by
integrating the beta density by parts validates real first shapes with integer
companions, shifts through int32 maximum and subnormal complementary coordinates.
Fractional-shape integral series, huge-center asymptotics and tiny-companion
digamma coefficients provide additional independent checks.

[Benchmarks](cdflib-beta-shift-benchmark.json) compare batches with repeated
scalar calls to this Python API, requiring identical results. They do not
measure speed relative to Fortran.

The later [paired beta integral](cdflib-bratio.md) completes all 35 F95
mathematical procedures. Their [constants](cdflib-constants.md) are also implemented.
Other CDFLIB support interfaces and the rest of the catalog remain open.
CDFLIB90 remains partial.
