# CDFLIB beta continued fraction

`bfrac(a,b,x,y=None,eps=5e-15)` computes I_x(a,b) for finite a,b>1.
Either complementary coordinate can be supplied; pass x=None when supplying only
y. Both coordinates must lie in [0,1] and sum to one within eight machine
epsilons. The smaller coordinate is preserved. All inputs broadcast to owned,
immutable float64 arrays.

```python
from mdanderson_stats import bfrac

ordinary = bfrac(20, 50, 0.2)
reflected = bfrac(100, 100, 0.9)
endpoint = bfrac(15, 15, 1)  # one, repairing the native zero
huge_midpoint = bfrac(1e308, 1e308, 0.5)  # exactly one half
explicit_complement = bfrac(1e308, 15, None, 1e-308)
```

## Coordinate and convergence handling

The source's lambda=(a+b)*y-b is a redundant argument. The Python interface
derives it internally using compensated a*y-b*x products, accounting for the
rounded larger coordinate and avoiding an overflowing shape sum. To specify a
probability through a displacement instead of coordinates, use [basym](cdflib-basym.md).

Coordinates above the mean are reflected before evaluation and the resulting
tail is complemented. This repairs the audited negative-lambda inaccuracies,
values above one, and timeout. Endpoints and the symmetric midpoint are exact.
The [native audit](cdflib-beta-remaining-reference.md) remains unchanged, including
all its failures.

The source's numerator/denominator recurrence runs in batches of at most 2048
active elements, with at most 512 iterations. Algebraic regrouping avoids forming
b/a before multiplying by x. Each recurrence update is divided by its beta
coefficient before combining numerator and denominator terms. Converged elements
leave the active set. Complete logarithmic normalization combines the fraction
with the beta factor before exponentiation, retaining small probabilities.

The existing [power series](cdflib-bpser.md) handles b*x<=0.7, with an exact product
domain check. For both shapes greater than 100 and lambda<=0.03*min(a,b), the
existing asymptotic expansion handles the center, following the source driver's
routing. Other cases use the continued fraction. If a fraction becomes invalid or
has not converged after 512 iterations, the paired legacy beta adapter evaluates
the integral. An unconverged fraction is never returned.

The positive finite eps parameter is capped at 5e-15 and floored at four machine
epsilons. It controls successive-fraction convergence, not a universal forward
error bound or a correct-rounding guarantee. Invalid domains raise ValueError;
invalid numerical results raise ArithmeticError. True underflow returns zero.

## Validation and performance

[Tests](../tests/test_cdflib_bfrac.py) compare every native audit case against
independent 800-digit integrals. Broader tests use positive binomial sums,
midpoint integrals and exponential bounds, covering shapes near one, large and
unequal shapes, explicit subnormal coordinates, underflow, endpoints, algorithm
transitions, tolerance extremes, broadcasting and batches exceeding 2048 values.

This work also exposed and repaired a pre-existing cancellation error in the
shared beta factor. Normal coordinate ratios are now logged directly; subnormal
ratios retain separate logarithms. Direct independent factor tests cover this
repair in both orientations, including shapes paired with 1e308.

[Benchmarks](cdflib-bfrac-benchmark.json) compare a broadcast call with repeated
scalar calls and require identical outputs. They measure batching benefits,
not speed relative to Fortran.

The later [paired beta integral](cdflib-bratio.md) completes all 35 F95
mathematical procedures. Their [constants](cdflib-constants.md) are also implemented.
See the [CDFLIB90 completion audit](cdflib90-completion.md) for the
complete library scope.
