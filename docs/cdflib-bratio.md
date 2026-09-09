# CDFLIB paired beta integral

`bratio(a,b,x,y=None)` returns `(I_x(a,b), I_y(b,a))` as independently owned,
immutable float64 arrays. All inputs broadcast. Shapes are finite and nonnegative;
at least one must be positive. Either complementary coordinate can be supplied.

```python
from mdanderson_stats import bratio

p, q = bratio(2, 3, 0.5)
left_limit = bratio(0, 3, 0.5)  # (1, 0)
right_limit = bratio(3, 0, 0.5)  # (0, 1)
tiny_p, rounded_q = bratio(100, 1e-20, 0.001)
rounded_p, tiny_q = bratio(1e-20, 100, None, 0.001)
huge_midpoint = bratio(1e308, 1e308, 0.5)  # (0.5, 0.5)
```

## Source boundary and error contracts

Coordinates lie in [0,1]. Explicit pairs must sum to one within three machine
epsilons, retaining the source tolerance. The smaller coordinate is preserved
when reconstructing the larger one. A zero first shape gives (1,0), except that
a=x=0 is undefined. A zero second shape gives (0,1), except that b=y=0 is undefined.
For positive shapes, x=0 gives (0,1) and y=0 gives (1,0).

The source's IERR=1 through IERR=7 input errors become ValueError, with the source
code in the message: negative shapes, two zero shapes, invalid x, invalid y,
inconsistent coordinates, a=x=0 and b=y=0 respectively. Nonfinite inputs and
missing coordinates also raise ValueError. Invalid numerical results propagate
ArithmeticError from the evaluators. Inputs are never modified, and a batch with
an invalid entry raises rather than fabricating a probability pair.

## Preserving both tails

For both shapes above one, a compensated displacement identifies the side of the
mean. The [bounded fraction](cdflib-bfrac.md), including its power-series and
asymptotic routes, evaluates that side. Reflection restores the original order.
This handles all three huge-shape cases that time out in the native bratio audit.

For a shape at or below one, evaluation starts with the smaller coordinate.
The [direct upper-tail series](cdflib-apser.md) handles its exact tiny-first-shape
domain; the [power series](cdflib-bpser.md) handles its exact product domain and
second shapes at or below one. A lower probability at or below one half can be
complemented safely. Otherwise the paired legacy evaluator supplies a direct
upper tail. General cases also use that paired evaluator.

This dispatch repairs a lost tail at (a,b,x)=(100,1e-20,0.001): the legacy evaluator
alone returns (0,1), while the independent beta integral has a representable lower
tail of about 1e-322. Reflection retains the corresponding tiny upper tail too.
Small probabilities are not recovered by subtracting a rounded unit probability.
True underflow is allowed; correct rounding is not promised for every input.

## Validation and performance

[Tests](../tests/test_cdflib_bratio.py) cover all 47 native calls, including their
input statuses, and compare both tails against independent 800-digit integrals.
They cover zero and subnormal shapes, unit-shape identities, large positive
binomial sums, explicit subnormal complements, huge shapes, endpoints, the exact
coordinate-tolerance boundary, independent output ownership and mixed batches.
A finite positive integer-companion identity checks enormous unequal shapes;
separate moderate-shape integral checks validate that oracle too.

[Benchmarks](cdflib-bratio-benchmark.json) compare broadcast calls with repeated
scalar calls and require identical paired outputs. They include ordinary,
small-shape, subnormal-shape, subnormal-tail, huge-shape and tiny-upper-tail cases.
These measure batching benefits, not speed relative to Fortran.

All 35 F95 mathematical procedures and their [constants](cdflib-constants.md)
are implemented. Root-finder/state, console and adapter interfaces, and other
CDFLIB support scope remain open. CDFLIB90 stays partial, and the full software
catalog conversion continues.
