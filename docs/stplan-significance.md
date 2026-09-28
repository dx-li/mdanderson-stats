# STPLAN significance planning for exact count tests

`stplan_exact_binomial_significance` and `stplan_exact_poisson_significance`
find a one-sided, nonrandomized rejection region attaining at least the requested
power with the smallest attainable significance. They answer the discrete
inverse-significance question in STPLAN's QBIN1 and QPOI1 routines.

For an alternative above the null, reject when `X>=critical_count`.
For an alternative below the null, reject when `X<=critical_count`. The result's
`critical_tail` identifies this direction. Equal null and alternative values
are rejected because they do not specify a test direction.

```python
from mdanderson_stats import (
    stplan_exact_binomial_significance,
    stplan_exact_poisson_significance,
)

binomial = stplan_exact_binomial_significance(
    .2, .4, 40, target_power=.8,
)
assert binomial.critical_tail == "upper"
assert binomial.critical_count == 13
assert binomial.target_attained
assert abs(binomial.significance - .04324162237632384) < 1e-12
assert abs(binomial.achieved_power - .87149032192931575) < 1e-12

poisson = stplan_exact_poisson_significance(
    2, 1, 12.5, target_power=.8,
)
assert poisson.critical_tail == "lower"
assert poisson.critical_count == 15
assert abs(poisson.significance - .022293021307365317) < 1e-12
assert abs(poisson.achieved_power - .8060290010444158) < 1e-12
```

Poisson inputs are rates and a common exposure duration, so the two count means
are `null_rate*exposure` and `alternative_rate*exposure`. Keep the rates and
exposure in consistent units. These functions solve one design at a time.
The existing forward power functions remain available for array calculations.
Binomial sample sizes must be integers from 2 through 10 million. Poisson
count means must remain positive and finite after multiplying rates by exposure,
and must be below `2**48`; unrepresentable inputs fail explicitly. The Poisson
search expands its integer bracket to cover the requested tail without
allocating a support array.

## Attainment and unattainable targets

Discrete test power changes in jumps as the rejection region grows. A returned
power above the target is expected; the result is not a continuous equation root.
Selection works in integer count space, avoiding inversion of a stepwise power
curve with a continuous root finder.

As in the native inverse routines, the permitted significance is at most
`.99999999`. If the target is unattainable under that limit, the result describes
the most powerful allowed region and sets `target_attained=False`. Its
`achieved_power` is the actual probability under the alternative, not the
requested target or a native sentinel.

```python
small = stplan_exact_binomial_significance(.2, .4, 2, target_power=.8)
assert not small.target_attained
assert small.critical_count == 1
assert abs(small.achieved_power - .64) < 1e-12
assert abs(small.significance - .36) < 1e-12
```

Here rejection on at least one event has power .64. Reaching .8 would require
rejecting every possible count, giving significance one. Increasing nominal
alpha within the permitted range cannot create an intermediate nonrandomized
test. Randomized boundary tests are a different design and are not introduced
implicitly.

The returned frozen `STPLANDiscreteSignificance` records the distribution family,
critical tail/count, attained significance, achieved and target power, and
attainment flag. The region itself is the definitive result: passing a rounded
significance back to a quantile-based forward function can select a neighboring
region at a floating-point boundary.

[Native reference evidence](../research/stplan-discrete-significance-audit.md)
records the source contract and independently executed Fortran inverse routines.
See [general inverse planning](stplan-planning.md) for continuous parameters and
sample-size searches. Automatic branch discovery for other STPLAN inverses,
proportional K-group integer allocation and native sessions/reports remain open.
