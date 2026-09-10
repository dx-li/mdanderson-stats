# CONFINT: planning for confidence-interval length

[CONFINT, catalog entry 64](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/64)
plans studies around the probability that a two-sided confidence interval will
have at most a specified **total length**. This assurance probability differs
from the confidence level of the interval itself.

The [version 2.0 archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/CONFINT/CONFINT_V2.0.zip)
contains Fortran 90 source and a manual with worked examples. The actual source
menu includes two-sample methods omitted from the site's short description.
[Source hashes](confint-sources.json) identify the inspected archive and files;
original programs and documents are not redistributed.

## Normal calculations

| `target` | Interval | Sample-size convention |
|---|---|---|
| `mean` | Student t interval for one normal mean | One sample, n ≥2 |
| `sd` | Chi-square interval for one normal population SD | One sample, n ≥2 |
| `mean_difference` | Pooled Student t interval for independent means with a common unknown variance | Two samples, each ≥2 |

`confint_normal_probability` computes the probability that the interval's total
length is at most `max_length`. `population_sd` is the true population standard
deviation, **not** the standard error. It is the shared SD for the two-sample
calculation; this method does not use Welch's unequal-variance interval.

`confint_normal_sd_limit` returns the largest population SD that attains the
requested assurance. Both APIs broadcast NumPy inputs and return read-only
arrays, including a zero-dimensional array for scalar inputs. The two-sample
calculation requires `sample_size2`; supplying it for other targets is rejected.
Up to two million designs may be evaluated together.

`confint_normal_sample_size` returns the smallest integer n attaining assurance.
For `mean_difference`, this is **per group**, with equal group sizes. Fixed
unequal sizes are supported by the other two APIs. Integer bracketing/bisection
avoids relying on the native continuous root's rounding. Failure to meet the
target within `max_sample_size` raises an error.

Sample sizes are limited to 2–100,000,000; confidence to `[1e-6,1-1e-12]` and
assurance to `[1e-12,1-1e-12]`. Length and population SD must be finite and
positive. Calculations use log length/SD ratios to avoid squaring large or tiny
physical units. A small-argument gamma expansion preserves representable
probabilities even when the chi-square argument itself underflows. SD limits
outside positive float64 range raise an error.

## Distribution and source differences

Write the interval length as `sigma * c * sqrt(X)`, where X is chi-square
with degrees of freedom `df`. The probability is therefore
`P(X <= (max_length/(sigma*c))**2)`.

- Mean: `df=n-1` and `c=2*t/sqrt(n*df)`.
- SD: `df=n-1` and `c=1/sqrt(q_low)-1/sqrt(q_high)` for equal-tail chi-square quantiles.
- Independent mean difference: `df=n1+n2-2` and `c=2*t*sqrt((1/n1+1/n2)/df)`.

Here t is the upper equal-tail Student t quantile with the relevant df.
The first two calculations agree with the manual's formulas and examples.
The two-sample source `normal2_mean_ci_mod.f90` omits the factor df in its
chi-square argument. Python restores that factor using the pooled-variance
pivot underlying the [standard pooled t statistic](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/t_test.htm).
The source's sample-size function also contains debug output and a `STOP`.
Those behaviors are not reproduced.

The original normal and distribution modules were compiled with gfortran, but
the resulting reference driver returned invalid probabilities/NaNs for ordinary
examples. Native executable parity is **not** claimed. Validation instead uses
the manual's worked results, independent R distribution calculations, and
confidence-interval widths calculated from simulated normal datasets. Tests
also check minimum integer sizes, inverse calculations, unit changes of 1e±200,
and a representable probability whose squared intermediate would underflow.

```python
from mdanderson_stats import (
    confint_normal_probability,
    confint_normal_sample_size,
    confint_normal_sd_limit,
)

# Manual: a 90% mean CI of total length <=1, population SD 2, n=45.
p = confint_normal_probability(45, 1, 2, confidence=0.9)
assert abs(float(p) - 0.5212978426) < 1e-9
assert confint_normal_sample_size(1, 2, confidence=0.9, assurance=0.9) == 57

# Manual: a 90% SD CI of total length <=0.5 with 80% assurance.
assert confint_normal_sample_size(0.5, 1, confidence=0.9, assurance=0.8, target="sd") == 30

# Two independent groups, n=20 and n=30, common population SD=1.
p = confint_normal_probability(20, 1, 1, target="mean_difference", sample_size2=30)
assert abs(float(p) - 0.0930111029) < 1e-9
sd = confint_normal_sd_limit(45, 1, assurance=[0.05, 0.5, 0.95], confidence=0.9)
assert sd.shape == (3,)
```

**Catalog status is partial.** Binomial, binomial-difference, Poisson,
exponential-survival, and native session/report workflows remain pending.
