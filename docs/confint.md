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

## One-binomial calculations

For `X ~ Binomial(n,p)`, the interval is the equal-tail Clopper–Pearson
interval. The successful outcomes satisfy `width(X,n) <= max_length`.
CONFINT uses the symmetry of interval width and its increase toward the middle
count. Python searches **integer counts** on the lower half and sums the two
disjoint binomial tails with CDF/SF functions. A cutoff of zero includes the
zero/all-event outcomes; no qualifying outcomes and all qualifying outcomes
are handled separately. The native manual explicitly treats monotonicity of
Clopper–Pearson width toward the middle as an assumption; Python retains it.
Focused checks compare this search with enumeration of every count.

| API | Result |
|---|---|
| `confint_binomial_probability` | Probability that the CI has at most the requested total length; inputs broadcast |
| `confint_binomial_length` | Smallest attainable length whose cumulative probability reaches `assurance` |
| `confint_binomial_event_limit` | p* such that assurance holds for p ≤p* or p ≥1−p*; `None` if impossible, .5 if all probabilities qualify |
| `confint_binomial_sample_size` | First qualifying integer sample size, searching from 1 up to the supplied bound |

The last three functions handle one design at a time. Sample sizes must be
1–1,000,000; maximum lengths in (0,1]; event probabilities in [0,1]. Confidence
and assurance use the same ranges as the normal APIs. Probability outputs
are read-only arrays, with at most two million broadcast designs.

**Sample-size assurance need not increase monotonically with n.** Consequently,
Python scans successive sizes in batches of up to 4096, finding the first
success. The native continuous root plus rounding does not establish this
minimum. Sample-size planning is more costly than a fixed-n calculation;
it requires O(N log N) interval evaluations in the worst case. An inadequate
search bound raises an error. It does not guarantee that every larger sample
size will also attain the requested assurance.

### Corrections to the manual's worked examples

The manual's fractional-cutoff search uses a tolerance of .005 on the event
proportion, which can shift the count boundary. At n=250, p=.25, confidence=.9,
and maximum length .1, the exact qualifying lower cutoff is **77**, not 78.
Python's width probability is .9842333927, rather than the reported .9890.

For p=.7, confidence=.8, maximum length .1 and assurance .8, the manual reports
n=162. Direct Clopper–Pearson enumeration gives assurance .7501050966 at 162;
Python finds the first qualifying size **164**, with assurance .8173377771.
Independent R `qbeta`/`dbinom` enumeration confirms all three probabilities.
The Python results intentionally follow the stated interval definition rather
than the approximate reference numbers.

```python
from mdanderson_stats import (
    confint_binomial_probability,
    confint_binomial_sample_size,
    confint_binomial_length,
    confint_binomial_event_limit,
)

p = confint_binomial_probability(250, 0.1, 0.25, confidence=0.9)
assert abs(float(p) - 0.9842333927) < 1e-9
assert confint_binomial_sample_size(0.1, 0.7, confidence=0.8, assurance=0.8) == 164
length = confint_binomial_length(250, 0.25, confidence=0.9, assurance=0.9)
assert confint_binomial_probability(250, length, 0.25, confidence=0.9) >= 0.9
limit = confint_binomial_event_limit(250, 0.1, confidence=0.9, assurance=0.5)
assert abs(limit - 0.3102536151) < 1e-9
```

**Catalog status is partial.** Binomial-difference, Poisson,
exponential-survival, and native session/report workflows remain pending.
