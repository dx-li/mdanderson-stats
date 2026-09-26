# STPLAN study planning

STPLAN 4.5 (catalog entry 41) combines power and inverse planning procedures for
binary, count, continuous, survival, and correlation outcomes. Python coverage is
partial. The first eight power procedures are available as independent mathematical
implementations with comparison against the original Fortran routines.

## Continuous and correlation power

```python
from mdanderson_stats import (
    stplan_normal_one_sample_power,
    stplan_lognormal_two_sample_power,
    stplan_correlation_two_sample_power,
)

# User-manual example: mean 100 versus 101, SD 10, and 156 observations.
power = stplan_normal_one_sample_power(1, 10, 156)  # approximately 0.344

# Arithmetic means and common coefficient of variation (SD / mean).
lognormal_power = stplan_lognormal_two_sample_power(100, 80, 0.35, 15, 22, sides=2)
correlation_power = stplan_correlation_two_sample_power(0, 0.5, 20, 30, sides=2)
```

All power functions return NumPy arrays, including zero-dimensional arrays for
scalar input. Numeric inputs broadcast, with at most 200,000 input values and
200,000 broadcast cases per call. Continuous planning sample sizes may be
fractional; integer enrollment rounding is the caller's responsibility.

| Python function | STPLAN procedure and parameters |
| --- | --- |
| `stplan_normal_one_sample_power(difference, sd, sample_size)` | Unknown-variance normal mean, noncentral t, `n - 1` degrees of freedom; `n >= 2`. |
| `stplan_normal_two_sample_power(difference, sd, n1, n2)` | Independent normal means with common SD, pooled t; each size at least 2. |
| `stplan_welch_two_sample_power(difference, sd1, sd2, n1, n2)` | Unequal normal variances, Satterthwaite approximation; each size at least 2. |
| `stplan_lognormal_two_sample_power(mean1, mean2, cv, n1, n2)` | Log-normal observations, arithmetic means and common positive CV; equal-variance t on the log scale. |
| `stplan_exponential_one_sample_power(ratio, sample_size)` | Uncensored exponential observations; `ratio = alternative mean / null mean`; exact chi-square pivot. |
| `stplan_exponential_two_sample_power(mean1, mean2, n1, n2)` | Uncensored exponential observations; exact F pivot with the larger mean in the numerator. |
| `stplan_correlation_one_sample_power(null_correlation, alternative_correlation, sample_size)` | Current source's corrected Fisher-transform approximation; correlations strictly between -1 and 1 and `n >= 4`. |
| `stplan_correlation_two_sample_power(correlation1, correlation2, n1, n2)` | Two independent bivariate-normal samples with the corrected Fisher approximation; each size at least 4. |

Each function accepts keyword arguments `alpha=0.05` and `sides=1`, with
`0 < alpha < 0.5` and `sides` equal to 1 or 2. The alternative direction follows
the sign of the effect. As in STPLAN, `sides=2` halves the significance level and
calculates only rejection in the direction of the alternative. It **omits the
opposite rejection tail**. Consequently, equal-mean continuous designs report
`alpha / 2` under the null when `sides=2`; these values should not be described as
full two-tailed rejection probabilities.

## Source conventions and numerical evaluation

The log-normal conversion uses log-scale SD `sqrt(log(1 + cv**2))` and the
logarithm of the arithmetic-mean ratio. CV means SD divided by mean, as in the
source routine; a reversed definition in the methods manual is inconsistent with
its formula. Python evaluates the conversion in the log domain to avoid squaring
large CVs. Welch variances are scaled before computing the degrees of freedom.

For correlation `rho` and sample size `n`, the current source uses transformed
mean `atanh(rho) + rho / (2 * (n - 1))` and variance
`1 / (n - 1) + (4 - rho**2) / (2 * (n - 1)**2)`. The one-sample calculation uses
the **null** standard deviation for its standardized difference. The two-sample
calculation uses the sum of transformed variances. These choices reproduce the
current implementation, including its unequal-size bias correction: equal
nonzero correlations with unequal sample sizes need not give exactly the nominal
significance level. Some older formulas in the methods document differ.

SciPy evaluates distribution tails directly. The original inverse routines silently
clamp some probabilities to `[1e-8, 1 - 1e-8]`; Python does not reproduce that clamp.
Results outside the original numerical range therefore are not claimed to match
native output. Invalid inputs raise errors, and unrepresentable probabilities
raise `ArithmeticError` instead of returning a fabricated result.

## Validation and remaining coverage

`tests/fixtures/stplan-continuous.csv` contains 23 outputs from compiled original
STPLAN 4.5 routines, including fractional planning sizes, unequal variances,
both exponential effect directions, and equal-correlation unequal-size cases.
`tools/reference_stplan_continuous.f90` is the independent probe used to generate
these values. The tested maximum absolute difference is approximately `1.8e-8`;
comparison tolerances allow the original inverse solvers' `1e-7` tolerance.
Focused mathematical checks also cover the published normal example and input
bounds. Original software and source files are not redistributed.

Still open: binomial and Poisson methods, attrition, all censored-survival
procedures, inverse calculations for sample size/effect/significance, and native
session/report workflows. The old matched-pairs option is present in the archive
but commented out of the current main menu; it will be tracked separately from
active menu features. See [source provenance](stplan-sources.json) for the archive
identity and reproduction inputs.
