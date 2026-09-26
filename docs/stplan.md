# STPLAN study planning

STPLAN 4.5 (catalog entry 41) combines power and inverse planning procedures for
binary, count, continuous, survival, and correlation outcomes. Python coverage is
partial. Seventeen power and retention procedures are available as independent mathematical
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

## Binary and count outcomes

```python
from mdanderson_stats import (
    stplan_exact_binomial_power,
    stplan_arcsine_binomial_two_sample_power,
    stplan_retention_probability,
)

exact_power = stplan_exact_binomial_power(0.2, 0.4, 40)
approximate_power = stplan_arcsine_binomial_two_sample_power(0.2, 0.4, 40, 60)
retention = stplan_retention_probability(100, 80, 0.05, 3)
```

| Python function | Interpretation |
| --- | --- |
| `stplan_arcsine_binomial_two_sample_power(probability1, probability2, n1, n2)` | Normal approximation after the `2 * asin(sqrt(p))` variance-stabilizing transform; each sample size at least 2. |
| `stplan_median_split_power(overall_probability, delta, total_size)` | Two equal halves with probabilities `overall - delta` and `overall + delta`; total size at least 4. |
| `stplan_historical_binomial_power(control_probability, experimental_probability, control_size, experimental_size)` | Historical-control arcsine approximation: the critical boundary includes both sample variances, while alternative variability comes from the experimental sample. |
| `stplan_responder_normal_approximation_power(conservative_probability, expensive_probability, conservative_size, expensive_size, margin)` | Probability of demonstrating that the expensive treatment's response advantage is at most `margin`; keyword `confidence=0.95`. |
| `stplan_binomial_k_sample_power(probabilities, sample_sizes)` | Noncentral chi-square approximation; groups occupy the final array axis and the default is `sides=2`. |
| `stplan_retention_probability(initial_size, minimum_remaining, dropout_rate, duration)` | Binomial probability that at least the requested number remain, using individual retention `(1 - dropout_rate)**duration`. |
| `stplan_fisher_exact_approx_power(probability1, probability2, sample_size)` | The original option named “Fisher exact” uses a Casagrande–Pike–Smith normal approximation, with equal group sizes. **This is not exact Fisher-test power.** |
| `stplan_exact_binomial_power(null_probability, alternative_probability, sample_size)` | Exact nonrandomized binomial rejection-tail power, selected by integer bisection. |
| `stplan_exact_poisson_power(null_rate, alternative_rate, exposure)` | Exact nonrandomized Poisson-rate rejection-tail power, also selected by integer bisection. |

These functions share the bounded broadcasting convention above. Apart from
retention, responder confidence, and the K-sample exception below, significance
arguments are `alpha=0.05, sides=1` and two-sided planning halves alpha. Exact
binomial sample sizes and retention counts are integers; approximate planning
sizes can be fractional. Exposure and duration use the same time units as rates.
The attrition rate is a per-period dropout probability, not an exponential hazard.

The K-sample default uses the ordinary chi-square cutoff at `alpha`, with
noncentrality `sum(n * (p - pooled_p)**2) / (pooled_p * (1 - pooled_p))`.
For two groups only, native `sides=1` uses a chi-square cutoff at `2 * alpha`.
This special rule does not mean that every K-sample calculation halves alpha.
Python rejects `sides=1` for more than two groups. More than the native menu's
ten groups are supported within the array-size limit.

Exact tests choose the largest one-sided rejection region whose null probability
is at most `alpha / sides`. The direction follows the alternative; equality uses
the lower tail, matching the source. Empty regions have power zero. The original
returns a `-1` sentinel instead, and silently truncates fractional binomial sizes;
Python requires integral sizes. Binomial probabilities may include 0 and 1, and
Poisson rates may be zero. Poisson means must remain below `2**48` to retain
count resolution; unbracketable tails raise an error.

The Fisher approximation reproduces the source's absolute continuity correction,
`abs(n * abs(p2 - p1) - 1) / sqrt(n)`. For small effects or sample sizes it can
behave nonmonotonically, even producing power above alpha when the two
probabilities are equal. These are properties of the native approximation,
not evidence of an exact test's operating characteristics. The Python formula
also accepts positive planning sizes below the original menu's minimum of two.

The responder calculation evaluates the documented normal formula even when
`margin - expensive_probability + conservative_probability` is zero or negative.
The original interactive routine rejects slack at or below `1e-8`; Python's
extension is useful for showing low power outside the favorable alternative.
Its `confidence` must be strictly between 0.5 and 1.

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

SciPy evaluates distribution tails directly. The exponential two-sample method
uses a reciprocal-F lower-tail calculation so that tiny significance levels
(such as `1e-20`) do not disappear in subtraction from one. The original inverse routines silently
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
bounds. A further 25 native reference cases cover all nine binary/count/retention
procedures, with explicit treatment of the two native empty-region sentinels.
The second probe is `tools/reference_stplan_discrete.f90`. Original software and
source files are not redistributed.

Still open: case-control and matched case-control methods, two-sample Poisson
power, all censored-survival procedures, inverse calculations for sample
size/effect/significance, and native
session/report workflows. The old matched-pairs option is present in the archive
but commented out of the current main menu; it will be tracked separately from
active menu features. See [source provenance](stplan-sources.json) for the archive
identity and reproduction inputs.
