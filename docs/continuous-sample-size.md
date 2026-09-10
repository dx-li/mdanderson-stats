# Continuous-endpoint sample-size planning

Catalog entry **119**, MD Anderson's
[Nnormal application](https://biostatistics.mdanderson.org/shinyapps/Nnormal/),
is implemented for all its statistical test families. Five source documents are
recorded in [provenance](continuous-sample-size-sources.json). This independent
implementation uses NumPy broadcasting, SciPy distribution kernels, and integer
searches that verify power at the chosen enrollment and its predecessor.

## Mean comparisons, including paired data

```python
from mdanderson_stats import normal_mean_sample_size, normal_mean_power

one = normal_mean_sample_size(0.3, sd=1, sides=1)
assert one.group_sizes == (71,)

two = normal_mean_sample_size(0.3, sd=1, allocation_ratio=1, sides=1)
assert two.group_sizes == (139, 139)
print(two.power, two.previous_power)  # 0.80234, 0.79982

paired = normal_mean_sample_size(0.5 - 0.3, sd=0.5, sides=2)
assert paired.group_sizes == (52,)
```

`difference` is treatment minus control. For paired data, pass the mean paired
difference and **SD of individual paired differences**, not an arm's marginal
SD or the standard error. A standardized effect can be supplied with `sd=1`.

With no `allocation_ratio`, enrollment is for one sample or a set of pairs;
historical controls are not newly enrolled. With a ratio, the result's two
`group_sizes` are control and treatment. The ratio means **treatment/control**.
For each integer control size, treatment size is `max(2, ceil(ratio*control))`.
Calculations use these actual sizes, a common SD, and pooled residual degrees of
freedom; this is not a Welch unequal-variance design. Ratios range from `1e-6`
to `1e6`. Every estimated variance requires at least two observations per group.

The mean-test objectives are:

| `objective` | Alternative, with difference d and positive margin M |
| --- | --- |
| `equality` | Nonzero d; `sides=1` chooses the direction of d, `sides=2` either direction |
| `equivalence` | -M < d < M, using two one-sided tests (TOST) |
| `noninferiority` | d > -M; larger outcomes are favorable |
| `superiority` | d > M; larger outcomes are favorable |

Margin tests use one-sided alpha; `sides` applies only to equality. To plan for
smaller favorable outcomes in a directional margin test, reverse the outcome's
sign first. Sample-size searches require an effect inside the alternative.
Power evaluation also accepts effects outside it.

```python
equivalence = normal_mean_sample_size(
    0.15, sd=0.1, objective="equivalence", margin=0.2, allocation_ratio=1
)
assert equivalence.group_sizes == (51, 51)

power_curve = normal_mean_power(
    [20, 30, 40], 0.15, sd=0.3, objective="equivalence", margin=0.3, exact=True
)
```

By default, equality planning uses the noncentral-t upper tail in the direction
of the effect (alpha/2 for two-sided tests). Equivalence uses the nonnegative
lower bound `max(0, 1 - failure_left - failure_right)`. Both are conservative
relative to the full rejection region. Noninferiority and superiority powers
already use their exact normal-model noncentral-t distributions.

`exact=True` adds the opposite tail for two-sided equality. For equivalence, it
integrates the actual TOST rejection interval conditional on the residual
chi-square variance. This preserves positive power when the conservative bound
is zero. Integration requests absolute error `2e-10` and rejects estimated
errors above `1e-8`; "exact" describes the statistical model, not symbolic
arithmetic. This optional calculation loops over broadcast scenarios; the
ordinary distribution calculations are vectorized.

## Correlation and balanced ANOVA

```python
from mdanderson_stats import (
    correlation_sample_size,
    correlation_power,
    anova_sample_size,
    anova_effect_size,
    anova_power,
)

assert correlation_sample_size(0.4, method="t").group_sizes == (46,)
assert correlation_sample_size(0.4, method="z").group_sizes == (47,)
print(correlation_power([30, 40, 50], 0.4))

assert anova_sample_size(0.25, groups=4).group_sizes == (45, 45, 45, 45)
effect = anova_effect_size([-0.5, 0, 0.2, 0.8], sd=1)
print(anova_power([10, 20, 30], effect, groups=4))
```

Both correlation methods are **Fisher-transformation approximations**, not exact
correlation-test power. The z method uses Fisher location `atanh(abs(r))` and
standard error `1/sqrt(n-3)`. The t method transforms a Student-t critical value
and adds the Pearson–Hartley location correction `abs(r)/(2*(n-1))`. Two-sided
planning retains the dominant tail in both methods. Correlation sample sizes
count paired observations and start at four.

ANOVA uses an upper-tail noncentral F distribution with degrees of freedom
`m-1, m*(n-1)` and noncentrality `m*n*f**2`. Here n is **per group**, and Cohen's
f is the population RMS spread of the group means divided by common within-group
SD. `anova_effect_size` uses the final array axis for group means and broadcasts
over scenarios. Constant means give f=0 and null F-test power equals alpha.

All searches default to target power .8, alpha .05 and a maximum reference size
of ten million. Alpha must be in (0, .5). `max_size` bounds single/control size,
paired-observation count, or ANOVA per-group size. Failure to reach target within
that bound raises an error. `previous_power` is absent only at the smallest valid
size. `total_size` sums actual group sizes.

## Source discrepancies and validation

The [one-sample](https://biostatistics.mdanderson.org/shinyapps/Nnormal/One-sample-help.pdf)
and [paired](https://biostatistics.mdanderson.org/shinyapps/Nnormal/Paired-sample-help.pdf)
documents supply the dominant-tail planning convention. The
[two-sample document](https://biostatistics.mdanderson.org/shinyapps/Nnormal/Two-sample-help.pdf)
has inconsistent allocation wording, equivalence signs and scaling. We use the
application's treatment/control convention, pooled standard errors and level-alpha
TOST, consistent with its worked examples. The
[ANOVA document](https://biostatistics.mdanderson.org/shinyapps/Nnormal/Multiple-sample-help.pdf)
labels a CDF as power; we use the upper rejection tail. The
[correlation document](https://biostatistics.mdanderson.org/shinyapps/Nnormal/Correlation-help.pdf)
mislabels the Fisher standard error as a variance; its subsequent sample-size
formulas use the correct scaling. These are document discrepancies, not claims
that the live application computes those erroneous expressions.

Fifteen focused tests reproduce all twelve worked sample-size outputs: one- and
two-sample equality, equivalence, noninferiority and superiority; paired data;
both correlation methods; and ANOVA. Independent simulations from raw normal
observations check the exact TOST and ANOVA probabilities. Further checks cover
null power, unequal integer allocation, unit scaling, predecessor power and
infeasible designs. Statistical calculations and power curves are available as
Python APIs; the original Shiny interface and report formatting are not copied.
