# STPLAN case-control and two-sample Poisson power

These APIs extend the [STPLAN planning methods](stplan.md). They implement the
current source's disease-risk parameterization and the intended conditional tests.
The unmatched calculation agrees with the original mathematical routine. The
matched and two-sample Poisson calculations intentionally correct source defects,
so their results are not presented as exact native-output reproductions.

## Case-control studies

```python
from mdanderson_stats import (
    stplan_case_control_power,
    stplan_matched_case_control_power,
)

# Population exposure prevalence, disease risk when exposed/unexposed,
# then the numbers of sampled cases and controls.
unmatched = stplan_case_control_power(0.3, 0.1, 0.05, 60, 60)
matched = stplan_matched_case_control_power(0.3, 0.1, 0.05, 60)
```

For exposure prevalence `f` and disease risks `re` and `ru`, the disease
prevalence is `f * re + (1 - f) * ru`. Bayes' rule gives the exposure prevalence
among cases and among controls. Both groups must have positive population
probability; a case-control study cannot be defined when disease is impossible
or universal. These risks are probabilities, not a risk ratio. The older public
manual retains a rare-disease relative-risk interface, while the current source
accepts the two risks explicitly.

The unmatched calculation applies the arcsine normal approximation to those two
exposure prevalences. Its default one-sided alternative is **greater exposure
among cases**. Reversing the disease risks therefore lowers its power. Case and
control planning sizes may be fractional and must each be at least two.

The matched calculation assumes independent exposure draws from the case and
control populations within each pair, as in the source model. This does not
model additional within-pair exposure correlation from a matching process.
If their exposure probabilities are `pc` and `pu`, the probability that a pair is
discordant is `d = pc * (1 - pu) + (1 - pc) * pu`. Conditional on discordance,
case-only exposure has probability `pc * (1 - pu) / d`. Power averages the exact
binomial rejection probability against null probability `0.5` over the binomial
number of discordant pairs. With no discordant pairs the test does not reject.
The number of pairs must be an integer of at least two.

Both functions accept `alpha=0.05, sides=1`. The native screens are one-sided;
`sides=2` is an explicit Python extension that halves alpha and uses the
alternative's direction. It omits the opposite rejection tail, following the
package's STPLAN dominant-tail convention.

The matched source calls its binomial power routine with null and alternative
probabilities reversed, contrary to the user manual's null probability of one
half. It also uses a relative-contribution stopping rule and propagates
empty-region sentinels. Python evaluates the intended null and the full finite
mixture. The direction error is substantial for protective exposures: with
`f=0.3`, disease risks `0.05` and `0.1`, 60 pairs, and alpha `0.05`, the source
returns approximately `0.456376`; the intended fixed-upper-tail test has power
approximately `0.000184743`.

## Two independent Poisson counts

```python
from mdanderson_stats import stplan_poisson_two_sample_power

power = stplan_poisson_two_sample_power(1, 2, 10, 10)
# Approximately 0.5130055.
```

The inputs are `rate1, rate2, exposure1, exposure2`. Rates are nonnegative,
exposures positive, and rate units must correspond to the exposure units.
Conditional on the total number of events, the first-arm count is binomial.
Its null probability is `exposure1 / (exposure1 + exposure2)`, and its alternative
probability is the first arm's share of expected events. Power averages the exact
binomial test over the Poisson total-event distribution. Zero events give zero
rejection probability. Equal rates use the lower-tail convention of the other
STPLAN exact-count APIs.

Keyword arguments are `alpha=0.05`, `sides=1`, and `tail_tolerance=1e-10`.
The infinite mixture is summed without renormalization. The omitted Poisson mass
bounds the truncation error by `tail_tolerance`; floating-point evaluation adds
ordinary rounding error. There is no switch to evaluating power at a single
expected event count.

The original uses a 99% retained-mass generator, renormalizes the resulting sum,
can duplicate an integer Poisson mode, and can propagate `-1` empty-region
sentinels. Above 200 expected events it substitutes a single conditional test at
the truncated mean. Python instead evaluates the conditional-test mixture with
an explicit error bound. For rates `1` and `2` with exposure `10` in both groups,
the native calculation gives `0.5179029`, compared with `0.5130055` from the
independent finite sum.

## Resources and validation

Inputs broadcast within the shared STPLAN limit of 200,000 values/cases. Mixture
work is additionally limited to 200,000 support terms across the entire batch,
and evaluated in bounded chunks. Requests exceeding the work limit raise a clear
error before the summation. A batch can be split into smaller calls; an individual
case that exceeds the limit is unsupported.

`tools/reference_stplan_mixtures.R` uses independent base-R finite probability
sums for 13 cases, including protective exposure, equal risks/rates, and a zero
Poisson rate. The R Poisson sums omit less than `1e-14` probability. Four unmatched
cases also agree with compiled original Fortran output. Six additional native
outputs document the matched/Poisson differences described above; they are
comparison evidence, not correctness targets. The original source is not bundled.
