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

## Poisson calculations

The Poisson APIs use equal-tail Garwood intervals on the **event rate**.
For count k and exposure t, the interval has lower gamma shape k (zero bound
at k=0), upper gamma shape k+1, and both bounds are divided by t. Exposure
can be time or another unit of Poisson observation. The true rate may be zero.

| API | Result |
|---|---|
| `confint_poisson_probability` | Probability the rate CI has at most the requested total length; inputs broadcast |
| `confint_poisson_length` | Smallest attainable length reaching the requested cumulative probability |
| `confint_poisson_rate_limit` | Largest true rate attaining assurance, or `None` if no count qualifies |
| `confint_poisson_exposure` | Earliest exposure attaining assurance |

The probability calculation searches integer counts and evaluates the Poisson
CDF at the last qualifying count. Unlike the native `arg<=1` shortcut, it
includes cases where only zero or one event qualifies. The length calculation
uses a discrete Poisson quantile; the rate limit uses inverse gamma-tail
probability directly, avoiding approximate outer root searches.

**Exposure assurance is not monotone.** If w(k) is the unscaled interval width,
count k becomes acceptable at exposure `w(k)/max_length`. Between these
thresholds, the qualifying count stays fixed while the Poisson mean grows,
so assurance decreases. The first success must occur at a threshold. Python
checks them in increasing-count batches; it does not assume that every later
exposure will also attain assurance. Returned floating-point exposure/length
thresholds are rounded upward by one representable step when needed to include
the defining outcome.

Probability calculations and length quantiles support qualifying event counts
through 100 million. Exposure planning scans from zero through `max_events`
(default one million, inclusive) and raises if the target is not reached.
Exposure, maximum length and rate must be finite, with positive exposure/length
and nonnegative rate. Confidence/assurance ranges match the other CONFINT APIs.
Probability evaluation supports two million broadcast designs; other Poisson
functions handle one design. Unrepresentable output lengths/rates/exposures
raise errors. Changing time units by factors 1e±200 preserves probability.

For rate 20, exposure 210, maximum length 1 and confidence .9, the qualifying
count is 4035 and the probability is .0053469235. This agrees with the manual's
commentary, while its adjacent printed block gives a different number. Independent
R gamma quantiles and Poisson CDFs confirm the probability, length quantile,
and rate-limit results. With assurance .9, the first qualifying exposure is
222.5920387855, at count 4537, with probability .9001183457; all earlier
thresholds have probability at most .8987881519.

```python
from mdanderson_stats import (
    confint_poisson_probability,
    confint_poisson_length,
    confint_poisson_rate_limit,
    confint_poisson_exposure,
)

p = confint_poisson_probability(210, 1, 20, confidence=0.9)
assert abs(float(p) - 0.005346923511) < 1e-12
assert abs(confint_poisson_length(210, 20, confidence=0.9) - 1.0300012109) < 1e-9
assert abs(confint_poisson_rate_limit(210, 1, confidence=0.9) - 18.8323846149) < 1e-9
exposure = confint_poisson_exposure(1, 20, confidence=0.9, assurance=0.9)
assert abs(exposure - 222.5920387855) < 1e-8
assert confint_poisson_probability(exposure, 1, 20, confidence=0.9) >= 0.9
```

## Difference of two binomial proportions

The native `bin2_ci_mod.f90` plans the **plain, unadjusted Wald interval**.
Its total length is
`2*z*sqrt(phat1*(1-phat1)/n1 + phat2*(1-phat2)/n2)`, where z is the upper
normal equal-tail quantile. This differs from the adjusted CID2BP intervals:
there is no continuity correction, special boundary adjustment or clipping.
A high probability of a short interval does not establish its nominal coverage;
small samples can have zero estimated variance and misleadingly narrow CIs.

`confint_binomial_difference_probability(n1,n2,p1,p2,max_length)` evaluates
width assurance for independent samples. Python folds complementary counts
in the smaller group, then uses binomial CDF/SF tails for acceptable counts
in the other group. This needs O(min(n1,n2)) time and memory, rather than
allocating the full joint-outcome grid. Stable quadratic roots and integer
cutoff checks locate the variance boundary. Every folded count contributes;
the source's relative-term early stopping is omitted.

The source also uses its first-group midpoint flag to decide whether to add
the second group's upper tail. Those are separate conditions: both second-group
tails must be included even at a first-group midpoint. For n1=2, n2=3, p1=.5,
p2=.4, length 1.5 and confidence .95, the full width probability is .64.
Independent joint-outcome enumeration verifies this and other boundary cases.

`confint_binomial_difference_event_limit` holds p1 fixed and returns the largest
p2 in [0,.5] attaining assurance. The symmetric region p2≥1−limit also qualifies.
It returns `None` when impossible and .5 when all probabilities qualify.
`confint_binomial_difference_sample_size` scans every equal per-group integer
n in the requested range, because assurance can decrease between adjacent n.
The native search range 10–1000 is the default; users may specify bounds within
1–10000. This is the first qualifying size **within that range**. At n1=n2=1
all estimated variances are zero, illustrating why a lower planning bound
matters for this approximation. Larger sizes need not all attain assurance.

All three APIs accept scalar designs. Fixed sample sizes may be 1–1,000,000;
probabilities are in [0,1] and max_length must be positive and finite.
Confidence/assurance ranges match the other CONFINT APIs. The implementation
matches independent R joint enumeration, including .3145255737 for
n1=12, n2=17, p1=.2, p2=.4 and length .6 at confidence .95.

```python
from mdanderson_stats import (
    confint_binomial_difference_probability,
    confint_binomial_difference_event_limit,
    confint_binomial_difference_sample_size,
)

assert abs(confint_binomial_difference_probability(2, 3, 0.5, 0.4, 1.5) - 0.64) < 1e-12
assert confint_binomial_difference_sample_size(0.3, 0.2, 0.4, assurance=0.9) == 74
p = confint_binomial_difference_event_limit(40, 60, 0.2, 0.3, assurance=0.8)
assert abs(p - 0.0679396627) < 1e-9
assert abs(confint_binomial_difference_probability(40, 60, 0.2, p, 0.3) - 0.8) < 1e-10
```

## Exponential-survival width assurance

`confint_survival_fixed_events(events,hazard,max_length,target=...)` implements
the source's gamma approximation conditional on the event count. For n>0,
let T be total time at risk with the planning distribution `Gamma(n, rate=h)`.
Write g_low and g_high for the equal-tail quantiles of `Gamma(n, rate=1)`.
The hazard interval is `[g_low/T,g_high/T]`, and the mean-survival interval is
`[T/g_high,T/g_low]`. Thus width assurance is a gamma survival probability for
`target="hazard"` and a gamma CDF for `target="mean"`. Zero events receive
zero assurance because the source treats their interval as undefined.

The manual calls the gamma model a **general-censoring approximation**. It is
not the exact conditional distribution of observed follow-up in every
calendar-based trial. The source's gamma parameter named `scale` is a rate;
Python uses the explicit name `hazard` to avoid that ambiguity.

`confint_survival_probability(hazard,accrual_rate,accrual_time,followup_time,max_length)`
averages this conditional assurance over a Poisson event count. Accrual is
Poisson, entry is uniform over the accrual period, and survival is exponential,
with no dropout or competing risks. If A is accrual duration and F additional
follow-up, the per-patient event probability is
`1-exp(-h*F)*(1-exp(-h*A))/(h*A)`. The event-count mean is `accrual_rate*A*p_event`.
This event-count distribution follows from Poisson thinning; the conditional
follow-up approximation remains a separate assumption.

Python reuses the package's stable uniform-accrual calculation. A log-domain
fallback preserves expected event counts when the per-patient probability
underflows but the expected count remains representable. Fixed-event gamma
evaluation also uses log arguments, avoiding overflow from changing time units.
Stable Poisson mass factors avoid cancellation near large count means.

The result `CONFINTSurvivalAssurance` reports:

- `probability`: the unnormalized sum over retained positive event counts;
- `expected_events` and `event_probability`;
- `omitted_probability`: an absolute bound on omitted positive-count contributions;
- `included_events`: inclusive count limits, or `None` for a zero mean.

The default truncation tolerance is 1e-12, configurable from 1e-14 to 1e-4.
The omitted-mass bound addresses truncation only; it does not bound floating-
point error or model approximation. In exact arithmetic the sum is a lower
bound and the full mixture lies at most `omitted_probability` above it.
The source instead stops near 99% mass or 1000 evaluations and renormalizes.
Python's results intentionally differ from that approximation.

For h=1, accrual rate 5, accrual duration 10 and no extra follow-up, expected
events are 45.0002269996. With confidence .95 and total length .5, independent
R summation gives .154951144202 for hazard-width assurance and .128574843760
for mean-width assurance. The manual reports approximately .153 and .127284.
Tests check R results within the omitted-mass bounds and validate conditional
width probabilities using independently generated exponential samples.

Fixed-event calculations broadcast up to two million designs, with integer
counts 0–200 million. Study calculations are scalar, with expected counts at
most 100 million and at most 200,000 retained count terms. Hazard, length and
accrual duration must be positive and finite; accrual rate and follow-up must
be nonnegative, with finite total duration. Confidence uses the same range as
other CONFINT APIs. Hazard lengths have inverse-time units; mean lengths have
time units. Changes of time units by 1e±200 are checked.

```python
from mdanderson_stats import confint_survival_fixed_events, confint_survival_probability

result = confint_survival_probability(1, 5, 10, 0, 0.5, target="hazard")
assert abs(result.probability - 0.154951144202) < 1e-11
assert result.omitted_probability <= 1e-12
mean = confint_survival_probability(1, 5, 10, 0, 0.5, target="mean")
assert abs(mean.probability - 0.128574843760) < 1e-11
fixed = confint_survival_fixed_events([0, 10, 45], 1, 0.5)
assert fixed.shape == (3,) and fixed[0] == 0
```

## Survival quantiles and design inversions

`confint_survival_solve` solves for one of `hazard`, `accrual_rate`,
`accrual_time`, `followup_time`, or `max_length`. Supply the other four and
leave the unknown as `None` (or omit it). Specify `assurance`, `confidence`,
and interval `target` just as for the forward calculation. Solving for
`max_length` gives the mixture width quantile; studies with zero events have
infinite width, so some high quantiles are unattainable at any finite length.

The result `CONFINTSurvivalSolution` contains the solved `parameter`, its
`value`, the requested `assurance`, and the forward `achieved` result with its
expected event count and omitted-mass bound. The default summation tolerance
is 1e-14. Root finding uses log coordinates to retain relative accuracy under
changes in physical units.

**Supply a bracket for the desired crossing.** `bounds=(low,high)` defaults to
(1e-4,1e4), matching the native non-hazard search range, in the solved parameter's
units. Both endpoints must be within the forward calculation's resource and
validity limits. Zero is permitted as a lower bound only for accrual rate and
follow-up. The endpoint assurances must straddle the target, or an endpoint
must attain it. Otherwise the function raises rather than returning an
unverified boundary value.

This API returns **one root within the bracket**, not the smallest design,
a unique global solution, or every admissible value. Some curves are
nonmonotone. For a hazard-width interval, two different hazards can produce
the same assurance. Bracket the lower or upper crossing separately. A broad
bracket enclosing both can have same-sign endpoints and be rejected despite
containing roots. Use `confint_survival_hazard_range` below to locate peaks and crossings automatically.

Independent R full-mixture inversions give, for h=1, accrual rate 5, accrual
10, follow-up 0 and confidence .95:

| Solved quantity | Hazard interval | Mean-survival interval |
|---|---:|---:|
| Median CI length | .588236492657 | .604554196094 |
| Accrual rate for length .5 and assurance .9 | 9.526211616013 | 9.689856286516 |
| Accrual duration for length .5 and assurance .8 | 16.467345537066 | 16.853286568888 |
| Extra follow-up for accrual duration 16, length .5 and assurance .8 | .629882372084 | 1.919274257303 |

For the hazard interval with length .5 and assurance .5, the lower and upper
hazard solutions are .002858333869 and .840906351881. Tests verify both
crossings, the other inversions, and time-unit changes by factors 1e±200.
These use the full mixture rather than the manual's 99%-mass approximation.

```python
from mdanderson_stats import confint_survival_solve

median = confint_survival_solve(
    hazard=1,
    accrual_rate=5,
    accrual_time=10,
    followup_time=0,
    assurance=0.5,
    bounds=(0.1, 2),
)
assert median.parameter == "max_length"
assert abs(median.value - 0.588236492657) < 1e-9
followup = confint_survival_solve(
    hazard=1,
    accrual_rate=5,
    accrual_time=16,
    max_length=0.5,
    assurance=0.8,
    bounds=(0, 10),
)
assert abs(followup.value - 0.629882372084) < 1e-9
assert abs(followup.achieved.probability - 0.8) < 1e-10
```

## Automatic hazard peaks and admissible ranges

`confint_survival_hazard_range(accrual_rate,accrual_time,followup_time,max_length)`
finds a numerical peak and the hazard ranges attaining `assurance`. The default
`hazard_bounds=(1e-4,1000)` match the native outer range for a reference hazard
of 1. Bounds are in the user's inverse-time units and must be positive and
increasing, with a resolvable separation in log coordinates. All evaluations
must meet the forward calculation's resource limits.

A logarithmic grid (65 points by default, configurable from 17 to 1025) locates
candidate extrema. Both local maxima and minima are refined before each target
crossing is bracketed. Evaluations are cached within the search. The result
`CONFINTSurvivalHazardRange` contains:

- `peak_hazard` and its forward `peak` result, including the omitted-mass bound;
- `intervals`, an immutable tuple of admissible `(lower,upper)` pairs;
- the search `bounds`, plus `lower_clipped` and `upper_clipped` flags;
- `peak_at_boundary`, identifying whether the selected peak is at a search endpoint.

An empty interval tuple means no qualifying range was found. If a range extends
to an accepted search endpoint, the corresponding flag is true: that endpoint
is a search limit, not a solved threshold. Both hazard-width and mean-width
targets are supported. Flat probabilities near one can make the numerical peak
location nonunique or sensitive to rounding; the crossing thresholds remain
the useful design quantities in that case.

This grid/refinement procedure is a numerical search, **not a proof of a global
maximum or exhaustive detection of arbitrarily narrow components**. Increasing
`grid_points` or choosing a more focused search interval can check resolution.
The manual assumes a single peak for hazard-width assurance; Python can return
multiple detected admissible intervals.

The native `set_hazard_info` call passes accrual duration as follow-up during
peak search. Python consistently uses the supplied follow-up duration.
For accrual rate 5, duration 10, follow-up 0 and maximum length .5 at confidence
.95, independent R calculations give peak hazard .1564455377 with probability
.999978426873. At assurance .5, the hazard-width range is approximately
[.002858333869,.840906351881]. For mean-width assurance, the crossing is
1.197442895773 and the accepted range continues to the upper search limit.
Tests verify these values, restricted-bound clipping, insufficient designs,
grid refinement and changes of time units by 1e±200.

```python
from mdanderson_stats import confint_survival_hazard_range

result = confint_survival_hazard_range(5, 10, 0, 0.5, assurance=0.5)
assert abs(result.peak_hazard - 0.1564455377) < 1e-6
assert len(result.intervals) == 1
low, high = result.intervals[0]
assert abs(low - 0.002858333869) < 1e-9
assert abs(high - 0.840906351881) < 1e-9
assert not result.lower_clipped and not result.upper_clipped
```

**Catalog status is partial.** Native session/report workflows remain pending.
