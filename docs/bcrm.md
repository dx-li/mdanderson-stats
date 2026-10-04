# bCRM: single-outcome continual reassessment

Catalog entry [15](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/15)
is **partially implemented**. This module supplies the single-outcome logistic
CRM supported by MD Anderson's bCRM program, with deterministic Bayesian
posterior summaries. It is distinct from the unrelated CRAN package named
`bcrm`. The original Windows application also supports a bivariate model and
additional trial features; those are not implied by the scalar implementation.
See [source records](bcrm-sources.json).

## Curve and observations

For a strictly increasing probability skeleton `s`, fixed intercept `alpha`,
and fixed lower and upper asymptotes `L` and `U`, the model is

```text
x[j] = log(s[j] - L) - log(U - s[j]) - alpha
p[j, beta] = L + (U-L) * expit(alpha + beta*x[j])
```

The nonnegative slope is shared across dose levels. At `beta=1` the curve
reproduces the skeleton. Physical dose strengths are labels; do not substitute
them for the transformed `x` values. The official guide commonly uses
`alpha=3` or `alpha=-3` and specifies a uniform slope prior on `[0, 3]`.
At `x=0` the response does not depend on the slope.

`BCRMCurve` constructs and stores this mapping. `bcrm_probabilities(x, beta)`
also accepts a transformed dose vector directly. A slope array of shape `(...)`
produces probabilities of shape `(..., number_of_doses)`; a scalar slope returns
a dose vector. `bcrm_log_probabilities` appends a final cell axis in
`(no event, event)` order and preserves finite log tails even when ordinary
probabilities round to zero or one. No probability clipping is used.

`bcrm_log_likelihood` takes grouped `events` and `subjects`, with one count per
dose row. It omits the parameter-independent binomial coefficients. All
observations must be complete; pending outcomes are not imputed.

## Posterior summaries

```python
from mdanderson_stats import BCRMCurve, fit_bcrm

# Skeleton from the official Goodman example; counts are illustrative.
curve = BCRMCurve([0.05, 0.10, 0.20, 0.35, 0.50, 0.70], alpha=3)
fit = fit_bcrm(curve, events=[0, 0, 1, 2, 0, 0], subjects=[2, 4, 6, 4, 0, 0])
mean_event_probability = fit.dose_mean
probability_interval = fit.dose_interval  # one [lower, upper] row per dose
```

The fit returns the slope mean, standard deviation, equal-tailed interval and
log evidence, alongside two distinct estimates of dose probability:

- `dose_mean` is the posterior expectation of `p(beta)`.
- `plugin_dose_probability` evaluates `p` at the posterior mean slope.

These generally differ. The available guide does not resolve which convention
the native allocator uses, so both are exposed explicitly. `log_evidence`
integrates the likelihood without binomial coefficients against the normalized
uniform prior; it is not a binomial-data marginal probability including those
coefficients. Stored arrays are read-only.

Python uses adaptive one-dimensional quadrature in place of native MCMC.
Likelihood scaling protects the normalizer against underflow, and centered
moments avoid cancellation in a concentrated posterior variance. Quantiles
are transformed and ordered separately for each dose, including doses where
probability decreases as the slope increases. Integration failures raise an
error rather than returning an unchecked fit.

Curves accept 1-100 dose levels and `abs(alpha) <= 50`. Fitting permits up to
10,000 subjects and a uniform prior interval contained in `[0, 3]`; narrowing
that interval is an explicit Python option. Predictions allow at most 200,000
slope values and 200,000 slope-dose pairs. Fitting also caps likelihood
evaluations at 200,000. These limits bound memory and work; this implementation
does not start parallel workers. `integration_error` is the sum of the
unnormalized quadrature error estimates for the normalizer and first and
centered second moments, not a confidence bound on all returned quantities.

## Single-outcome dose decisions

`bcrm_decision` takes an explicit vector of estimated probabilities and the
cumulative complete-outcome subject counts at each dose. Supply either posterior
mean probabilities or probabilities at the mean slope deliberately; the helper
does not silently select a convention or refit the posterior.

```python
from mdanderson_stats import bcrm_decision

decision = bcrm_decision(
    fit.dose_mean,
    [2, 4, 6, 4, 0, 0],
    target=0.2,
    max_subjects=24,
    min_subjects=18,
    cohort_size=2,
    stop_at_target=6,
    selection="nearest",
    max_increment=1,
)
# next_index is a zero-based dose index, or None if the trial should stop.
```

Selection may be `below` (highest probability at or below the target), `above`
(lowest at or above), or `nearest`. The default is `below`, as for native
toxicity targeting. Exact equality qualifies. Nearest ties within floating-point
roundoff choose the lower dose index; these equality and tie conventions are
explicit Python behavior.
If the requested side has no qualifying dose, the helper uses the lowest dose
for `below` or the highest for `above`, with `target_attainable=False`. The
lowest-dose fallback is documented for native stage one; the highest-dose
fallback is an explicit Python convention.

Before any subjects are treated, `start_index` controls allocation. Later,
escalation is capped at `max_increment` levels above the **highest dose already
tried**, not above the last dose. `target_index` reports the unrestricted target
or fallback and `next_index` reports the capped allocation. Reaching the sample
cap stops the trial. Once `min_subjects` is reached, sufficient cumulative
enrollment at the current target also stops it. This count is not required to
come from consecutive cohorts. Subject counts and sample-size limits must be
whole cohorts. The helper handles only single-outcome/stage-one decisions.

For optional extra allocation to an extreme dose, the simulation guide gives
`bcrm_extreme_allocation_probability(target_fraction, allocated_fraction,
correction=2)`. It evaluates
`p_T ** (1 + correction * (p_o - p_T))`, where `p_T` is the requested allocation
fraction and `p_o` is the observed allocation fraction at that dose. The result
is clipped to `[0.1, 0.5]`; setting correction to zero returns the target after
clipping. The target is required to lie strictly inside `(0, 1)` because the
source formula is undefined for some zero-target cases; observed allocation
fractions may lie in `[0, 1]`. Correction is finite and nonnegative. This
returns a randomization probability only: it does not select a dose, adjust an
efficacy estimate, or reproduce the native trial controller.

## Coverage and evidence

The original 1.1.3 archive was retrieved and inspected without running its
installer. It contains a compiled model engine, Java GUI classes, configuration
and guides. No model source was recovered. Output fields for `Psi` confirm an
association parameter in the two-outcome implementation. Furthermore, the
guide says two-outcome mode averages the toxicity and efficacy transformed dose
vectors. Two separately fitted scalar models therefore do not reproduce the
documented bivariate program.

The joint toxicity/efficacy likelihood and association prior, two-stage trial
conduct, futility monitoring, post-trial four-parameter logistic fit, native
file formats and full simulation workflow remain pending. No native random-seed
or numerical output parity is claimed.

Independent [base-R reference calculations](../tools/reference_bcrm.R) integrate
the same scalar posterior for the official Goodman skeleton, bounded
asymptotes, and concentrated and boundary posteriors. Their observation counts
are synthetic audit cases, not native simulation results.
