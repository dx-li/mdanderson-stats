# Proportional Density: censored survival treatment effects

`proportional_density(time, event, treatment)` implements the estimation method
of Shen, Qin and Costantino (2007), JASA 102:1235–1244,
[DOI 10.1198/016214506000001446](https://pmc.ncbi.nlm.nih.gov/articles/PMC2721282/),
corresponding to MD Anderson catalog entry 78. Event indicators are 1 for an
observed failure and 0 for right censoring; treatment is 0 for control and 1 for
treatment. The archived age-at-entry column is unused, so it is omitted here.

```python
from mdanderson_stats import proportional_density

fit = proportional_density(
    time=[1, 3, 5, 7, 9, 12, 2, 4, 6, 8, 10, 12],
    event=[1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    treatment=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
)
print(fit.beta, fit.incidence)
print(fit.time, fit.disease_survival)
```

## Model and returned quantities

The model is `g_c(t) / f_c(t) = exp(alpha_star + beta*t)`, where the densities
condition on eventual disease. A positive beta tilts the treatment density
toward later failure times. It is not a proportional-hazards coefficient.
The implementation uses the archive's linear time transform; the paper also
discusses other transforms, which this API does not implement.

The fit uses failures from both arms, separate reverse Kaplan–Meier censoring
survivals H0 and H1, and the paper's equations (2.2)–(2.3). With m0/m1 the arm
failure counts, its logistic predictor is
`log(m1/m0) + alpha + beta*t + log(H1(t)) - log(H0(t))`.
Estimated conditional observed-failure masses are
`p_i = expit(-eta_i)/m0`, `q_i = expit(eta_i)/m1`.
Dividing these by H0/H1 and separately normalizing gives disease-conditional
masses. This also determines the intercept correction to alpha_star.

The immutable result supplies alpha, beta, alpha_star, incidence estimates and
Greenwood standard errors; unique failure times; two-column probability masses
conditional on observed failure and eventual disease; fitted disease survival;
arm-specific nonparametric disease survival; censoring survival; the archive's
likelihood-ratio and goodness-of-fit statistics; and convergence diagnostics.
Every two-column array is ordered **control, treatment**. Incidence is one minus
the arm's final Kaplan–Meier survival: it estimates the susceptible fraction,
**not** the cure fraction despite a misleading comment in the source. Disease
survival is conditional on eventual disease, not unconditional survival.

Independent censoring within each arm and sufficient follow-up to identify the
susceptible fraction are substantive assumptions. An observed KM plateau alone
does not establish that follow-up is sufficient. The software cannot establish
that assumption from these vectors. Incidence estimates and conditional curves
retain the original method's interpretation and limitations.

## Inference and coverage boundary

The default estimates separate censoring distributions and returns
`likelihood_pvalue=None`. The paper explicitly says the LR statistic's null
calibration is generally **not** ordinary chi-square with estimated unequal
censoring. The archive nevertheless computes that p-value unconditionally.

Use `equal_censoring=True` only when a shared censoring distribution is justified.
This pools reverse KM estimates, sets their log-ratio offset to zero, and returns
the asymptotic chi-square-one-degree-of-freedom LR p-value for alpha=beta=0.
Pooling is an explicit Python extension; the archive always estimates H0/H1
separately. This is not an exact finite-sample test. The incidence p-value follows
the archive's unpooled Greenwood Wald calculation, not the paper's pooled-null
variance expression. It is `None` when both standard errors are zero.

`proportional_density_pepe(time, estimated_survival, nonparametric_survival)`
returns the archive's squared-curve-area statistic, using unit weight and
**right-endpoint rectangles** from time zero to the final supplied failure.
This differs from integrating right-continuous step functions, and from the
paper's suggested common-follow-up truncation. Inputs may be unsorted but must
have unique, nonnegative times and valid monotone survival vectors. The fit's
`goodness_of_fit` applies this statistic to the control disease-survival curves.
It is not a calibrated goodness-of-fit p-value.

**Entry 78 remains partial.** Both supplied R routines have Python counterparts,
and the paper’s alternative failure-only goodness-of-fit bootstrap is available
as described below. Full-data disease-curve bootstrap calibration, unequal-
censoring treatment-effect calibration and bootstrap parameter uncertainty
remain unimplemented. No bootstrap code is present in the downloaded archive;
a comment instructs the user to bootstrap. Missing inference workflows remain
tracked rather than represented as completed.

## Source repairs and numerical behavior

The archive contains exactly `parameterest.R` and `modelchecking.R`. It calls
`condlik` and `lsort`, neither of which is included. The missing likelihood was
reconstructed from the paper; stable sorting supplies the missing ordering.
Source hashes and complete member inventory are in
[proportional-density-sources.json](proportional-density-sources.json).

Current R requires a formula for `survfit`; the old direct Surv-object call fails.
Also, `summary(...)$std` is ambiguous between std.err and std.chaz and returns
NULL in the installed version. The native validation harness therefore adapts
the call to `survival::survfit(formula ~ 1, timefix=FALSE)`, changes `$std` to
`$std.err`, supplies the missing helpers, and replaces only the final return
statement to expose the already-calculated intermediates. It does not alter
the archive's estimation formulas or default nlminb optimizer. These repairs
are required to run the archived routines; comparisons are not claimed against
an unchanged runnable distribution.

Python additionally:

- Fits centered/scaled time with Newton solves and backtracking; rejects
  complete/quasi separation, constant failure times and nonconvergence.
- Uses log-add-exp likelihoods, logistic complements and log-sum-exp mass
  normalization to avoid exponential overflow. Reverse cumulative sums preserve
  small upper-tail survival probabilities without subtracting from one.
- Aggregates tied failure masses before computing survival, so each time has
  one well-defined right-continuous value. The archive's per-record cumulative
  values at duplicate times are not a valid single-valued curve and can distort
  its rectangle statistic. With exact ties, failures and censors share the risk
  set, matching the existing Kaplan–Meier convention; near ties are not merged.
- Explicitly extends KM plateaus beyond an arm's last observation. The archive's
  unextended disease KM queries can return shorter, misaligned vectors.
- Rejects zero censoring survival at any pooled failure instead of replacing
  an unsupported tail with an arbitrary positive number.
- Supports a single-point standalone area statistic (the native indexing fails)
  and returns explicit missing calibration rather than a fabricated p-value.

## Validation

Three censored, overlapping-arm datasets were evaluated by the repaired native
R routines and independently with R's binomial-logit `glm` using the same offset.
The fixture stores the inputs, parameters, Greenwood errors, probability masses,
survival curves, censoring curves and goodness-of-fit statistics. Native nlminb
coefficient differences are below 3e-6; independent GLM coefficient differences
are below 5e-9. Native cumulative-curve differences are below 4e-7. Python's
scaled score residual is at most 1e-12.

Three focused tests also check mass normalization, immutable outputs, arm
exchange, time-unit changes by 1e-150 and 1e150, tied identical-arm analytic
results, input-order invariance, separation, unidentifiable slopes and unsupported
censoring tails. No CI jobs or dependencies were added.


## Failure-only goodness-of-fit bootstrap

`proportional_density_bootstrap(time, event, treatment, replicates=999, seed=None)`
implements the computationally cheaper alternative in section 3.1 of the paper.
It tests the observed-failure density ratio, using the same fixed censoring
log-ratio estimated from the original data. Each replicate draws the original
number of failures independently within each arm from the fitted observed-failure
masses, refits the offset likelihood, and compares the fitted control CDF with
its empirical failure-time CDF. It does not resample censoring observations.

```python
from mdanderson_stats import proportional_density_bootstrap

result = proportional_density_bootstrap(
    time=[1, 3, 5, 7, 9, 12, 2, 4, 6, 8, 10, 12],
    event=[1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    treatment=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    replicates=999,
    seed=7808,
)
print(result.statistic, result.pvalue)
print(result.failed_replicates, result.pvalue_lower, result.pvalue_upper)
```

This is the paper's **Delta_1n** statistic, not the archive's disease-curve Pepe
statistic. Unit weight and exact integration of squared right-continuous CDF
differences are used. The default integration endpoint is the minimum of the
two arms' maximum follow-up times; `tau` can select a smaller positive endpoint
that extends beyond at least one failure. Values at jumps have measure zero;
each post-jump difference applies until the next jump or tau. The constant tail
after the final failure is included when applicable.

The result stores the observed statistic, endpoint, all bootstrap statistics,
failed-replicate counts/reasons, an upper-tail Monte Carlo p-value and its
approximate simulation standard error. With B replicates and E exceedances,
the p-value is `(1+E)/(B+1)`, counting ties as exceedances. This finite-simulation
correction does not make the fitted-model bootstrap an exact finite-sample test.
The standard error uses `sqrt(p*(1-p)/B)` and measures simulation error only.
A local NumPy generator makes a supplied integer seed reproducible without
changing global random state.

Discrete resampling can produce separated or constant failure times, so a finite
fit need not exist for every replicate. Such replicates are retained as NaN,
with their exception reasons counted. When K fits fail, the single p-value and
its standard error are `None`; reported bounds are `(1+E)/(B+1)` and
`(1+E+K)/(B+1)`. These bound the contribution of unresolved replicates; they are
**not confidence intervals**. Failed replicates are neither discarded nor
redrawn. Original-data fitting failures still raise immediately. Replicates
are limited to 10,000 and total sampled failures to 50 million.

Five fixed resamples agree with independent R `glm` offset fits and `ecdf`-based
step integration within 2e-13 absolute error. The fixture includes the sampled
times, labels and offsets, rather than claiming native bootstrap outputs from
an archive that has no bootstrap implementation. Two additional focused tests
cover that reference, time-unit invariance, a hand-calculated step integral,
immutable output and honest calibration bounds when resamples are unidentified.
The treatment-effect LR test and full-data disease-curve bootstrap are separate
procedures and are not calibrated by this function.
