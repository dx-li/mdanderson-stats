# BOP2 single-arm survival monitoring

BOP2 catalog **112** now includes specified-parameter survival monitoring,
follow-up-time boundaries, calendar replay, and Monte Carlo operating
characteristics, plus Monte Carlo grid calibration with independent validation.
Expected-enrollment and minimax sample-size searches are also available.
Two-arm/joint survival models and integrated reports remain pending.

The model is the exponential/inverse-gamma model described by
[Zhou et al. (2020), DOI 10.1002/pst.2030](https://pubmed.ncbi.nlm.nih.gov/32524679/).
The [current app prior guide](https://biostatistics.mdanderson.org/shinyapps/BOP2/TTEhelp.pdf)
and a [live app table capture](bop2-survival-app.json) establish its prior inputs
and the boundary's dependence on enrolled patients, events and total observation
time. Sources are recorded in [provenance](bop2-sources.json). This implementation
does not claim native optimizer or simulated-OC parity.

## Model and prior

For exponential survival with median `m`, the event rate is `log(2)/m`.
An event contributes its event time; a censored subject contributes observed
follow-up through the analysis time. With `d` events and total observation time
`T`, a prior `m ~ IG(a,b)` gives the posterior

$$
m\mid D \sim IG(a+d,b+\log(2)T).
$$

Here the inverse-gamma density is proportional to
`m**(-a-1) * exp(-b/m)`. Its scale `b` has time units.
The posterior probability of improving upon null median `m0` is the regularized
lower incomplete gamma function evaluated at `(b+log(2)*T)/m0`, with shape `a+d`.
This lower tail is computed directly.

The default prior follows the app guide's **elicitation equations**:
`a=1+prior_effective_events`, `b=prior_effective_events*m0`.
Thus the prior mean of the median survival parameter is `m0`; the default
effective event count is `0.05`. The guide also prints `IG(1.05,0.072*m0)` as a
default, which conflicts with those equations when interpreted as a prior on the
median. Approximately `0.072*m0` is the scale for the **mean-survival** parameter;
multiplication by `log(2)` converts that scale to approximately `0.05*m0` for the
median. This implementation uses the exact elicitation equations, not the rounded
and inconsistently labeled default sentence. An explicit `prior=[shape,scale]`
always refers to the **median** parameter and supersedes the ESS input.

The returned posterior scale is reported as `posterior_scale_ratio`, relative to
`null_median`. Keeping this dimensionless avoids multiplying large or small time
units unnecessarily. The same decisions should result after consistent changes
of time units.

## Monitoring

```python
from mdanderson_stats import bop2_survival_design

design = bop2_survival_design(
    30,
    null_median=6,
    cutoff_scale=0.85,
    gamma=0.85,
    looks=[15, 30],
)
state = design.monitor(events=8, total_observation_time=90, sample_size=15)
print(state.success_probability, state.decision)
print(design.total_time_boundary(events=[0, 5, 10, 15], sample_size=15))
```

These cutoff parameters are illustrative and **not automatically calibrated**.
At a scheduled look with `n` enrolled subjects, require posterior improvement
probability greater than `cutoff_scale*(n/N)**gamma`; otherwise stop for futility.
`equality_continues=False` follows the app's displayed `total time <= boundary`
stopping convention. Set it to `True` for strict-inequality futility instead.
At `N`, return `final_positive` or `final_negative`. There is no early efficacy
stopping; off-schedule evaluations return `continue`.

Both the event count and enrolled count are needed: censored patients contribute
follow-up without increasing the number of events. Inputs broadcast over leading
axes. `monitor_records(followup,event_observed)` aggregates observed patient data
on a final patient axis; event indicators must be zero or one. Do not pass latent
event durations for censored patients to this aggregation method.

`total_time_boundary` algebraically inverts the posterior rule for a specified
event/enrollment count. A negative threshold means no nonnegative follow-up time
can satisfy futility. The reported boundary is rounded floating-point output;
use `monitor` for decisions near it. Unrepresentable or underflowed numerical
quantities raise errors where they would invalidate a decision or reported
boundary. Monitoring supports up to 200 subjects, positive null median, scales
strictly in `(0,1)`, and `gamma` in `[0,1]`.

## Calendar replay and simulation

```python
from mdanderson_stats import run_bop2_survival_trial, simulate_bop2_survival

# Use the design defined above. Times below are durations from enrollment.
trial = run_bop2_survival_trial(
    design,
    enrollment_times=[(i + 1) / 1.5 for i in range(30)],
    event_times=[10] * 30,
    final_followup=12,
)
print(trial.calendar_times)
print(trial.states[-1].decision)
oc = simulate_bop2_survival(
    design,
    true_median=10,
    accrual_rate=1.5,
    final_followup=12,
    n_trials=10000,
    arrival="fixed",
    rng=960,
)
print(oc.success_probability, oc.success_mcse, oc.expected_sample_size)
```

Replay assesses each interim at the enrollment of that look's last patient.
The final analysis occurs at the last enrollment plus `final_followup`. All
subjects are administratively censored at each analysis; only patients enrolled
through that look are included. Replay stops immediately on the first negative
interim decision. Nonnegative latent event durations and `+inf` for no event are
accepted. Enrollment times must be nonnegative and ordered. Tied enrollment times
follow the supplied order of patients.

Simulation generates exponential event times. `arrival="fixed"` uses equally
spaced enrollment, while `"poisson"` uses exponential interarrival times. Both
include one interarrival interval before the first enrollment. Simulation uses
batches of up to 5,000 trials, supports up to 100,000 trials per call, and returns
per-trial sample sizes, event counts, calendar stopping times, and successes,
plus the empirical success probability, its Monte Carlo standard error, and mean
sample size. Run separate calls for different true medians. These are Monte Carlo
estimates, not exact OC or error-control guarantees.

This workflow assumes a constant event hazard and administrative censoring only.
It does not model dropout, competing risks, delayed data entry, or joint survival
and categorical outcomes. No calendar-time information after a stopping decision
is used to revise that decision.

Validation compares posterior probabilities with a 70-digit Erlang identity,
checks boundaries and both equality conventions, changes time units by factors
of `1e-200` and `1e200`, verifies censoring and absence of future-event leakage,
compares vectorized simulation with individual trial replay, and checks a
single-look simulation against its analytic success probability.


## Monte Carlo parameter calibration

`optimize_bop2_survival` searches cutoff scales and exponents for a fixed maximum
sample size under the exponential model above. This implements the paper's
power-maximization principle with an explicit Python finite grid and simulation
algorithm. Native app grid, random-number stream and tie-breaking parity are not
claimed. The default grids are scales `0.50,0.51,...,0.99` and exponents
`0,0.05,...,1`, giving 1,050 candidates.

```python
from mdanderson_stats import optimize_bop2_survival

fit = optimize_bop2_survival(
    30,
    null_median=6,
    alternative_median=10,
    accrual_rate=1.5,
    final_followup=12,
    looks=[15, 30],
    type1_error=0.1,
    n_trials=10000,
    n_validation=10000,
    rng=960,
)
print(fit.cutoff_scale, fit.gamma)
print(fit.calibration_oc.success_probability)  # Estimated null error, alternative power.
print(fit.validation_oc.success_probability)  # Independent post-selection estimates.
print(fit.validation_oc.success_mcse)
```

Within each of the null and alternative scenarios, every candidate uses the same
simulated enrollment and event paths. Posterior probabilities are computed once
at each look, then compared across batches of cutoff scales. A trial continues
only while it passes every preceding look. Expected enrollment is calculated
from the number still continuing at each interim; integer enrollment totals are
accumulated before division. This avoids repeating gamma calculations and
simulating fresh patients for each candidate. Scenario samples are independent.

The default `error_control="strict"` requires **estimated** null error no greater
than `type1_error`, and maximizes estimated alternative power. Power ties favor
smaller estimated null enrollment, then the input scale/exponent order.
`error_control="closest"` first minimizes absolute distance between estimated
null error and the target, then applies the same ranking; it can exceed the
nominal error. An optional `minimum_power` constrains the estimated power.
`objective="expected_sample_size"` instead minimizes estimated null enrollment,
with power as tie-breaker, and requires both `minimum_power` and strict mode.
No empirically feasible candidate raises `BOP2InfeasibleError`.

Selection always uses the null-centered default weak prior. An optional
`analysis_prior=[shape,median_scale]` never affects selection. The chosen
`calibration_design` is evaluated on fresh null and alternative trials, yielding
`validation_oc`. If an analysis prior is supplied, `analysis_design` is evaluated
on additional fresh trials, yielding `analysis_oc`; otherwise `analysis_oc` is
`validation_oc`. No re-selection occurs after these evaluations. The arrival law
and follow-up convention are the same as in standalone simulation.

All three OC summaries contain two-element arrays ordered **null, alternative**:
`success_probability`, `success_mcse`, and `expected_sample_size`, plus the number
of trials per scenario. The plug-in binomial MCSE is descriptive; calibration
MCSE does not account for selecting the best grid candidate. Independent
validation reduces that selection bias but **does not guarantee true type I
error control or target power**. Zero estimated MCSE after zero or all successes
does not establish certainty. An informative analysis prior may change error
substantially. Inspect independent validation and use adequate simulation sizes
before interpreting a selected design's performance.

Custom grids allow at most 10,000 pairs. Calibration and validation allow up to
100,000 trials each per scenario. Calibration also limits each scenario to
10 million trial/look combinations to bound memory. A supplied integer seed is
reproducible; a supplied NumPy generator advances through calibration, validation,
and optional analysis-prior validation in that order.

Focused calibration validation enumerates a small parameter grid using complete
standalone trial simulations, checks both objectives under fixed and Poisson
arrivals, reproduces the independent holdout, crosses the 5,000-trial batch
boundary, verifies closest-error ranking, and confirms that an informative
analysis prior leaves selection unchanged.


## Survival sample-size search

`optimize_bop2_survival_sample_size` searches a declared, strictly increasing grid
of maximum sample sizes in `1..200`. It applies the same expected-enrollment and
minimax criteria as the [categorical searches](bop2-sample-size.md), using Monte
Carlo estimates rather than exact categorical operating characteristics.

```python
from mdanderson_stats import optimize_bop2_survival_sample_size

search = optimize_bop2_survival_sample_size(
    [20, 25, 30, 35, 40],
    null_median=6,
    alternative_median=10,
    minimum_power=0.8,
    accrual_rate=1.5,
    final_followup=12,
    type1_error=0.1,
    objective="expected_sample_size",
    n_trials=10000,
    n_validation=10000,
    rng=112,
)
best = search.best
print(search.feasible_sample_sizes)
print(best.calibration_design.max_subjects, best.cutoff_scale, best.gamma)
print(best.calibration_oc.expected_sample_size)
print(best.validation_oc.success_probability, best.validation_oc.success_mcse)
```

Every size is searched, including sizes after an infeasible one. Within each size,
the selected candidate minimizes estimated null enrollment subject to estimated
null error at or below `type1_error` and estimated power at or above
`minimum_power`; power breaks enrollment ties. Across sizes,
`objective="expected_sample_size"` selects minimum estimated null enrollment,
then smaller maximum size, then higher estimated power. `objective="minimax"`
selects the smallest feasible maximum size, then smaller estimated null
enrollment and higher power. The result retains one empirically feasible fit per
size in `feasible_designs`, and the complete requested grid in
`searched_sample_sizes`. An entirely infeasible grid raises `BOP2InfeasibleError`;
invalid input and numerical failures propagate separately.

`interim_looks` supplies a common schedule of positive enrolled-patient counts.
For each maximum size `N`, keep supplied counts below `N` and append `N` as the
final analysis. An empty schedule requests a single final analysis. If omitted,
the standard `min_subjects=10, cohort_size=5` schedule is used for each size.
Final follow-up, accrual rate and arrival law have the same meaning as in the
fixed-size optimizer. A fixed follow-up duration after the last enrollment means
calendar duration can differ across maximum sizes.

The parent random generator draws one integer seed for **every searched size**,
including infeasible sizes. These are returned in `simulation_seeds`, aligned
with `searched_sample_sizes`; pass a recorded seed as `rng` to the fixed-size
optimizer with the same settings to reproduce that size's fit. Each size has its
own random stream. Changing `n_validation` or supplying an informative
`analysis_prior` leaves all size-specific calibration samples unchanged.
Changing the ordered size grid changes how seeds are assigned.

Only calibration estimates determine feasibility and ranking. Each feasible
size retains independent validation, and optional informative-prior evaluation,
from the fixed-size optimizer. The final chosen size's validation data were not
used in size selection. **Validation estimates may miss either requested error
or power target**; the search does not discard such results or silently select a
different size. Searching more candidates increases selection uncertainty, so
interpret calibration and independent validation separately. This is a finite,
Monte Carlo design search, not a claim of exact error control or global optimality.

Validation enumerates complete trial simulations over both sample size and
parameter grids under fixed and Poisson enrollment, checks both size objectives,
and verifies that analysis-prior and validation-size changes preserve selection.
