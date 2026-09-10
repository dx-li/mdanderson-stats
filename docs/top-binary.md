# TOP delayed binary-response monitoring

`TOPBinaryDesign` implements the binary efficacy portion of
[TOP](https://biostatistics.mdanderson.org/shinyapps/TOP2/), catalog entry 134.
The public app reports version 1.0.6.0, updated 01/05/2026. The primary reference
is Lin, Coleman and Yuan, *JNCI* 112:38–45 (2020),
[doi:10.1093/jnci/djz049](https://doi.org/10.1093/jnci/djz049), especially Tables 2–4.
[Provenance](top-binary-sources.json) records the public documents used.

## Posterior and decisions

At a scheduled look, let n be enrolled patients, r observed responses, m pending
patients, and w the sum of their conditional time-to-response CDF weights.
The effective sample size is `n-m+w`. Known responses and known nonresponses
receive full information credit. Under uniform conditional response timing,
each pending patient's weight is follow-up divided by assessment-window length.

With a Beta(a,b) prior, TOP approximates the posterior as
`Beta(a+r, b+effective_sample_size-r)`. This is a fractional-information
approximation, not the exact pending-outcome likelihood. The default prior has
one effective observation centered on the null response rate: `(p0, 1-p0)`.
An explicit positive prior pair can be supplied.

The futility rule is posterior `P(p>p0) < C*(n/N)**gamma`, evaluated using the
direct upper tail. C and gamma are supplied tuning parameters, not automatically
calibrated by this constructor. The cutoff depends on **enrolled n**, not effective
sample size. Equality continues. Returned actions are `continue`, `stop_futility`,
`suspend`, or `success` at a completed final analysis.

Interim suspension is applied when responses do not already meet the complete-data
go threshold and pending counts reach the suspension threshold. A final analysis
always waits for all pending outcomes, including when enough responses already
meet its go threshold. The default `suspension="table"` uses
`pending >= ceil(n*n/N)` at interim looks. `suspension="strict"` follows the
article's prose instead: `pending > n*n/N`. These differ when `n*n/N` is an integer.

## Interfaces

`evaluate(patients, responses, pending, pending_weight)` broadcasts arrays of
summary counts and summed weights. Patient counts must be scheduled looks, and
response plus pending counts cannot exceed enrollment. Custom conditional timing
weights can be supplied explicitly; their appropriateness is a modeling choice.

`evaluate_followup(nonpending, responses, pending_followup, window)` calculates
uniform weights using the package's existing time-to-event ESS kernel. The last
axis contains pending patients. Follow-up and the window must share units;
normalization occurs before summation. Known outcomes include early responses
as well as completed nonresponses.

`boundaries()` returns scheduled enrollment, complete-data response minima,
suspension pending-count minima, and effective-size futility crossings. Rows
are looks and columns are response counts. Crossing values are unrounded:
strictly greater ESS stops for futility, unless suspension applies first.
Negative infinity means always futile over the feasible ESS range, positive
infinity means no crossing up to enrolled n, and NaN denotes an impossible count.
Use `evaluate` for decisions instead of rounding these crossings.

`complete_data_design()` returns the equivalent BOP2 design, whose existing
operating-characteristic methods apply when responses are completely observed.
Those operating characteristics do not validate performance with delayed responses.
Defaults use interim looks starting at 10 subjects and every 5 thereafter,
including the final sample size; supply explicit looks to match a protocol.
Up to 200 subjects are supported.

```python
import numpy as np
from mdanderson_stats import TOPBinaryDesign

design = TOPBinaryDesign(40, 0.2, 0.86, 0.95, looks=[10, 20, 30, 40])
table = design.boundaries()
np.testing.assert_array_equal(table.complete_go_min, [2, 4, 8, 12])
result = design.evaluate_followup(11, 3, [85, 78, 66, 48, 32, 28, 10, 8, 5], 120)
assert result.effective_sample_size == 14
assert result.decision == "continue"
np.testing.assert_allclose(table.futility_effective_size[1, 3], 15.091039493297)
```

Three focused tests reproduce the published table, verify the follow-up example
and time-unit invariance over 400 orders of magnitude, distinguish suspension
conventions, check final follow-up requirements, compare with complete-data BOP2,
and verify posterior crossings from both sides.

## Calendar replay and simulation

`run_top_binary_trial(design, interarrival, response_delays, window)` takes one
planned gap and potential response delay per patient. A finite delay in
`[0,window]` denotes a response; infinity denotes a nonresponse ascertained at
window completion. Early responses are immediately available. Decisions use only
responses and follow-up observed at the current calendar time.

Analysis occurs immediately after enrolling the patient who reaches a scheduled
look. If that look is suspended, the clock advances to the next response or window
completion and re-evaluates the same look. After continuation, the next arrival
gap begins; accrual does not build a queue while suspended. The first gap is
measured from trial time zero. These scheduling conventions are explicit Python
choices, not a claim of native event-scheduling parity.

Replay retains enrollment times, response times actually observed by termination,
pending indicators, the terminal decision, and a chronological decision history.
Unobserved response times are infinity even when a supplied potential response
would occur later. Duration ends at the terminal decision; follow-up of pending
patients after a futility stop is outside that duration. Interim accrual pauses
and waiting for final outcomes are reported separately.

`simulate_top_binary` batches trials with NumPy. It supports fixed or exponential
arrival gaps, uniform conditional response timing, and the package's calibrated
Weibull/log-logistic alternatives through `response_distribution` and
`late_probability`. The latter is the probability a response falls in the second
half of the window, conditional on responding by the window. Uniform generation
does not accept that argument. Analysis always uses uniform TOP weights, allowing
sensitivity checks under other true timing distributions. These simulations do
not optimize C/gamma or certify type I error control.

The result retains per-trial enrollment, observed responses and pending counts,
terminal actions, durations, accrual-pause times and final-wait times, as well as
success probability and its Monte Carlo standard error. It supports at most
100,000 trials and 2 million trial-patient cells. Positive time increments that
cannot be represented at the current clock magnitude raise an error instead of
silently changing event order.

```python
from mdanderson_stats import run_top_binary_trial, simulate_top_binary

small = TOPBinaryDesign(4, 0.2, 0.86, 0.95, looks=[2, 4])
trial = run_top_binary_trial(small, [0, 0, 0, 0], [0.2, np.inf, 0.1, np.inf], 1)
np.testing.assert_allclose(trial.enrollment_times, [0, 0, 0.2, 0.2])
np.testing.assert_allclose(trial.final_time, 1.2)
assert trial.decision == "success"

simulation = simulate_top_binary(design, 0.4, window=4, accrual_rate=2, trials=10000, rng=134)
assert simulation.success_mcse > 0
```

That simulation took approximately 0.17 seconds in the development environment,
with estimated success probability .8584 (MCSE .00349). This is an illustrative
benchmark for these explicit scheduling and timing choices, not a reproduced
native operating-characteristic result.

Three additional checks cover hand-calculated pause/resumption/final waiting,
time-unit scaling, absence of look-ahead, retained pending status after stopping,
calendar representability, and a final-only design against exact binomial power.

## Calibration with independent validation

`optimize_top_binary` searches an explicit Cartesian grid of `cutoff_scales` (C)
and `gammas`. It chooses the highest estimated alternative success probability
among candidates whose estimated null success probability is at most
`type1_error`. Ties prefer smaller mean null enrollment, then input order.
All candidates share potential outcomes and arrival gaps, reducing Monte Carlo
noise in comparisons. Candidates with identical complete-data boundaries are
retained because their fractional-information decisions can differ.

The selected design is evaluated on independent validation simulations; validation
never changes the selection. Both stages report null/alternative probabilities,
Monte Carlo standard errors, mean enrollment and mean duration. Grid rows follow
C then gamma input order, with null and alternative in the two columns. Returned
seeds reproduce each stage with `simulate_top_binary`.

```python
from mdanderson_stats import optimize_top_binary

calibrated = optimize_top_binary(
    20,
    0.2,
    0.4,
    window=4,
    accrual_rate=2,
    cutoff_scales=[0.8, 0.9, 0.95],
    gammas=[0.5, 0.75, 1],
    looks=[5, 10, 15, 20],
    trials=10000,
    validation_trials=10000,
    rng=134,
)
chosen = calibrated.design
print(calibrated.parameter_pairs[calibrated.selected_index])
print(calibrated.validation_probability, calibrated.validation_mcse)
```

The constraint is an estimated point-null constraint, not a guarantee of type I
error control. Grid selection introduces Monte Carlo uncertainty, and independent
validation can exceed the target. Assess that uncertainty and timing assumptions
before adopting a design; do not repeatedly redraw validation until it passes.
This is an explicit Python search, not reproduction of the native tuning grid.
An infeasible grid raises `TOPInfeasibleError`. Each stage supports 100–100,000
trials with at most 2 million trial-patient cells; the grid has at most 500 pairs.
Focused checks compare final-only designs with exact binomial power and verify
reproducibility of delayed-outcome calibration and independent validation.

## Source differences and remaining coverage

The published Table 4 labels `(gamma,C)=(.86,.95)`, but its numerical crossings
are reproduced by **C=.86, gamma=.95**. The example above uses the verified order.
At n=20 of N=40, the table suspends with 10 pending while the prose's strict
inequality requires 11. Both conventions are exposed. The earlier arXiv table
has different thresholds; this implementation targets the published table.

**Catalog status is partial.** Co-primary efficacy
and efficacy/toxicity models,
the app's nonuniform timing elicitation, native reports and app version parity
remain pending. Original PDFs and application files are not redistributed.
