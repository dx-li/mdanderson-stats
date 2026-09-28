# CiBolus complete-outcome trial contract

The [core source audit](cibolus-audit.md) records the existing likelihood,
posterior, joint response-category/toxicity predictions and allocation rules.
The [author-hosted paper](https://odin.mdacc.tmc.edu/~pfthall/main/Biometrics_IAtPA_2011.pdf),
Section 3.3, chooses the first cohort's regimen explicitly, updates after cohort
data, and subsequently maximizes posterior mean utility among acceptable
regimens subject to concentration-only no-skip. Final selection removes the
concentration restriction. Equations 11–12 define the conditional-toxicity and
response screens; the existing decision component implements them.

The observed-data model supplies a joint distribution over bolus response,
detected response intervals and failure, crossed with binary toxicity.
Generating directly from these cells preserves endpoint dependence and uses
the interval upper endpoint for the delivered treatment and toxicity model.
A complete-outcome cohort simulation does not require inventing a continuous
response-time approximation or independent toxicity generator.

The intended scope assumes both outcomes are known at each cohort boundary.
The motivating application observes toxicity later than infusion/response;
calendar timing and pending-outcome conduct remain outside this component.
Truth parameters, priors, utility, grids and sampling precision are explicit.
No native executable or random-stream equivalence is claimed.

## Independent references

`tools/reference_cibolus_trial.R` uses base-R numerical integration of the
continuous response hazard and interval densities. Fixed priors isolate
allocation and observation conversion from posterior Monte Carlo variation.
It produces 36 joint-category rows, six patient records, 18 regimen/look rows
and two trial summaries. The generator completed in 0.18 seconds with warnings
treated as errors.

The first four-patient path uses uniforms `0, .2, .6, .95`, starts at regimen
indices `(0,0)`, and moves to `(1,1)` for the second cohort. Outcomes include
bolus response, two detected response intervals, and failure with toxicity.
The unrestricted final recommendation is the untried regimen `(2,1)`. The
second path keeps the same truth while fixing the prior baseline toxicity
parameter at three; it stops after the first cohort without a recommendation.

The Python implementation matches all 36 joint rows with maximum absolute
error `5.27e-16`, all six patient records, all 18 regimen/look rows and both
trial summaries. Maximum posterior utility error is `2.84e-14`. The safe path
fits exactly twice; the unsafe path fits once and keeps no recommendation.
The model generates the declared dependent outcome cells and interprets their
observation intervals correctly.

A separate four-patient replay allows log baseline toxicity to vary. Its two
cohort fits match sequential calls to the actual fitter exactly under the same
random stream: utility grids, likelihood evaluations and work counts agree.
The independent comparison takes 0.947 seconds after imports, peaks at
117.50 MiB process RSS and reports zero swaps.

Luna checkpoint `232601f` adds the simulator and three focused tests. All three
pass in 1.57 seconds; module lint, formatting, typing and diff checks pass.
No new CI or large simulation was introduced. These references do not validate
arbitrary prior convergence, calendar timing or operating characteristics over
large simulated populations.
## Aggregate operating characteristics

The serial aggregate interface reuses the validated single-trial implementation
and discards histories between replicates. Returned per-trial seeds reproduce
individual simulations. It reports selection/no-selection and stop probabilities
with binomial Monte Carlo errors, mean enrollment/allocation with across-trial
errors, and pooled observed toxicity/response/category rates with trial-clustered
ratio errors. Unassigned-regimen rates are undefined rather than zero. One
replicate cannot estimate sample-based ratio or mean uncertainty.

Three focused aggregate tests pass in 1.64 seconds. Replay checks verify counts,
selection, mean enrollment and the ratio-error calculation; budget lower bounds
reject before consuming random state. Diagnostic review also fixed a
single-trial summary bug that discarded infinite Rhat. Aggregation preserves
infinity, reports diagnostic maxima and counts undefined-diagnostic steps.
Targeted lint/format/type checks pass. The new interface implements explicitly
configured complete-outcome OCs, without native timing or executable parity.

Array bounds include sufficient statistics together with one trial's state.
Likelihood/work limits apply to all replicates cumulatively, with truth
validation included. No process pool, repeated CI job or large simulation was
introduced.
