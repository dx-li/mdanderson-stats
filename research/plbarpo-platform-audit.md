# PLBARPO platform extension contract

The official [PLBARPO application](https://biostatistics.mdanderson.org/shinyapps/PLBARPO/)
and its support document were inspected on September 28, 2026. The saved local
snapshot is under ignored `research/raw/PLBARPO/`; the page identifies version
2.0.3.0, updated January 6, 2026. The current Python implementation provides
[control selection and monitoring](../docs/plbarpo-control.md) and
[active-arm allocation](../docs/plbarpo-allocation.md), plus
[no-control platform trials](../docs/plbarpo-trials.md) and
[persistent-control trials](../docs/plbarpo-control-trials.md). Catalog entry 137
remains partial: control-trial aggregate simulation and delayed outcomes are
separate missing workflows. Compact no-control operating characteristics are
also available.

## What the support document establishes

The platform has a maximum number of arms and a current active-arm capacity K.
An arm that stops or reaches its sample-size cap can be replaced until the
maximum arm count or total enrollment is reached. Trial monitoring also permits
manual opening and closing. Per-arm beta-binomial posteriors, minimum enrollment
before early stopping, maximum enrollment per arm and total trial enrollment
are explicit inputs. Monitoring can be manual or by cohort size.

Equal randomization uses blocks of size K. Adaptive randomization uses one of
four methods throughout the trial and permits an initial fixed-randomization
sample size per arm. With q_i the posterior probability that arm i is best,
the documented unnormalized adaptive weights are:

- BARCP: `q_i ** tau`.
- BARN2N: `q_i ** (n_total / (2 * N_trial))`.
- BARMTV: `sqrt(q_i * Var(theta_i) / (n_i + 1))`, following the previously
  audited BARPO formula and the cited Connor et al. method.
- DBCD: `[y_i * (y_i / x_i) ** tau] ** tau1`. The desired targets y_i must be
  supplied because this document does not define how to estimate them.

The control is the first arm. Monitoring supports all controls or controls
concurrent with a treatment. Raising a control probability to its specified
minimum rescales other arms proportionally. Experimental-arm floors are also
allowed, but the simultaneous multi-floor projection is unspecified; the
existing explicit BARPO projection policy can be reused and labeled as such.

## Implementation implications

Interpret the K-arm best-probability calculation over currently active arms,
recompute that posterior competition and map inactive arms back to probability
zero. This follows the guide's active-arm formulation; native executable parity
has not been verified. Passing a stopped mask to `barpo_allocation` alone is
insufficient: its supplied posterior retains the original competition set.

BARN2N's n is all patients enrolled in the trial, including patients on closed
arms. Slicing counts to active arms and calling the current BARN2N helper would
lose those historical enrollments. Keep a separate total or use BARCP with the
explicit global exponent. DBCD's common scaling of x over active rather than
all historical assignments cancels after normalization, but zero-count arms
still need a declared initial-allocation policy.

Reuse the bounded numerical posterior kernels, including their failure/error
estimates. The existing BARPO joint best-probability implementation permits at
most ten competing arms. PLBARPO control monitoring's larger independent
comparison limit does not justify increasing joint competition to 100 arms.

## Protocol choices that must remain explicit

The source does not resolve newcomer burn-in alongside older adaptive arms,
replacement ordering, block reset after K changes, simultaneous closing/opening
and stopping order, control replacement, or reopening an arm with old data.
These choices need explicit caller inputs and documented Python behavior before
a trial simulator can be called reproducible. Do not infer a single native
scheduler from the formulas alone.

## Active allocation checkpoint

Luna checkpoint `d1dfb3e` supplies the current-active-arm allocation API, a
validated full arm ledger and the global enrollment count. Root's independent
base-R checks matched all eight scenarios, including a closed dominant arm:
recomputing the two remaining competitors gives `(5/6, 1/6)`, while masking the
original three-arm probabilities would incorrectly give `(15/16, 1/16)`.
The fixture generator and the public [allocation guide](../docs/plbarpo-allocation.md)
record formulas, scope and validation. A second read-only review found no
material issue in the count, active competition or global-N contracts.

The trial tranche adds no-control, complete-outcome trials with explicit entry and
replacement order, enrollment limits, global monitoring looks and entrant-only
burn-in. These are declared Python scheduling choices where the source does not
fully specify native transitions. Preserve compact operating-characteristic
summaries, zero allocation to inactive arms, and serial bounded computation.

## Trial reference preparation

`tools/reference_plbarpo_trial.R` records three hand-specified replays of the
declared Python platform protocol with independently calculated base-R beta
tails and BARN2N probabilities. The fixtures cover cap closure between global
looks, replacement, final efficacy at a cap, serial futility/efficacy replacement
and simultaneous all-arm futility without another candidate. Their 14 patient
rows, 12 monitoring comparisons and nine final-ledger rows are small and
deterministic. They describe the declared Python scheduler, without asserting
that the native application uses the same scheduler.

## Integrated no-control trial

Luna checkpoint `c2a2902`, integrated as `eba837a`, adds the complete-outcome
trial controller and immutable patient, look and arm summaries. Root reran the
three independent R replay comparisons against the integrated module: all 14
allocations/outcomes, 12 posterior comparisons and nine final-ledger rows match.
The check took 0.011 seconds after import, peaked at 117.69 MiB and reported no
swaps. The public [trial guide](../docs/plbarpo-trials.md) example also passes.

Review corrected the burn-in/adaptive transition, terminal stop reasons,
independent optional monitoring criteria and cap-at-look decision flags. Three
focused regressions, lint/format and targeted type checks pass. Aggregate
operating characteristics remain the next tranche, followed by control/
concurrent-control scheduling; delayed responses and native reports are still
outside this no-control implementation. Catalog entry 137 remains partial.

## Integrated no-control operating characteristics

Luna checkpoint `0851603`, integrated as `31a63ea`, adds `simulate_plbarpo`.
It reuses the validated controller, prepares the design once and discards each
replicate's patient history after aggregation. All-trial and entry-conditional
arm rates, assignment/response means, trial enrollment and stopping summaries,
Monte Carlo standard errors and explicitly labeled null-arm error rates are
available. Trial seeds reproduce individual controller runs; starting a new
simulation from one of those seeds creates a new hierarchy instead.

The strengthened regression (`6086ce7`, integrated as `76c3861`) independently
replays every recorded seed and reconstructs all seven arm metrics, false
efficacy and familywise counts, means and sample standard errors. Two focused
simulation checks pass against the root integration, as do lint and targeted
mypy. Both public trial/simulation examples pass in 1.55 seconds including
import, at 106.86 MiB peak resident memory with no swaps. The small public
simulation is a usage example, not a precision benchmark. No new CI workflow,
dependency or large simulation was added for this feature.

## Integrated persistent-control trial

Luna checkpoint `a9af713`, integrated as `e3e81f0`, adds a persistent control,
entire-trial or treatment-specific concurrent comparisons, and complete-outcome
replacement scheduling. Control index zero participates in allocation but is
never an efficacy hypothesis or a replacement candidate. Final assessments at
an arm cap apply only to that arm. Simultaneous closures fill every vacant slot.
Actual comparison windows, control counts and numerical error estimates remain
in immutable look/final summaries; unassessed final values are explicitly NaN.

`tools/reference_plbarpo_control_trial.R` independently establishes four small
replays: staggered entry under both control modes, an unrelated active arm at
another arm's cap, and simultaneous futility with two replacements. All 94
allocation cells over 26 patients, 12 posterior comparisons and 14 final-ledger
rows match the root integration. The comparison took 0.042 seconds after import,
peaked at 117.98 MiB and reported no swaps. Recorded windows were also checked
against the actual underlying control observations.

Review found a same-look conflict when futility and cap-final efficacy both
declared an arm. Both controllers now reject that contradictory state, with
focused regressions (`2b60138`, integrated as `7f8e990`). The source's posterior
criteria are preserved; queue, burn-in and transition details remain explicit
Python conventions, without a claim of native scheduler parity.

Seven focused controller checks, targeted lint/type checks and the public
control-trial example pass after integration. The example takes 1.54 seconds
including import, at 105.38 MiB peak resident memory with no swaps. No new CI
workflow or large Monte Carlo run was added.
