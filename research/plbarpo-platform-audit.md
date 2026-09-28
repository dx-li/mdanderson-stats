# PLBARPO platform extension contract

The official [PLBARPO application](https://biostatistics.mdanderson.org/shinyapps/PLBARPO/)
and its support document were inspected on September 28, 2026. The saved local
snapshot is under ignored `research/raw/PLBARPO/`; the page identifies version
2.0.3.0, updated January 6, 2026. The current Python implementation provides
[control selection and monitoring](../docs/plbarpo-control.md) and
[active-arm allocation](../docs/plbarpo-allocation.md). Catalog entry 137
remains partial: platform trial progression is a separate missing workflow.

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

Next is a no-control, complete-outcome platform trial with explicit entry and
replacement order, enrollment limits, global monitoring looks and entrant-only
burn-in. These are declared Python scheduling choices where the source does not
fully specify native transitions. Preserve compact operating-characteristic
summaries, zero allocation to inactive arms, and serial bounded computation.
