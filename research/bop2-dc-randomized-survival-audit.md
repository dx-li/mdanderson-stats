# BOP2-DC randomized survival comparison

## Statistical contract

The randomized two-arm section of the BOP2-DC paper (§2.4, cached at
`research/raw/BOP2-DC/paper.txt`) compares treatment-minus-control median
survival differences against explicit signed lower and clinically meaningful
margins. The event-time model is exponential with an inverse-gamma prior on
each arm's mean survival. With `d` observed events and total observed exposure
`t`, the conjugate posterior is inverse-gamma with shape `a+d` and scale `b+t`;
the corresponding exponential median is `log(2) * mean`. The implementation
therefore evaluates `P(log(2)*(mean_E-mean_C) > margin)` rather than a hazard or
survival ratio.

The caller must supply both positive inverse-gamma shape/scale priors, the
fixed 0/1 arm-assignment tape, and total-enrollment looks. The paper does not
specify a universal allocation ratio. No randomization policy or operating
policy is included here; aggregate simulation follows the caller-specified fixed
allocation tape.

## Probability computation and replay

At a zero median-difference margin, the comparison reduces exactly to a beta
ratio probability for the independent reciprocal-mean gamma variables, using
the existing stable log-beta-CDF helper. Nonzero margins use a one-dimensional
integral over the control reciprocal-mean gamma CDF scale, transformed to a
uniform gamma-quantile coordinate. The reported quadrature error is an estimate,
not a rigorous bound; the shared randomized-decision helper checks all four
error-interval corners and raises if a strict action could change.

Calendar replay accepts complete event-duration tapes (`+inf` means no event)
and ordered enrollment times. Each scheduled look administratively censors at
its as-of calendar time, stratifies event count and exposure by the fixed arm
tape, and stops at the first no-go or optional graduation. Final follow-up is
added after the last enrollment. Input order breaks tied arrival times. Native
randomization, accrual timing, and RNG parity are not claimed. Aggregate
randomized-survival operating characteristics and finite-grid calibration are
implemented as bounded Python extensions. Native RNG/timing parity remains out
of scope.

Replay preflights total numerical work before its first posterior calculation.
The conservative bound charges `21 * (2 * 300 - 1)` quadrature evaluations
for each nonzero margin at every scheduled look, with a 20-million-evaluation
replay limit; the monitor separately caps each batch. Explicit event duration
zero is supported as one observed event with zero exposure, the valid
exponential likelihood boundary (although it has probability zero under a
continuous event-time generator).

The aggregate simulator uses the fixed allocation tape in the design, arm-specific
exponential truth medians, and the single-arm survival simulator's fixed or
exponential interarrival conventions. It replays each trial through the same
calendar runner and stores only per-trial enrollment, arm counts/exposures,
terminal decision (including interim graduation when enabled), and duration.
The default of 100 trials is a workload choice, not a paper default. Preflight
bounds path cells, repeated-look scans, retained summaries, and the conservative
inverse-gamma quadrature total before seed use. NumPy RNG and calendar parity with
the source software are not claimed.


## Independent reference and integrated verification

`tools/reference_bop2_dc_randomized_survival.R` integrates directly against a
standardized control Gamma density, splitting at the negative-margin support
boundary. It uses neither Python's Gamma quantile transformation nor its
calendar implementation. Ten fixed tapes and 17 reached analyses cover all
terminal actions, an empty arm, zero-duration events, tied arrivals, pending
censoring, unequal allocation and `1e-100`/`1e100` time-unit changes. A
nonsymmetric zero-margin Beta-ratio case independently verifies probability
`2/3` and the experimental-minus-control orientation.

The checker also supplies exact fixed/Poisson-arrival and exponential-event
tapes for 48 simulated trials with 71 reached looks. R independently computes
as-of sufficient statistics and stopping decisions; all per-trial arm counts,
events, exposure, calendar times and decision summaries agree, as do aggregate
means and Monte Carlo errors. The 763 checked numeric summaries include 24
interim graduations. Maximum posterior discrepancy is `4.922e-11`, within
reported numerical errors. The check takes .5517 seconds after imports;
Python and R peaks are 110.13 and 82.67 MiB (a conservative combined upper bound
of 192.80 MiB), with zero Python swaps.

Parent review repaired an omitted graduation category in simulated frequency
summaries and expanded the retained-memory estimate to include owned copies,
Unicode decision storage and replay scratch. Decision counts must conserve
all trials. Five monitor/replay tests and four simulation tests pass, including
an explicit graduation regression, unit invariance and pre-RNG work rejection.

## Finite-grid calibration

The finite-grid calibrator accepts explicit futile and effective control/treatment
median pairs. It requires positive arm medians, a strictly ordered treatment-effect
difference, and an effective difference at least CMV; the caller-declared futile
difference is not constrained to be at or below LRV. It varies only the four
lambda/gamma grids while preserving the design's priors, fixed allocation, looks,
graduation setting, and margins. Shared standardized arrival and event draws are
reused across candidates and both truth pairs. Posterior tails/error estimates are
computed once per truth, trial, and look, then reused; candidate decisions must be
invariant over all four estimated-error interval corners.

Calibration and validation use separate seeds. The selected candidate is checked
on independent validation paths without reselection. False-go counts interim
graduation or final go under the futile truth; false-no-go counts interim no-go or
final no-go under the effective truth; correct go includes graduation. Optional
final-consider control uses the larger of the two truth-specific rates. The
empirical finite-grid constraints are not guarantees about true operating
characteristics. The calibration default of 100 trials per stage is a workload
choice, and a hard work preflight limits comparisons, candidate decisions, path
scans, and retained tables before RNG use.
