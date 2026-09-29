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
randomized-survival operating characteristics are implemented as a bounded
Python extension. Calibration and native RNG/timing parity remain out of scope.

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
