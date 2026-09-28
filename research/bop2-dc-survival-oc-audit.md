# BOP2-DC survival calendar simulation contract

This implementation adds calendar replay and Monte Carlo operating
characteristics for the existing exponential/inverse-gamma survival monitor.
It does not add the paper's separate finite-grid design-calibration routine.

## Statistical model and terminal decisions

The cached primary paper is `research/raw/BOP2-DC/paper.txt`. Section 2.1.3
(printed pp. 8–9; text lines 204–231 and 318–362) models event duration as
exponential with mean `theta_tilde`, puts an inverse-gamma prior on that mean,
and reports median survival `theta = log(2) * theta_tilde`. With `d` observed
events and total observed time `T`, the posterior is
`IG(prior_shape + d, prior_scale + T)`. Thus
`P(theta > m) = P(Gamma(shape, rate=1) <
(prior_scale + T) * log(2) / m)`, the lower regularized gamma probability
computed by `BOP2DCSurvivalDesign.monitor`.

At an interim look `n`, stop for no-go only when both LRV and CMV posterior
probabilities are strictly below their respective `lambda * (n/N)**gamma`
cutoffs; otherwise continue. The paper does not use superiority stopping at
interims. At the final sample size, go requires both tails strictly above
their fixed cutoffs, no-go requires both strictly below, and all other cases
are consider. Equality therefore continues at an interim and is consider at
the final look. The replay calls the existing monitor directly, preserving
these comparisons and terminal categories.

## Calendar convention

At a configured interim look, enrolled patients are administratively censored
at the arrival time of the last patient in that look. The final analysis is
at the last enrollment plus caller-supplied final follow-up. Events on the
analysis boundary count as observed. A replay stops at the first interim
no-go. This is a Python calendar protocol for applying the source posterior;
the paper does not define an arrival-gap distribution or claim exact software
calendar/RNG parity.

For generated trials, truth is exponential with mean
`true_median / log(2)`, and only administrative censoring is simulated. The
existing fixed and Poisson arrival modes are explicit simulation choices;
the paper's example states accrual rate and follow-up but does not pin a gap
law. The aggregate result returns its effective integer RNG seed, which
replays the whole simulation call.

## Operating-characteristic scope

Reported disjoint outcomes are early `stop_no_go` and final `final_go`,
`final_consider`, or `final_no_go`. The simulator summarizes their counts,
probabilities, binomial Monte Carlo standard errors, enrollment, observed
events, total observed time, and duration. It does not infer null/effective
truth values, calibrate cutoffs, or optimize the candidate grid. The paper's
separate calibration objectives and error metrics are described in §2.3 and
§3.1 and remain a separate feature.

## Bounded resources

The serial simulator limits survival-path cells, summed repeated-look work,
and retained per-trial summary cells before consuming the caller's RNG. Paths
are generated in bounded chunks; only compact per-trial summary arrays and
the current path chunk are retained.
