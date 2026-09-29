# Randomized Normal BOP2-DC finite-grid calibration

This implementation calibrates the existing randomized Normal monitor over an
explicit four-dimensional grid of LRV/CMV cutoffs and spending exponents. It
uses the paper's independent arm-specific Normal-Inverse-Gamma posterior and
Student-t difference probability calculation. It does not replace the
posterior comparison with a normal approximation.

The caller supplies two `(control mean, treatment mean)` truth pairs and a
separate `(control SD, treatment SD)` pair for each scenario. The effective
mean difference must meet the design's CMV. The allocation tape, looks, priors,
and interim graduation choice are reused from the supplied design. Calibration
uses common standardized paths across candidates; the selected candidate is
then evaluated on an independent holdout stage without reselection. False-go
includes interim graduation and final go, while false-no-go includes interim
and final no-go. Optional false-consider control is the maximum across the two
scenarios.

The Python defaults of 100 calibration and 100 holdout paths are workload
choices, not source trial settings. All candidate evidence, stage seeds,
Monte Carlo errors, and quadrature error estimates are retained. Candidate ties
follow grid product/input order. If no grid point meets the requested empirical
limits, the optimizer raises rather than returning a fallback. Common additive
location offsets are removed before posterior fitting; difference margins
remain unchanged under that translation.

Preflight bounds cover path storage, candidate evidence, Student-t quadrature
work, and four-corner decision-stability checks over the look schedule. These
limits intentionally reject oversized jobs before random numbers are drawn.

## Independent numerical evidence

The standalone R reference integrates the control Student-t density against
the treatment tail, using directly updated NIG parameters. A deterministic
checker exports truth-centered Normal paths and independently replays all
candidates and the selected holdout. Three configurations (both objectives
and disabled graduation), each with 12 candidates, use 24 calibration and
18 validation trials per truth. All 1,092 probability/enrollment/MCSE summaries
and selections agree. CGR selects index 0; futile ESS selects index 6. The
ESS candidate meets the 13% no-go calibration limit at 3/24 but exceeds it
on independent validation at 3/18; the selected candidate is not replaced.
Control means and arm SDs differ between scenarios.

The checker completes in 33.881 seconds after imports, with Python peak RSS
118.80 MiB and R peak 86.66 MiB, zero swaps. These process peaks have a
conservative combined upper bound of 205.46 MiB. Nine focused optimizer/core
tests pass. Review also corrected combined-event arithmetic to sum integer
counts before division, ignored already-absorbed paths during decision-error
checks, and included holdout work and four-corner string workspace in preflight.
The independent checker is `tools/check_bop2_dc_randomized_normal_calibration.py`;
its generated inputs/results stay in ignored `research/raw/`.
