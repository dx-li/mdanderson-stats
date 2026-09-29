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
