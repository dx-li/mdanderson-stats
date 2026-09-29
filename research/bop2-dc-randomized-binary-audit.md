# BOP2-DC randomized binary comparison

## Source contract

The BOP2-DC paper's randomized two-arm section (§2.4, cached at
`research/raw/BOP2-DC/paper.txt`, around lines 466–495) compares the difference
between two response probabilities, `theta = p_E - p_C`, under independent
arm-specific beta-binomial models. The lower and clinically meaningful margins
are signed differences. The design applies its usual two-cutoff monitoring to
the posterior probability that `theta` exceeds each margin. Its optional
O'Brien–Fleming graduation boundary at interim total sample size `n` of maximum
`N` is `2 Phi(z_(1+lambda)/2 / sqrt(n/N)) - 1` for each margin. Final decisions
use the unadjusted final cutoffs and retain an equality case as consider.

The source does not prescribe one universal arm-allocation ratio. This module
therefore requires the caller to supply a fixed 0/1 allocation tape, cumulative
total-sample looks, and both beta priors. It does not generate a randomization
schedule or claim parity for such a schedule. Fixed-tape replay stops at the
first terminal scheduled decision. Exact operating characteristics enumerate
the reachable arm-response count states under explicit independent Bernoulli
truth rates; they do not retain individual response paths.

## Implementation and limits

`BOP2DCRandomizedBinaryDesign.monitor` and `.replay` use the shared
`compare_beta_difference` quadrature for each margin. `.operating_characteristics`
precomputes bounded posterior tail tables and propagates probability mass over
the arm-specific success-count grid. It reports interim no-go and optional
graduation probabilities, final go/consider/no-go probabilities, stopping-look
probabilities, expected sample size, and maximum comparison error by look.
The optional O'Brien–Fleming boundary is evaluated through the equivalent
`erf(erfinv(lambda) / sqrt(n/N))` form to retain small positive lambda values.

Before classifying, monitoring checks the four corners of each posterior-tail
interval formed from the reported quadrature error estimates. It raises an
`ArithmeticError` if any corner changes the combined action; the other margin
can still establish a stable action when one cutoff alone is straddled. The
comparison module explicitly describes these quadrature errors as estimates,
not rigorous bounds, so this guard is a fail-loud numerical diagnostic rather
than a proof of decision certainty. Exact symmetric comparisons report zero
error and preserve strict-cutoff equality as continue/consider.

Hard limits cap planned enrollment, monitor batches, posterior comparison
states, recursion work, truth scenarios, and retained OC results before the
corresponding large allocations. Numerical checks are recorded in the focused
test file; no native randomized trial output was used as a reference.

## Independent numerical reference

`tools/reference_bop2_dc_randomized_binary.R` calculates integer-shape Beta
CDFs from their finite binomial-polynomial formula and integrates the
independent arm difference directly. It includes the known
`Beta(2,1) - Beta(1,2)` zero-margin probability `5/6` and signed margins.
It then enumerates all 16 possible four-patient response tapes under four
designs: graduation off/on, a strict-equality configuration, and unequal
allocation with asymmetric priors and signed clinical margins. Three truth
pairs give twelve exact OC scenarios.

`tools/check_bop2_dc_randomized_binary.py` verifies 64 response paths, 100
reached analyses and 336 posterior/OC summaries. All reached and terminal
decisions, early graduation and stopping masses, final decisions, sample-size
distributions and expected sample sizes agree. The largest posterior-tail
discrepancy, `1.09375e-10`, is within the returned numerical error estimate.
The Python check takes .4811 seconds after imports, peaks at 119.64 MiB and
reports no swaps. Reference generation and comparison run serially.

Four focused tests additionally cover a cutoff-straddling error that can
change the action, an uncertain individual comparison whose combined action
is still stable, exact symmetry, tiny positive graduation cutoffs and exhaustive
small-state agreement. Shared decision rules are extracted for the remaining
randomized endpoint families; posterior and data models remain separate.
Randomized continuous/survival models and randomized calibration remain open.
