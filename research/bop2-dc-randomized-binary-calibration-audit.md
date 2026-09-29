# BOP2-DC randomized binary finite-grid calibration

The randomized-trial extension in §2.4 of the BOP2-DC paper applies the same
strict dual-criterion cutoffs to the posterior probabilities that the
experimental-minus-control response difference exceeds LRV and CMV. Section
2.3 defines FGR, FNGR, CGR, FCR, the CGR and futile-expected-N objectives, and
finite grid search over the four cutoff parameters. The paper's simulation
uses control response .2, futile treatment response .2, effective treatment
response .4, LRV 0, CMV .2, and 2:1 planned allocation; the implementation
requires both truth pairs explicitly rather than assuming the control rate is
shared or fixing the paper's example.

The Python calibration is exact recursion conditional on the supplied fixed
allocation tape. It uses FGR = early graduation plus final go at the futile
pair, FNGR = interim plus final no-go at the effective pair, CGR = early
graduation plus final go at the effective pair, and optional FCR as the larger
final-consider probability across pairs. This early-graduation inclusion is a
clearly stated randomized-design convention. Exact recursions have no Monte
Carlo error or independent holdout; numerical uncertainty is represented by
the cached Beta-difference quadrature diagnostics and the core's fail-loud
decision stability check.

The posterior response-count comparisons depend on the priors, allocation,
looks, margins and quadrature tolerance, but not on lambda/gamma. The
calibrator therefore computes all count-state tails once and reuses them for
every candidate; only strict-decision classification and exact Bernoulli
recursion are repeated. The source does not specify a randomization schedule,
so the result is conditional on the chosen tape and does not average over
allocation schedules. Candidate selection is only over the caller's finite
grid; ties preserve product/input order after the stated primary and secondary
objectives.

## Independent check

`tools/reference_bop2_dc_randomized_binary_calibration.R` integrates Beta
differences using an integer-shape finite-polynomial CDF and enumerates all
16 response paths for each four-patient candidate. The checker compares 66
candidate configurations in five grids: CGR with graduation, futile ESS with
graduation, no graduation, duplicate-candidate ties, and an infeasible grid.
All 1,980 probability/enrollment summaries agree within 2e-14 absolute error;
feasibility and selection agree. CGR selects index 0 and ESS index 4 in the
shared graduation grid. Runtime after imports was 0.264 seconds, peak RSS
109.79 MiB, and zero swaps on the validation machine.

This reference exposed floating accumulation changing an exact expected-N tie
by 4e-16. Selection and inclusive OC limits now recognize only relative
binary64 roundoff, scaled by trial length, without an absolute floor that
would erase rare-event differences. Posterior decision cutoffs remain strict.
A separate guard rejects whole-grid cutoff underflow before table allocation,
and resource preflight accounts for every shared-rule schedule scan and live
four-corner string workspace. No additional CI workflow was introduced.
