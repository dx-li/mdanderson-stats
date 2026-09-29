# rBOP2 finite-grid calibration source audit

The cached binary-efficacy and binary-toxicity help text describes a calibration
objective: choose futility/superiority probability cutoffs to maximize power
while controlling type-I error. For efficacy, the stated boundary null is
`p_E = p_C + margin`; for toxicity, it is `p_E = p_C - margin`. It separately
describes a vague/null-centered calibration prior and an informative prior for
analysis. The help files do not expose the native numerical cutoff grid or
search/rounding details.

This implementation makes the candidate grid and both priors explicit. It
uses the existing exact finite-state binary operating-characteristic engine
and selects among supplied cutoff curves only. Therefore it implements the
source's stated statistical objective, but does not claim a match to native
candidate generation or application output. Null and alternative rates are
explicit inputs; the relation above is source context, not silently enforced
on caller-supplied values.

The prior help's separation between calibration and analysis is retained in
the API. Feasibility and ranking use `calibration_prior`; returned
`analysis_*` characteristics use `analysis_prior` when one is supplied. They
may consequently have a different type-I error. A candidate is feasible only
when its calibration-null overall-positive probability is `<= alpha` using a
direct floating-point comparison. No numerical tolerance changes this rule,
and the constraint covers only the supplied null rate pair, not a composite
null or unknown control-arm rates.
Equal-power candidates are ordered by smaller calibration-null expected total
enrollment, then original input order.

`max_posterior_probability_error` records the largest estimated statewise
posterior integration error among the two evaluated priors. It is not an
operating-characteristic error bound. The existing monitor's near-cutoff
resolution behavior is preserved: exact rational beta comparisons where
available, otherwise tighter quadrature and an error if still unresolved.

Sources: cached `BEhelp1.txt`, `BEhelp2.txt`, `BThelp2.txt`, and
`SimNumberHelp.txt`; article, “Bayesian Optimal Phase II Design for
Randomized Clinical Trials,” DOI [10.1080/19466315.2022.2050290](https://doi.org/10.1080/19466315.2022.2050290).
The application help says it uses simulations, whereas this Python method
uses deterministic exact binomial recursion for the declared fixed arm-size
looks. This is a computational choice, not native parity.

Validation uses `tools/reference_rbop2_calibration.py`, a separate
standard-library rational calculation that integrates integer-shape beta
polynomials and enumerates all 16 complete two-look binary outcome paths.
The committed fixture covers five candidate curves (including a duplicate),
both endpoints and distinct calibration/analysis priors. Sixty candidate
diagnostics and four selected-design outputs agree within `2e-14`; both
endpoints select candidate index 3. Duplicate-only input selects the first
candidate, and infeasible input returns no selected design.

Twenty-four additional operating-characteristic comparisons cover fractional
priors, positive/negative margins and unequal arm sizes against the existing
binary engine. Fifteen focused tests, Ruff and mypy pass; the focused test
process peaked at 133.23 MiB with zero swaps. Candidate feasibility is not a
confidence bound on a simulated estimate: it uses deterministic recursion
and the declared direct floating-point comparison.
