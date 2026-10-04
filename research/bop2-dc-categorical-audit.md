# BOP2-DC generalized categorical endpoint audit

## Primary-source contract

The implementation follows BOP2-DC §§2.1.4, 2.2, 2.3 and 2.4 in Zhou et
al., *Bayesian optimal phase II design for multiple endpoints and randomized
controlled trials*, [arXiv:2112.10880](https://arxiv.org/abs/2112.10880).
The cached primary text used for this implementation is
`research/raw/BOP2-DC/paper.txt` in the source repository; the independent
implementation checkout did not modify or redistribute that source.

Section 2.1.4 defines the arbitrary joint-category representation: the
category counts have a Multinomial likelihood, the category probabilities a
Dirichlet prior, and their posterior is Dirichlet with prior shapes plus
counts. For an endpoint row `b` of zero/one category indicators, its event
probability `theta = b p^T` has the stated Beta posterior with shapes
`b(alpha + X)` and `(1-b)(alpha + X)`. The source says extension beyond two
categorical endpoints is straightforward but does not prescribe a particular
flattening of multidimensional categories. This API therefore treats the
caller-supplied columns as the joint categories and rows as explicit endpoint
membership definitions.

Section 2.2 applies endpoint-level dual-criterion actions and composes them:
for multiple endpoints, any endpoint may establish go and all must be no-go
to stop; for co-primary endpoints, all must establish go and any no-go stops.
Section 2.4 applies the same rules to treatment-minus-control posterior
differences in independent arms and specifies O'Brien–Fleming probability
cutoffs for optional randomized interim graduation. Section 2.3 defines
candidate-selection objectives based on correct-go rate or futile expected
sample size under false-go/false-no-go (and optional false-consider)
constraints. The code provides caller-supplied finite candidate comparison;
it does not reconstruct an unreported native grid, tie-breaking policy, or
native UI behavior.

For single-arm interim futility, §2.2 uses the cutoff
`lambda * (n / N)**gamma`, with `n` the current look and `N` the planned
maximum sample size. The independent reference generator was checked against
this printed formula; its initial reciprocal fraction was corrected before
the final comparison.

## Implemented scope and explicit choices

The categorical monitor supports both single-arm and fixed-allocation RCT
designs, arbitrary supplied binary indicator rows, `any` and `all` endpoint
composition, scheduled monitoring, absorbing replay, serial operating
characteristic simulation, and finite cutoff/exponent candidate comparison.
The Dirichlet prior is explicit. No endpoint dependence is discarded in
simulation: each arm samples one joint category per patient. In randomized
simulation, within each trial the deterministic seed generates control-arm
categories first and treatment-arm categories second, then stores them in the
fixed allocation slots. This ordering is part of seed replay behavior.

At candidate comparison, `cgr` maximizes correct-go probability and
`futile_ess` minimizes expected sample size under the caller-supplied futile
truth; false-go and false-no-go estimated limits, plus optional
false-consider, define feasibility. MCSEs are returned, but feasibility is
based on point estimates and is not guaranteed error control. An independent
confirmation run is needed after selecting a candidate to avoid selection
optimism.

For randomized endpoint tails, the existing Beta-difference quadrature is
reused. Combined-action stability is checked at all-lower and all-upper
favorable-probability interval corners. The any/all decision maps are
coordinatewise monotone, so if these extreme corners agree, intermediate
corners agree as well; this avoids exponential enumeration. Quadrature error
estimates are not rigorous mathematical bounds. Single-arm tails use the
regularized incomplete Beta functions.

## Validation record

Focused tests cover endpoint Beta aggregation, direction-sensitive tails,
`any`/`all` composition, randomized arm-specific Beta differences, fixed
allocation replay, serial seed reproducibility, calibration summaries, and
pre-conversion rejection of an oversized indicator row. The independent
base-R reference checker passed 36 posterior probabilities, final/interim
decision checks, replay, and reduction to the established two-endpoint paired
design. A bounded two-candidate, two-truth replay test also reconstructs
terminal rates and sample-size summaries from stored per-trial seeds for both
calibration objectives.
