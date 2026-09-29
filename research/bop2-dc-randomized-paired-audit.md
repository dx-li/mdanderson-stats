# BOP2-DC randomized paired-endpoint monitor

The implementation covers the fixed-allocation randomized paired binary
endpoints in the BOP2-DC paper, using caller-specified four-cell Dirichlet
priors. The joint cell order is `(both, endpoint 1 only, endpoint 2 only,
neither)`. Marginal Beta posteriors retain each endpoint's correct marginal
counts while the four-category observation tape preserves the joint outcomes
for replay and later operating-characteristic calculations.

The monitor evaluates treatment-minus-control risk differences at the two
endpoint-specific LRV and CMV margins. For efficacy/toxicity, the efficacy
criterion uses the upper risk-difference tail and the toxicity criterion uses
the lower treatment-minus-control tail. The `multiple_efficacy` mode composes
the endpoint rules with OR at final go and both endpoints required for interim
no-go; the `efficacy_toxicity` mode uses AND for final go and either endpoint
for interim no-go. Optional paired interim graduation composes those same
endpoint rules with the source O'Brien-Fleming cutoffs. The paper does not
provide separate paired-graduation pseudocode, so that composition is a
transparent Python convention rather than a verified native-app detail.

Quadrature error estimates are propagated to each of the four marginal tails.
A final combined decision is returned only when all 16 combinations of the
reported endpoint/criterion error-interval corners give the same action. The
per-endpoint action fields are explicitly nominal midpoint summaries; the
combined decision is the uncertainty-guarded result. Monitoring between
configured looks reports `continue`, and the prior-only state at total N zero
is valid.

The API accepts a fixed 0/1 allocation tape and a complete four-category
outcome tape. It stops replay at the first terminal look and retains only the
observed prefixes. It does not model accrual, delayed endpoint ascertainment,
or native randomization generation. Input, prefix-work, comparison-work, and
quadrature work are bounded before replay or posterior calculations. This is a
source-based Python implementation, not a claim of executable parity with the
native application.

## Monte Carlo operating characteristics

`simulate_bop2_dc_randomized_paired` accepts one explicit joint four-cell truth
vector per arm. It samples one categorical outcome for each patient according
to the fixed allocation tape, then replays the configured looks. The helper
reports per-look decision counts and unconditional probabilities/Monte Carlo
standard errors, terminal decision probabilities, terminal sample sizes, and
expected enrollment with its Monte Carlo standard error. A single seed or RNG
creates separate per-trial seeds; each returned seed can be passed to
`default_rng` to reconstruct that trial's category tape in allocation order.

Posterior tails are cached by look and endpoint-specific arm success counts,
so a repeated marginal comparison is integrated once even when paired
outcome histories differ. Preflight limits cover patient paths, distinct cache
entries and their conservative quadrature work, and aggregate result/workspace
cells. The simulation retains no per-patient or posterior-fit history. Its
truth, RNG, and Monte Carlo workflow are Python additions; they do not claim
native app simulation parity.
