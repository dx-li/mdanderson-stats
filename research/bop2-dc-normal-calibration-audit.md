# BOP2-DC continuous Normal endpoint

The primary paper's §2.1.2 specifies `Y_i ~ Normal(theta, sigma^2)` with a
Normal-Inverse-Gamma prior: `theta | sigma^2 ~ Normal(theta0, sigma^2/n0)` and
`sigma^2 ~ IG(a2, b2)`. The Python API requires these four values explicitly and
uses the exact conjugate posterior; it does not infer BCSTTE/app priors. The
marginal posterior for `theta` is Student-t. Section 2.2 supplies the strict
dual-criterion rule: an interim no-go requires both upper-tail probabilities to
fall below their information-scaled cutoffs; final go/no-go require both strict
comparisons in the same direction, with consider otherwise.

The complete-data replay and serial Normal-truth simulation implement those
rules at explicit sample-size looks. No accrual-time or missing-data model is
implied. The finite-grid optimizer reuses standardized Normal paths across both
truth means and all candidate settings. It enforces empirical false-go and
false-no-go limits and optional false-consider limits; Monte Carlo estimates
and MCSEs are not guarantees of the true operating characteristics. A separate
SeedSequence child generates independent validation paths. Validation failure
is reported without reselection.

Calibration is bounded by 10,000 candidates, 100,000 trials per stage, a
one-million-cell path budget, 50 million candidate/path work units, and a
two-million-cell combined evidence/workspace budget. These are Python resource
limits, not source defaults. Candidate order follows the product order of the
four caller-supplied grids, and exact ties retain the first candidate.
