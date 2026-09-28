# U2OET GAO probabilities and likelihood — September 28, 2026

The [author-hosted 2017 paper](https://odin.mdacc.tmc.edu/~pfthall/main/JRCCS_2017_ph12_2agent_utility.pdf)
Appendix A, page 22, defines the GAO comparison using raw-dose linear
predictors, endpoint-specific positive link shapes and one shared positive
interaction coefficient. It combines ordinal margins with a Gaussian copula.
The earlier general Aranda–Ordaz discussion on page 5 supplies the positive
shape domain and logistic/complementary-log-log limiting cases. Source hashes
are retained in [u2oet-sources.json](../docs/u2oet-sources.json).

The inspected text does not give sign or ordering restrictions for GAO slope
and intercept arrays. The implementation therefore accepts explicit finite
coefficients without importing PDS slope restrictions or native prior
assumptions. Continuation-ratio products produce each ordinal marginal.
The existing FGM probability kernel is not used for GAO's joint distribution.

The U2OET 1.8 user guide identifies model type 2 as GAO and describes generic
pseudo-trial prior calibration, but it does not establish the native GAO
prior-vector coordinate ordering and transformations. The archive's vector
lengths do not establish those meanings. Native priors, GAO posterior fitting,
and native fitting/calibration parity remain pending.

## Independent evidence and numerical scope

`tools/reference_u2oet_gao.R` directly evaluates the published equation and
integrates conditional-normal rectangles in base R. It covers binary,
four-by-three-category ordinal and small-link-shape cases, each at four dose
pairs and correlations -1, -.65, 0, .55 and 1. Its 440 joint cells and 15
grouped likelihoods agree with the Python implementation; maximum absolute
cell error is `9.992007221626409e-16`. The comparison took 0.0599 seconds after
import, peaked at 116.34 MiB, and reported no swaps. The reference generator
ran with warnings treated as errors. These are independent equation results,
not values captured from the native executable.

Log-domain marginal continuations and products avoid overflowing ordinary
exponentials. Independence preserves their log sums. Nonzero nonsingular
correlation uses log-tail Gaussian quantiles and the existing deterministic
conditional-normal rectangle integration, with total-mass and marginal checks.
Unresolved positive rectangles and unrepresentable thresholds raise errors.
Limiting correlations use ordinary interval overlap and can lose extremely
small intervals to rounding; no extreme Gaussian log-tail accuracy is claimed.

The public [GAO guide](../docs/u2oet-gao.md) distinguishes conditional parameter
evaluation from posterior inference and explains dose units and association.
No new numerical dependency or CI workflow was introduced.

Luna committed the component as `24072c2`. Its three focused tests passed in
1.52 seconds, covering the analytic logistic link, Gaussian dependence and
marginal preservation, and the small-shape complementary-log-log limit.
Ruff lint/formatting and targeted mypy checks passed. Root integration retains
the same numerical source validated by the independent R comparison.

## Earlier GAO parameterization is not interchangeable

The [2010 primary paper](https://odin.mdacc.tmc.edu/~pfthall/main/Biometrics_2dose_utility.pdf)
(DOI 10.1111/j.1541-0420.2009.01302.x), sections 3.2 and 4.1, uses centered
doses and an interaction for each endpoint. It describes normal priors for
linear coefficients and interactions, lognormal link shapes, and uniform
correlation. Negative interactions are permitted only subject to valid
probabilities. These details differ from the 2017 appendix's shared positive
interaction implemented here. The older priors must not be transplanted into
the current model as though they establish the native executable's contract.

The archived four-category mean vector contains enough values to warrant
investigating this distinction, but its length alone does not identify ordering,
transformations or constraints. A faithful native GAO fitter still requires
that mapping; an explicit-prior fitter for the 2017 model would be a separately
declared implementation choice. This source finding narrows the next audit and
does not change the existing probability API or its validation claim.
