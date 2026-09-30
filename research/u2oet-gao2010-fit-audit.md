# Original 2010 GAO posterior fitter audit

## Source-backed model and prior

The primary reference is Houede, Thall, Nguyen, Paoletti and Kramar,
“Utility-Based Optimization of Combination Therapy Using Ordinal Toxicity
and Efficacy in Phase I/II Trials,” *Biometrics* 66 (2010), 532–540,
DOI [10.1111/j.1541-0420.2009.01302.x](https://doi.org/10.1111/j.1541-0420.2009.01302.x).
The cached source is `research/raw/U2OET/gao2010.pdf`, SHA-256
`9ffc425871b3cd3f837ef22ce0bbec45ae247b06269928b7ad265332ccc344b8`; section
4.1 supplies the prior-family and variance statements. The probability
implementation is `u2oet_gao2010.py` and follows the section 3.1–3.2 model
equations described in `u2oet-gao2010-audit.md`.

Section 4.1 gives Normal priors for alpha/gamma, Lognormal priors for lambda,
and Uniform(-1,1) for the Gaussian-copula association. It reports
`Var(alpha)=10^2=100` and
`Var(log(lambda))=Var(gamma)=1.5^2=2.25`. The centers are obtained through an
extension of Thall–Cook least-squares elicitation. Neither those elicited
centers nor the native parameter-file coordinate mapping are reconstructed
here.

## Explicit Python posterior convention

The public fitter requires one Gaussian mean and SD per named coordinate,
with zero SD allowed for fixed values. It treats these Gaussian coordinates
as independent before imposing a single joint indicator that every endpoint
continuation bracket is strictly positive on the supplied dose grid and all
thresholds. It does not renormalize a gamma density conditionally at each
alpha/slope state. Association has a separate Uniform(-1,1) prior or may be
fixed by the caller. This defines a complete, reproducible posterior target
without claiming native prior-file or sampler parity.

The parameter order is a documented Python convention: per threshold, the
two intercepts followed by the two slopes; then endpoint log-lambda and gamma;
efficacy block, toxicity block, and raw association last. The observed-data
target uses the existing 2010 result's grouped complete and optional
toxicity-only likelihood. It does not add a model for evaluability/zeta.

The sampler uses Gaussian endpoint-block elliptical slice updates under the
joint support indicator and an independent Uniform(-1,1) Metropolis proposal
for association. This is not the paper's native Gibbs/two-level algorithm.
Callers must provide dispersed valid starts where needed; summaries and
acceptance are diagnostic estimates rather than convergence certificates.
Array/work/evaluation ceilings are documented in
`docs/u2oet-gao2010-fit.md` and checked before sampling begins.

## Validation

Focused tests exercise explicit coordinate ordering, fixed and free
association behavior, immutable retained draws, joint grid-support rejection,
and validation/budget failures before the supplied random generator advances.
Sixteen focused new/existing GAO tests pass with warnings treated as errors,
as do targeted Ruff, formatting and mypy checks. The worker run took 2.04
seconds, peaked at 131.77 MiB RSS and reported zero swaps.

The independent base-R reference integrates two variable Gaussian coordinates
(an intercept and signed interaction) jointly across the dose-grid support
boundary. It checks both the restricted prior and complete-data posterior.
The same marginal model supplies a toxicity-only reference. A separate
uniform-correlation posterior uses the analytic Gaussian quadrant formula
and integration over `asin(rho)`, including its `cos(angle)` Jacobian. Base-R
64/128-node refinements differ by at most `2.58e-10`.

Four integrated runs, each with two chains, 512 warmup and 4,096 retained draws,
pass all 31 posterior mean, variance, cross-moment and probability comparisons.
The maximum discrepancy is 1.891 estimated Monte Carlo errors; the maximum
classical split-Rhat among checked summaries is 1.00454. The run took 32.056
seconds, peaked at 116.58 MiB RSS and reported zero swaps. Reproduction uses
`tools/reference_u2oet_gao2010_fit.R` and `tools/check_u2oet_gao2010_fit.py`.
These reduced-dimensional comparisons validate the stated target and update
mechanisms, not mixing for every full-model analysis. This audit does not
claim native application parity or prior-center calibration.
