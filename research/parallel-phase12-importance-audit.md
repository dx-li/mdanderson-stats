# Parallel Phase I/II importance-integration audit

## Source contract

The cached P12Xuelin six-dose C++ source is under
`research/raw/P12Xuelin/extracted/Phase12Xuelin/DFKernel/`.
`DFKernel.h`, `Kernel::operator()` (around lines 244–355), adds the four
independent normal log-prior densities to the response-binomial log likelihood.
`FullIntegrand.cpp` then subtracts `MixtureDistribution::LogPDF`, so the
importance integrand is posterior kernel divided by the density of the complete
mixture. `MixtureDistribution.cpp` defines that density and draws from a mixture
of the mode-centered multivariate normal and the product of independent normal
priors. The kernel configures the multivariate weight to `.99`.

For each proposal draw the native vector integrand contains component zero,
the evidence integral, then 60 summaries: six dose-zero comparisons, six
efficacy threshold indicators, six future-study threshold indicators, 36
ordered strict pairwise response comparisons, and six squared response
probabilities. `DFKernel.cpp` divides each summary integral by component zero.
The toxicity exceedance calculation is separate and analytic under the beta
prior.

`MyVVIS.h` computes the ordinary sample standard error from the vector sample
sum and sum of squares, checking every 100 draws. With relative target `r`, it
continues while any component standard error exceeds
`r * max(component_mean, .01 * evidence_mean)`. This is a raw-integral criterion;
it is not a ratio standard error. Native defaults in `DFKernel.h` are 10,000
integrations, `.001` target relative error, and `.99` multivariate proposal
weight; `TrialDesign.cpp` applies these to this kernel. The source returns its
estimate at the cap without a convergence flag.

## Python implementation choices

`fit_phase12_importance` implements the same normal-prior posterior kernel,
mixture-density importance ratio, source summary components and componentwise
stopping rule. It adds six posterior response-mean components, but they do not
change the native component stopping criterion. The result also reports
paired-draw delta-method standard errors for summary ratios and the log-scale
standard errors of the raw integral components. It labels cap exhaustion with
`converged=False` when the cap is reached before the final checkpoint meets the
criterion, and retains no proposal draws. Its convergence flag means only that
the raw-integral rule passed; it does not imply a requested relative MCSE bound
for every normalized ratio, particularly below the one-percent evidence floor.
Named ratio-MCSE outputs remain available for decision review. `log_evidence`
omits binomial combinatorial constants, matching the archived likelihood
kernel.

The mode is optimized by bounded BFGS using the analytic logistic gradient;
the inverse exact posterior Hessian supplies proposal covariance. The archived
program instead uses its private Nelder–Mead and curvature-regression
procedures, so optimizer/proposal covariance details, random-number stream and
bitwise outputs differ. No jitter is added to an ill-conditioned proposal:
non-positive-definite covariance fails clearly. The Python cap is 1,000,000
draws with 100-draw checks, and every request is limited to 67 million
summary-component evaluations. A zero-hit component receives zero empirical
standard error as in the native sample-moment stopping rule; this can make a
rare event appear precise when none of its indicators was sampled.

The existing `tools/reference_phase12_model.R` is a distinct independent
importance calculation using a `.95` inflated-Laplace / `.05` prior-normal
mixture. It validates posterior moments, not native proposal or stopping
parity. The C++ source is cached for source inspection only and is not
redistributed by this Python package.

## Focused validation

The 10,000-draw Python fit on the reference response table had maximum absolute
dosewise response-mean discrepancy `0.0010111` from the independent R
importance estimate; the largest discrepancy was `1.04` times the combined
reported Monte Carlo standard error. A separate empty-data check verified
log-evidence zero under the normalized normal prior, seeded replay, dose-zero
contrast `.5` under the equal prior mean logits, pairwise complementarity, and
the inequality `E[p²] <= E[p]`. A 100-draw replay independently recomputed the
paired weighted-ratio residual standard error and matched the returned MCSE.

Focused checks: 9 importance/model/decision tests passed in 4.00 seconds at
peak RSS 138,919,936 bytes with zero swaps; after the final prior-contrast
assertions, all 5 importance tests passed in 2.70 seconds at peak RSS
136,822,784 bytes with zero swaps. Targeted Ruff check/format, mypy on both
affected source modules, and `git diff --check` passed.
