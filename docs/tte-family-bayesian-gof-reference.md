# Independent checks for the TTE-family posterior workflows

The independent base-R reference in
[`tools/reference_tte_family_bayesian_gof.R`](../tools/reference_tte_family_bayesian_gof.R)
uses only the Gamma, inverse-Gamma, and log-logistic equations from cached
BCSTTE guide sections 4.2–4.4. It checks explicit Gaussian priors and the
ordinary right-censor likelihood; those are independent Python conventions,
not recovered BCSTTE priors or fitting behavior. The R script uses tensor
Gauss–Legendre quadrature, no added packages, six complete/right-censored
cases, four quadrature settings, and writes a posterior summary CSV plus
analytic extreme-tail checks. The 91-node, ±9 standard-deviation summaries are
the comparison reference. The raw outputs are in
[`tte-family-bayesian-gof-reference.csv`](../tests/fixtures/tte-family-bayesian-gof-reference.csv)
and
[`tte-family-bayesian-gof-tail-reference.csv`](../tests/fixtures/tte-family-bayesian-gof-tail-reference.csv).

The bounded Python comparison used 4 serial chains, 500 warmup updates and
1,000 retained draws per chain, 20 contiguous batch means, and split R-hat. It
compares log-shape/log-scale means, shape/scale means, log-parameter variance
and covariance, and posterior mean CDFs evaluated with the same paired draws.
For censored cases, CDF summaries are formula checks only; they are not returned
as a censored Johnson diagnostic. The exact estimates, quadrature values,
Monte Carlo errors, absolute discrepancies, split R-hat values, and work counts
are recorded in
[`tte-family-bayesian-gof-python-comparison.csv`](../tests/fixtures/tte-family-bayesian-gof-python-comparison.csv).

The comparison initialized each chain at the supplied prior mean. It used
`numpy.random.default_rng(817 + 10 * family_index + case_index)`, with family
indices Gamma=0, inverse-Gamma=1, log-logistic=2 and case indices complete=0,
censored=1. For Monte Carlo errors, each chain's 1,000 retained draws were
split into 20 consecutive batches of 50; the 80 batch means were pooled and
their sample standard deviation (`ddof=1`) divided by `sqrt(80)`. The reported
split R-hat divides each chain into its first and last 500 draws and uses the
usual within/between-chain variance estimate on those eight half-chains. The
largest recorded split R-hat is 1.00544474 (inverse-Gamma censored).

| Case | Largest discrepancy / batch-means MCSE | Metric | Maximum split R-hat | Likelihood evaluations / work units |
|---|---:|---|---:|---:|
| Gamma, complete | 3.23 | log-shape variance | 1.003 | 13,764 / 68,820 |
| Gamma, censored | 1.62 | first posterior CDF mean | 1.001 | 11,656 / 58,280 |
| Inverse-Gamma, complete | 1.88 | log-shape variance | 1.000 | 14,686 / 73,430 |
| Inverse-Gamma, censored | 1.46 | first posterior CDF mean | 1.005 | 15,434 / 77,170 |
| Log-logistic, complete | 1.60 | first posterior CDF mean | 1.001 | 11,476 / 57,380 |
| Log-logistic, censored | 1.58 | log-shape/log-scale covariance | 1.001 | 10,886 / 54,430 |

Focused tests also compare exact event densities, right-tail likelihood terms,
and CDFs with SciPy distribution formulas; exercise unit rescaling by
`exp(200)`; verify a prior-only all-zero censored sample; and compare extreme
Gamma and inverse-Gamma log tails with closed forms from the R fixture. The
small-shape/tiny-argument case verifies that an underflowed intermediate does
not turn a nonzero Gamma CDF/survival into zero or one. A Gamma event-density
check at shape `1e12` exercises the existing stable CDFLIB gamma factor near
its mode. These checks support the implemented formulas and sampler behavior;
they do not establish native BCSTTE parity or guarantee MCMC convergence.
