# BaCIS subgroup classification and borrowing

Catalog entry 153 is **partial**. The Python implementation includes both stages
of Bayesian classification and information sharing (BaCIS), described by
[Chen and Lee, Biometrical Journal (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6546564/).
Sources include the [official app](https://biostatistics.mdanderson.org/shinyapps/BaCIS/)
(PID 1072, v1.0.1.0, updated January 6, 2026), its help slides and the authors'
[bacistool 1.0.0 R package](https://cran.r-project.org/package=bacistool).
Hashes are recorded in [bacis-sources.json](bacis-sources.json). Vendor code is
used as a reference and is not distributed with this independent implementation.

```python
from mdanderson_stats import bacis_classify, bacis_fit

responses = [2, 3, 7, 6, 10]
patients = [25, 25, 25, 25, 25]
classification = bacis_classify(responses, patients)
print(classification.high_probability)
print(classification.cluster)  # 1 = low response, 2 = high response

fit = bacis_fit(responses, patients, draws=1000, warmup=500, chains=2, seed=153)
print(fit.posterior_mean)
print(fit.efficacy_probability)
print(fit.summary.split_rhat, fit.summary.batch_mean_mcse)
```

## Classification

For each subgroup, responses are binomial with a logit-normal prior centered on
either `logit(phi_low)` or `logit(phi_high)`. The two clusters have equal prior
probability. Their common precision is `classification_precision`, defaulting to
`(6 / (logit(phi_high) - logit(phi_low)))**2`. The defaults are `phi_low=.1` and
`phi_high=.3`. Precision means inverse variance throughout this API.

`bacis_classify` computes each component's binomial marginal likelihood by
one-dimensional quadrature, centered and scaled at its posterior mode. It uses
log-domain likelihoods and computes both probability tails directly. The latent
variable's precision `tau2` does not need an API parameter: a zero-centered
Gaussian has equally likely signs for every positive precision, and integrating
its sign gives the same two-component model. Classification thus needs no MCMC.

The result's `log_evidence` has shape `(subgroups, 2)` and contains conditional
log marginal likelihoods, including binomial coefficients and excluding the
common cluster weight of one half. `quadrature_error` contains the integrator's
estimated relative errors for those integrals. A failed integral raises an error.
These error estimates are numerical diagnostics, not posterior uncertainty.

There are two conflicting adaptive-cutoff definitions in the sources:

| `adaptive_weighting` | Observed response rate used | Source |
| --- | --- | --- |
| `"subgroup"` (default) | Unweighted mean of subgroup response rates | CRAN R implementation |
| `"patient"` | Total responses divided by total patients | Official help slide 4 |

For either definition, subtract `(phi_low + phi_high)/2` from the observed rate
and apply `expit(-2 * difference / (phi_high - phi_low))` to obtain the cutoff.
Equal subgroup sizes make the definitions coincide. Supplying
`classification_cutoff` directly bypasses the adaptive calculation.

A subgroup enters the high cluster only when its high-cluster posterior
probability is **strictly greater** than the cutoff; equality enters the low
cluster. A sentence in the paper/help reverses those labels. The latent-sign
model, the native code and the stated purpose of the clusters support the rule
implemented here. The Shiny backend was not available for direct comparison.

## Within-cluster borrowing

Classification is fixed before the second stage, as in the native two-step
procedure; classification uncertainty is not averaged over alternative groupings.
For each cluster with two or more subgroups, the model is

```text
responses[i] ~ Binomial(patients[i], p[i])
logit(p[i]) ~ Normal(mu, variance=1/tau)
mu ~ Normal(logit(cluster center), variance=1/mean_precision)
tau ~ Gamma(shape=precision_shape, rate=precision_rate)
```

The low cluster's center is `phi_low`; the high cluster's center is `phi_high`.
Each cluster has its own mean and precision, so it shares no information with
the other cluster during this stage. The existing logistic-normal hierarchy
implementation supplies elliptical-slice group updates and Gibbs hyperparameter
updates. No JAGS, rjags, new dependency or parallel worker is required.

The defaults `mean_precision=.1`, `precision_shape=50`, `precision_rate=10`
match the app's displayed inputs. The CRAN function instead defaults to rate 2;
set `precision_rate=2` for that configuration. A singleton cluster follows the
native special case: its posterior is exactly `Beta(1+responses, 1+nonresponses)`,
with no logistic-normal borrowing.

`posterior_mean`, `posterior_sd`, `efficacy_probability` (`Pr(p > phi_low)`) and
`high_response_probability` (`Pr(p > phi_high)`) use exact Beta calculations for
singletons and retained MCMC draws for larger clusters. `efficacious` compares
the efficacy probability strictly with `efficacy_cutoff`, default .92. The
classification probability and these within-cluster response probabilities are
different quantities.

`probability_samples` has axes `(chain, retained draw, subgroup)`. All `summary`
fields are draw-based, including for singletons: means, medians, standard
deviations, empirical shortest 80% intervals, classical split R-hat and
batch-means MCSE for posterior means. `cluster_fits` retains the two hierarchical
fits in low/high order, with `None` for empty or singleton clusters. Arrays are
read-only. MCMC diagnostics do not guarantee convergence; the Python sampler's
draws and iteration conventions differ from the native JAGS runs.

## Scope and numerical checks

Inputs support 1–100 subgroups, each with 1–10,000 patients. Classification
precision is bounded to `[1e-6, 1e6]`, including its automatically calculated
value. Hierarchical precision hyperparameters must be positive and at most
`1e12`. Sampling permits 2–8 chains, 8–10,000 retained draws per chain and
0–10,000 warmup iterations. The products `chains * draws * subgroups` and
`chains * (draws + warmup) * subgroups` are capped at 500,000 and 1,000,000.
Chains update within one process; no patient-level arrays are constructed.

`tools/reference_bacis.R` uses base R only. Its 19 classification cases include
the native example, unequal sample sizes, diffuse precision and 1,000-patient
all-response/no-response extremes. Both cutoff definitions and small complementary
probabilities are compared independently. Exact Beta references check singleton
summaries; a concentrated-hyperprior limit checks that both borrowing clusters
use their own correct centers against independent one-dimensional integration.

DIC, native effective-sample-size calculations, latent-variable density plots,
native file/report formats and operating-characteristic simulation remain open.
The mathematical references validate the declared model, not native random
streams, convergence for arbitrary priors or complete application parity.
