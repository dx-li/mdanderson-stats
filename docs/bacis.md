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

## Latent classification posterior

`bacis_theta_posterior` evaluates the posterior density, CDF and upper-tail
probability of the first-stage latent variable on an explicit grid.
`sample_bacis_theta` draws independent samples from that same distribution:

```python
import numpy as np
from mdanderson_stats import bacis_theta_posterior, sample_bacis_theta

theta = bacis_theta_posterior(classification, [-np.inf, -30, 0, 30, np.inf])
assert np.allclose(theta.survival[2], classification.high_probability)
samples = sample_bacis_theta(classification, draws=1000, rng=np.random.default_rng(153))
assert samples.shape == (1000, len(responses))
```

The native classification model depends only on theta's sign. Its magnitude
therefore retains the prior half-normal distribution on each side of zero,
with the classifier's low/high probabilities as the two posterior weights.
`latent_precision=.001` is the native default inverse variance. Changing it
rescales theta but leaves the classification probabilities unchanged. This
precision must be positive and yield a representable normal scale and variance.

Density/CDF/survival arrays have `(grid points, subgroups)` axes; `mean` and
`variance` are analytical posterior moments. Infinite grid endpoints have the
usual distribution limits. At zero the density uses its right-hand value,
consistent with the native step function; the CDF is continuous and there is
no point mass. Both tails are evaluated directly to preserve small probabilities.

The returned arrays are readonly. Grid evaluation permits at most 200,000
group-grid cells, and sampling at most 100,000 draws and 500,000 total cells.
These independent draws have the native latent posterior as their target;
they do not reproduce a JAGS random stream or its finite-sample smoothed plot.
The [posterior audit](../research/bacis-theta-audit.md) gives the reduction and
independent R references.

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

## Equivalent sample size

`bacis_equivalent_sample_size` summarizes the information in retained response
probabilities using the native software's variance-matching rule:

```python
from mdanderson_stats import bacis_equivalent_sample_size

ess = bacis_equivalent_sample_size(fit.probability_samples, responses, patients)
print(ess.equivalent_sample_size)
print(ess.candidate_roots)
print(ess.relative_variance_residual)
```

For each subgroup, it pools the chain and draw axes and calculates the unbiased
sample variance `v`. It then solves

```text
Var[Beta(y + 1, N - y + 1)] = v,  with N >= y,
```

where `y` is the observed number of responses. When two admissible roots exist,
the selected root minimizes `abs(y/N - y/n_observed)`; exact ties select the
smaller root. All candidate roots and their rate discrepancies are retained so
the choice can be inspected. `N=0` is allowed for zero responses, with the
corresponding rate defined by its zero limit.

This is a total equivalent patient count. It is different from an MCMC
effective sample size or a number of additional prior patients. The paper's
description of matching both beta moments is less specific than the software;
this helper implements the software's fixed-response-count variance match.

The original code can select a fabricated `.0001` root for a zero-response
subgroup. The Python helper retains only admissible solutions: for the variance
of `Beta(1, 26)`, it correctly returns 25. A zero sample variance or a variance
with no admissible solution raises an error. Computation uses monotone branches
of the variance function and a logarithmic tail coordinate to avoid unstable
unscaled cubic roots. See the [source audit](../research/bacis-ess-audit.md).
The supplied array must have `(chains, draws, subgroups)` axes, at least two
samples in total, and at most 500,000 cells. Eight base-R reference cases cover
ordinary and large equivalent counts, two admissible roots, all responses and
the documented zero-response defect.

## Classification-model comparison

`bacis_classification_dic` evaluates the first-stage two-component
classification model with the DIC definition used by the native software:

```python
from mdanderson_stats import bacis_classification_dic

score = bacis_classification_dic(responses, patients)
print(score.mean_deviance)
print(score.penalty)
print(score.total_dic)
```

It retains uncertainty about both mixture components and integrates their
posterior expectations directly. The penalty for subgroup `i` is
`patients[i] * Cov(p[i], logit(p[i]))`, including covariance between components.
Mean deviance includes the full binomial likelihood constant; `dic` contains
the per-subgroup sums and `total_dic` their total. The result also reports
posterior response/logit means, component probabilities and integration-error
estimates. The default component centers and precision match `bacis_classify`.

The source's score concerns classification, so it depends on its component
centers and precision. The adaptive cutoff and subsequent within-cluster
borrowing priors do not enter this score. Its deterministic integrals target
the population expectations of the native JAGS monitor, avoiding the sampling
variation of that monitor's finite chains.

The native wrapper also supplies the same random-generator initialization to
all five DIC chains. That can undermine their independence. The source audit
records this limitation; Python evaluates the independent-draw population
definition directly rather than reproducing that initialization behavior.

DIC is an asymptotic model-comparison approximation. A separated or multimodal
posterior can violate the assumptions behind its interpretation; a successfully
computed score alone does not establish that it is suitable for selecting a
model. Integration diagnostics describe numerical accuracy, not those modeling
assumptions. The [source audit](../research/bacis-dic-audit.md) records the exact
native definition and independent base-R reference calculations.

## Complete one-trial summary

`bacis_one_trial` fits both stages once, computes the variance-matched ESS from
the retained draws, and assembles the native one-trial table:

```python
from mdanderson_stats import bacis_one_trial

trial = bacis_one_trial([2], [25], draws=256, warmup=0, chains=2, seed=153)
for label, values in zip(trial.row_labels, trial.report_values, strict=True):
    print(label, values)
assert trial.values.shape == (10, 1)
assert trial.values[5, 0] == 3 / 27  # exact singleton posterior mean
```

Rows contain the two response-tail probabilities, latent high-cluster
probability, high-cluster and efficacy indicators, posterior and observed
response rates, response and patient counts, and equivalent sample size.
`values` retains full precision; `report_values` is rounded to three decimal
places for display. Both decisions use strict comparisons before rounding.
The readonly result also retains the original observations, `fit` and
`equivalent_sample_size`, including their diagnostics.

This wrapper defaults to `precision_rate=2`, matching the CRAN one-trial
function; `bacis_fit` defaults to the app's displayed value of 10. The sampler
defaults remain the package's bounded 2,000 retained draws, 1,000 warmup
iterations and two sequential chains. Supply priors and sampling settings
explicitly when comparing analyses. Singleton response summaries and latent
classification probabilities use the package's exact calculations rather than
the native finite-chain estimates. ESS uses the retained response draws.
The one-trial table does not implicitly calculate DIC or sample latent theta.

An undefined variance match raises the same error as
`bacis_equivalent_sample_size`; the report does not fabricate an ESS to fill
that row. This is a numerical result table, without native file-format or
random-stream equivalence. The [workflow audit](../research/bacis-trial-audit.md)
records the source row order and checks.

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

Latent-variable density plots, native file formats and
operating-characteristic simulation remain open.
The mathematical references validate the declared model, not native random
streams, convergence for arbitrary priors or complete application parity.
