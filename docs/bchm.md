# BCHM subgroup clustering and borrowing

Catalog entry 158 is **partial**. BCHM first estimates subgroup similarities
using Gaussian clustering of observed response rates, then fits a separate
similarity-weighted logistic-normal hierarchy for each target subgroup.
The method is described by
[Chen and Lee (2020)](https://pubmed.ncbi.nlm.nih.gov/32178585/).
Implementation evidence comes from the
[official application](https://biostatistics.mdanderson.org/shinyapps/BCHM/)
(PID 1099, v1.0.1.0, updated January 6, 2026), its help slides and the authors'
[BCHM 1.00 R package](https://cran.r-project.org/package=BCHM).
[Source hashes](bchm-sources.json) identify the inspected versions. No vendor
code, JAGS installation or additional Python dependency is included.

```python
from mdanderson_stats import bchm_cluster, bchm_fit, bchm_borrow

responses = [1, 2, 3, 7, 8]
patients = [15, 18, 10, 15, 20]
cluster = bchm_cluster(responses, patients, iterations=500, burn_in=200, seed=1234)
print(cluster.result.representative)
print(cluster.result.raw_similarity)

fit = bchm_fit(responses, patients, iterations=500, burn_in=200, draws=2000, warmup=500, seed=1234)
print(fit.posterior_mean, fit.raw_probability, fit.native_probability, fit.decision)
print([s.split_rhat for s in fit.summaries])

# Reuse a similarity estimate for one target (zero-based index).
target_fit = bchm_borrow(
    responses, patients, cluster.result.similarity, target=2, draws=1000, warmup=500, seed=158
)
print(target_fit.probability, target_fit.summary.batch_mean_mcse)
```

## Clustering and similarity

For subgroup `j`, let `x[j] = responses[j] / patients[j]` and
`w[j] = patients[j]`. Clusters have Gaussian base mean `mu`, base variance
`sigma02` and observation variance `sigmaD2`. The native algorithm treats each
subgroup rate as a Gaussian observation with likelihood raised to `w[j]`.
It also counts **patients**, rather than subgroups, in existing-cluster allocation
weights. This preserves the released algorithm; with unequal subgroup sizes it
should not be interpreted as an ordinary exchangeable CRP over subgroups.

Each sweep removes a subgroup from its current cluster before sampling its new
allocation. Given a remaining cluster's patient count `W` and weighted rate sum
`S`, its Gaussian posterior has variance `v = 1/(1/sigma02 + W/sigmaD2)` and
mean `m = v*(mu/sigma02 + S/sigmaD2)`. The log likelihood increment for an
incoming subgroup `(x,w)` is computed directly:

```text
-.5 * log1p(w*v/sigmaD2) - .5 * w*(x-m)**2/(sigmaD2+w*v)
```

This avoids subtracting two large integrated log likelihoods. Existing-cluster
weights multiply the increment's exponential by `W`; a new cluster uses the
base Gaussian and weight `alpha`. Normalization happens in log space. Empty
clusters disappear; there is no finite stick-breaking approximation.

The raw similarity matrix is the fraction of retained allocations in which
each pair of subgroups shares a cluster. The reported matrix floors that
fraction at `d0`; borrowing applies the additional native floor of `0.001`.
These are distinct outputs. The representative partition maximizes the mean
silhouette score across sampled partitions, using distances between observed
rates. It is not a posterior-MAP partition. Singletons have silhouette zero;
one-cluster and all-singleton partitions receive the native score `-0.1`.
Ties retain the first sampled partition. Allocation labels are one-based.

## Target-specific borrowing

For each target subgroup `i`, all subgroups enter this model with the corresponding
row of the borrowing matrix, `m[j] = C[i,j]`:

```text
responses[j] ~ Binomial(patients[j], p[j])
logit(p[j]) = eta[j]
eta[j] ~ Normal(mu1, variance=1/(tau1*m[j]))
mu1 ~ Normal(mu0, variance=1/tau2)
tau1 ~ Gamma(shape=alpha1, rate=beta1)
mu0 = logit(mean(responses / patients))
```

The empirical prior center uses an **unweighted subgroup mean**. When all
subgroups have zero responses, or all have only responses, this logit is
undefined; borrowing rejects those cases unless a finite `prior_mean` is
explicitly supplied to `bchm_borrow`. Clustering itself permits both extremes.
The override is a declared model choice, not native automatic smoothing.

Gaussian-prior elliptical-slice updates sample each subgroup logit. The shared
mean's conditional precision is `tau2 + tau1*sum(m)`; the Gamma conditional
shape is `alpha1 + J/2` and rate is
`beta1 + .5*sum(m*(eta-mu1)**2)`. Only the target subgroup's probability draws
are retained for each target-specific fit. The estimated similarity matrix is
fixed in this second stage, as in the native two-step procedure.

Efficacy means `Pr(p[i] > phi1 + deltaT)`. The result exposes both the raw
Monte Carlo estimate and the native three-decimal probability used for the
strict comparison with `thetaT`. This rounding can affect decisions near the
threshold. Native R starts four JAGS chains but extracts the first; Python uses
all requested chains and reports classical split R-hat, empirical shortest 80%
intervals and batch-means MCSE for posterior means. These diagnostics do not
prove convergence. Neither random streams nor draw counts match JAGS.

The rounding rule is checked against R 4.4.1: select the nearest floating-point
thousandth, with equal distances going to the even thousandth. Python's built-in
`round` and NumPy's scaled rounding each disagree with R for some decimal-looking
ties. Efficacy exceedances are counted on the logit scale, so probabilities
rounding to zero or one do not corrupt threshold-endpoint decisions.

`fit.borrowing[i].samples` has shape `(chains, draws)` and contains only target
`i` from that target-specific model. The accompanying `summary` has one-element
mean, standard-deviation, R-hat and MCSE arrays. `fit.raw_similarity`,
`fit.similarity` and `fit.borrowing_similarity` expose the three matrix stages.
Arrays are read-only; no full collection of all targets' latent states is kept.

## App and CRAN configurations

Python's statistical defaults follow the current app/help example. The CRAN
function's defaults differ; set the parameters explicitly to reproduce that
configuration. Variance and precision parameterizations must not be interchanged.

| Parameter | App / Python | CRAN BCHM 1.00 |
| --- | --- | --- |
| `mu` | .2 | .2 |
| `sigma02` | 20 | 10 |
| `sigmaD2` | .01 | .001 |
| `alpha` | .001 | 1e-60 |
| `d0` | 0 | .05 |
| `alpha1`, `beta1` | 30, 6 | 50, 10 |
| `tau2` | .1 | .1 |
| `phi1`, `deltaT` | .2, .15 | .1, .05 |
| `thetaT` | .5 | .6 |

Python uses smaller, bounded sampling defaults. The app requests 10,000
clustering burn-in sweeps, 20,000 retained clustering sweeps and 10,000 JAGS
iterations; CRAN uses 20,000 JAGS iterations. Those settings are not a
convergence guarantee or an instruction to allocate large local jobs.

## Resource limits

Data contain 1–20 subgroups, each with 1–10,000 patients and integer responses.
Clustering variances lie in `[1e-8, 1e8]`, concentration in `[1e-300, 1e100]`
and the absolute Gaussian prior mean is at most `1e4`. Borrowing hyperparameters
are positive and at most `1e12`; an explicit logit prior mean has absolute value
at most `1e4`.

With `J` subgroups, clustering limits retained `iterations*J` to 200,000 and
work `(burn_in+iterations)*J**2` to 2,000,000. Co-clustering is accumulated one
sweep at a time; allocation storage uses 16-bit integers. Borrowing permits
2–4 chains, 8–10,000 draws and 0–10,000 warmup iterations. Its work limit is
`chains*(draws+warmup)*J` for one target and
`chains*(draws+warmup)*J**2` for the full fit, capped at 1,000,000. Retained
target draws are capped at 200,000. Large subgroup counts may require reducing
the default sampling settings to satisfy these explicit limits.

All chains and target fits run sequentially in one process. Matrix allocation
and grouped sums use NumPy, with no patient-level replication, parallel workers
or large three-dimensional co-clustering tensor. Invalid configuration is
rejected before sampling; non-finite sampler states raise errors.

## Validation and remaining scope

`tools/reference_bchm.R` uses base R to evaluate native Gaussian allocation
probabilities, including unequal patient weights and very small new-cluster
probabilities. Exact enumeration of the five partitions of three equally sized
subgroups provides a stationary co-clustering reference. Independent nested
integration checks the borrowing model in a concentrated-Gamma precision limit
while retaining a random shared mean, so off-diagonal similarity weights still
affect the answer.

The complete two-stage fit is also compared with the official help's five-subgroup
example, allowing Monte Carlo error in both reported results. Small regressions
cover silhouette singletons, resource preflight, boundary empirical means and
probability rounding. These checks leave the existing CI configuration unchanged.

Native plot/report formats, interactive file workflows and direct end-to-end
Shiny/JAGS output parity remain open. The
mathematical checks validate the declared model and allocation algorithm, not
convergence for arbitrary inputs or complete application parity.
