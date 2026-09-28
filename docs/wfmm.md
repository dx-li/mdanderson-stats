# Wavelet functional mixed models

The [MD Anderson WFMM software](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/70)
fits functional fixed and random effects in a transformed coefficient space.
Python supports orthogonal transforms, coefficient-specific Bayesian mixed
models, reconstruction and posterior curve summaries. Catalog entry 70 is
**partial**: automatic initialization, other transform families and native
file workflows remain open.

## Transform and reconstruct curves

```python
import numpy as np
from mdanderson_stats import wfmm_basis, wfmm_transform, wfmm_inverse

curves = np.arange(24.0).reshape(2, 12)
basis = wfmm_basis(12, filter_length=8)
transformed = wfmm_transform(curves, basis)
reconstructed = wfmm_inverse(transformed.coefficients, basis)
```

Each row is a curve on an equally spaced grid. The default is a periodic
Daubechies db4 transform; even filter lengths 2–20 select db1–db10.
The default decomposition level is the largest `J` for which `2**J` divides
the curve length. Supply `levels` to use fewer levels. Odd lengths require
an identity or custom transform.

The coefficient order is `[a_J,d_J,...,d_1]`: coarsest approximation followed
by details from coarse to fine. Filtering samples even indices with forward
tap offsets and periodic wrapping. `coefficient_partition` separates the
approximation and each detail band; `coefficient_scale` records its level.
These are explicit Python conventions, not a claim to reproduce native WFMM
binary ordering or boundary-extension modes.

Use `transform="identity"` to retain the original columns. For
`transform="custom"`, provide a square `custom_matrix` with orthonormal
columns: the forward calculation is `curves @ custom_matrix`, and inversion
uses its transpose. Custom transforms do not center the input. PCA, energy
compression, alternate boundary modes and multidimensional transforms remain
open.

Transforms return readonly arrays and reject nonfinite values. They support
at most 4,096 time points and 2,000,000 input cells. Work estimates are checked
before filtering or custom matrix operations; `max_work` can lower the default
200,000,000-operation estimate. Wavelet transforms do not allocate a dense
time-by-time basis matrix.

## Fit functional fixed and random effects

The following small example illustrates the complete explicit-prior workflow.
Its short chains demonstrate the interface; they are not a convergence study.

```python
import numpy as np
from mdanderson_stats import (
    WFMMPrior, fit_wfmm_coefficients, wfmm_basis, wfmm_transform, wfmm_summarize,
)

grid = np.linspace(0, 1, 8, endpoint=False)
x = np.column_stack((np.ones(6), [-1, 1, -1, 1, -1, 1]))
z = np.eye(3)[[0, 0, 1, 1, 2, 2]]
curves = (
    np.sin(2 * np.pi * grid)[None, :]
    + .3 * x[:, 1, None] * np.cos(2 * np.pi * grid)[None, :]
    + (z @ np.array([-.1, .1, 0]))[:, None]
    + .2 * np.random.default_rng(7).normal(size=(6, 8))
)
basis = wfmm_basis(8, levels=2)
transformed = wfmm_transform(curves, basis)
prior = WFMMPrior(
    inclusion_probability=.5, slab_variance=1,
    random_shape=3, random_scale=.2,
    residual_shape=3, residual_scale=.2,
)
fit = fit_wfmm_coefficients(
    transformed.coefficients, x, z, prior=prior,
    random_variance=.1, residual_variance=.1,
    proposal_sd=(.05, .05),
    coefficient_partition=transformed.coefficient_partition,
    coefficient_scale=transformed.coefficient_scale,
    draws=64, warmup=64, chains=2, sample_random_effects=True,
    rng=np.random.default_rng(2026),
)
summary = wfmm_summarize(
    fit.coefficients, basis,
    effect_contrast=[[0, 1]], effect_sizes=[0, .2], confidence=.95,
)
print(summary.mean)                 # second fixed-effect curve
print(summary.simultaneous_lower)   # band over all eight grid points
print(summary.simultaneous_upper)
```

For `N` curves and `K` coefficients, `fixed_design` has shape `(N,P)` and
optional `random_design` has shape `(N,M)`. The coefficient model is
`D = X*beta + Z*U + error`. `random_strata` groups the columns of `Z` by a
shared variance parameter; `residual_strata` groups rows of `D` by their
residual variance. Labels are contiguous nonnegative integers starting at
zero. Both kinds of variance vary separately across coefficients, allowing
nonstationary covariance after inversion. Random effects and residuals use
identity between-curve correlation within these groups.

Each fixed coefficient has a mixture of a point mass at zero and a centered
normal distribution. `slab_variance` is a variance, not an SD. Its prior and
`inclusion_probability` may be scalars or `(P,K)` arrays; probabilities zero
and one are supported exactly. Basis scale/partition labels are retained as
metadata; they do not silently select or pool prior parameters.

Variance priors have density proportional to
`v**(-shape-1) * exp(-scale/v)`. Supply positive shape and scale values for
every estimated random and residual component. Component arrays are scalars
or `(groups,K)` arrays; a length-`K` vector is also accepted for a single
group. Starting variances and positive proposal SDs are explicit. Proposal
SDs are ordered `(random,residual)`; for a fixed-effects-only model, use
`proposal_sd=(None, residual_sd)`.

The sampler integrates out `U` while updating fixed effects and variances,
then optionally draws `U` from its Gaussian conditional. It uses sequential
chains, Cholesky solves and positive-truncated normal variance proposals
with the Hastings correction. Set `estimate_variances=False` to condition
on the supplied variances; in that mode, omit `proposal_sd` and variance
priors. Conditioning omits variance-estimation uncertainty.

Posterior arrays start with `(chain,draw)`: fixed coefficients and inclusion
indicators end with `(P,K)`, variance draws with `(groups,K)`, and optional
random effects with `(M,K)`. Results include variance acceptance rates and
the marginal Gaussian log likelihood conditional on each fixed-effect draw.
`fit.summary` contains fixed-coefficient summaries and classical split R-hat
and batch-means Monte Carlo errors. Use `summarize_chains` separately on
variance draws when assessing their mixing. These diagnostics are estimates,
not convergence guarantees.

## Reconstructed posterior inference

`wfmm_summarize` accepts `(chain,draw,effect,K)` coefficients and reconstructs
every draw before computing summaries. It therefore preserves uncertainty
across the wavelet coefficients. It also accepts `fit.random_effects` when
those draws were requested.

An optional `(contrasts,effects)` `effect_contrast` and `(time,regions)`
`time_contrast` apply `L @ reconstructed_draw @ LT`. Summaries include
means, sample SDs, linear-interpolated pointwise quantiles, and
`Pr(abs(effect) > threshold)` for every supplied effect size. Quantiles and
effect probabilities have leading probability/threshold axes. The sign-tail
score is `2*min(Pr(effect>0),Pr(effect<0))`, using strict inequalities; it is
not a posterior probability of the null, and all-zero draws give a score
of zero.

Simultaneous bands use each draw's largest absolute standardized deviation
across the output grid, separately for each effect contrast. They use the
requested confidence quantile of those maxima. `simbas_probability` is the
fraction of maxima at least as large as the absolute standardized posterior
mean at a coordinate. Constant-zero coordinates have score one; constant
nonzero coordinates have score zero. Quantile interpolation is explicit
Python behavior; native output quantile parity has not been verified.

By default, summaries discard reconstructed draws after computing the
results. `retain_curves=True` retains a readonly `(chain,draw,contrast,region)`
array. Reconstruction uses bounded chunks, checks intermediate contrast
dimensions before allocating them and evaluates effect thresholds one at a
time.

## Numerical scope and remaining work

Fitting supports up to 500 curves, 512 coefficients, 100 fixed effects and
500 random effects, subject to tighter combined storage/work checks. It
requires 2–4 chains, 8–5,000 draws and 0–5,000 warmup iterations. Retained
posterior buffers are limited to 2,000,000 cells; factorization-work estimates
are limited to 250,000,000. These joint bounds can reject a requested fit even
when its individual dimensions are valid. Posterior reconstruction and
contrasts have separate 2,000,000-cell and 200,000,000-work limits.

Validation includes hand-computed Haar transforms, scalar db4 calculations,
reconstruction/energy identities, independent exact Gaussian-mixture posterior
references, a conjugate inverse-gamma variance posterior and hand-calculated
contrast/band summaries. Native MCMC random-number parity is not claimed.

Empirical-Bayes shrinkage calibration, automatic variance initialization and
proposal selection, native inverse-gamma defaults, additional transforms and
boundary rules, compression, prediction/covariance workflows and native file
formats remain open. The [source and implementation audit](../research/wfmm-audit.md)
tracks these gaps and the independent references.
