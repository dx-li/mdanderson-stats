# Wavelet functional mixed models

The [MD Anderson WFMM software](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/70)
fits functional fixed and random effects in a transformed coefficient space.
Python supports orthogonal wavelets and supplied custom transform pairs,
coefficient-specific Bayesian mixed models, reconstruction, posterior curve
summaries and [future-curve prediction](wfmm-prediction.md). An explicit Python REML
initializer supplies starting variance estimates. Catalog entry 70 is
**partial**: native variance/proposal initialization, other transform families and
native file workflows remain open.

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
Five synthetic periodic Haar (`filter_length=2`) transforms match native WFMM
3.1 with `extended_mode=1` at 16–64 points and one–four decomposition levels.
This does not establish other native wavelets or extension modes: native db4
with 64 points/two levels retains 77 coordinates, while the Python orthogonal
transform retains 64. See the [compression audit](../research/wfmm-native-compression-audit.md).

Use `transform="identity"` to retain the original columns. For
`transform="custom"`, provide either a square `custom_matrix` with orthonormal
columns, or separate square `analysis_matrix` and `synthesis_matrix` numerical
inverses. The former keeps transpose reconstruction; the latter uses the
guide's explicit forward and reverse products. See the
[paired custom-transform guide](wfmm-custom-transform.md) for validation and
covariance interpretation. Custom transforms do not center the input. PCA,
alternate boundary modes and multidimensional transforms remain open. Energy
compression of supplied coefficients is now native-verified below.

Transforms return readonly arrays and reject nonfinite values. They support
at most 4,096 time points and 2,000,000 input cells. Work estimates are checked
before filtering or custom matrix operations; `max_work` can lower the default
200,000,000-operation estimate. Wavelet transforms do not allocate a dense
time-by-time basis matrix.

## Fit selected coefficients and restore curves

`wfmm_select_coefficients` retains explicit zero-based coefficient indices or
complete `coefficient_partition` bands. Supply exactly one of `indices` and
`partitions`. Requested indices are sorted into the original packed order;
scale and partition labels keep their original values. For example, in a
two-level transform, partitions `[0, 2]` keep the approximation and finest
detail band, omitting the coarser detail band.

The selection can feed both empirical-Bayes calibration and the coefficient
model. Restore posterior draws to their original positions before summarizing
curves. The omitted coordinates are exactly zero; this is a reduced model,
so summaries do not include uncertainty from omitted coefficients.

```python
import numpy as np
from mdanderson_stats import (
    calibrate_wfmm_shrinkage,
    fit_wfmm_coefficients,
    wfmm_basis,
    wfmm_transform,
    wfmm_select_coefficients,
    wfmm_restore_coefficients,
    wfmm_summarize,
)

basis = wfmm_basis(8, levels=2, filter_length=2)
curves = np.arange(48.0).reshape(6, 8) / 20
x = np.ones((6, 1))
selected = wfmm_select_coefficients(
    wfmm_transform(curves, basis).coefficients,
    basis,
    partitions=[0, 2],
)
calibration = calibrate_wfmm_shrinkage(
    selected.coefficients,
    x,
    random_variance=None,
    residual_variance=0.1,
    coefficient_partition=selected.coefficient_partition,
)
fit = fit_wfmm_coefficients(
    selected.coefficients,
    x,
    prior=calibration.prior,
    residual_variance=0.1,
    estimate_variances=False,
    coefficient_partition=selected.coefficient_partition,
    coefficient_scale=selected.coefficient_scale,
    draws=32,
    warmup=16,
    chains=2,
    rng=np.random.default_rng(21),
)
restored = wfmm_restore_coefficients(fit.coefficients, selected)
summary = wfmm_summarize(restored, basis)
assert restored.shape == (2, 32, 1, 8)
assert np.all(restored[..., [2, 3]] == 0)
```

Selection and restoration preserve arbitrary leading dimensions and bound
input/output arrays to 2,000,000 cells. Restoration also supports coefficient
variance arrays; zero-filled variances describe the retained subspace only.
The native guide records retained indices as `DIndex` and their count as
`Kstar`. Explicit selection remains available; automatic energy compression
now has a separate verified API.

### Compress by the native energy rule

`wfmm_compress_coefficients(coefficients, basis, alpha=..., t=...)` ranks squared
coefficients within each curve. A column receives a vote when cumulative energy
**including that coefficient** is strictly below `alpha` times total energy.
Retain columns with **more than `t` votes**. This native rule excludes the
crossing coefficient, so it can retain less than the requested energy.
`alpha=1` bypasses compression and retains every column, regardless of `t`.

```python
import numpy as np
from mdanderson_stats import wfmm_basis, wfmm_compress_coefficients, wfmm_restore_coefficients

basis = wfmm_basis(4, transform="identity")
compressed = wfmm_compress_coefficients(
    [[3, 1, 1, 1], [1, 3, 1, 1], [3, 1, 1, 1]],
    basis,
    alpha=0.8,
    t=1,
)
assert compressed.selection.retained_indices.tolist() == [0]
assert compressed.vote_counts.tolist() == [2, 1, 0, 0]
np.testing.assert_allclose(compressed.energy_fraction, [0.75, 1 / 12, 0.75])
restored = wfmm_restore_coefficients(compressed.selection.coefficients, compressed.selection)
assert restored.shape == (3, 4)
```

The result's `selection` feeds existing fitting/restoration APIs; it preserves
original scale/partition identities. `energy_fraction` reports actual retained
energy per curve, with zero for a zero-energy curve. `vote_counts` covers all
original columns, or is `None` when alpha=1 bypasses voting. Inputs and outputs
remain bounded at 2,000,000 cells; sorting uses one row of temporaries.

Native C++ sorting does not guarantee stable ties. Python rejects a threshold
that splits equal-energy coefficients by default. `tie_policy="original_index"`
explicitly chooses smaller indices first, without promising that ordering for
all native inputs. An empty selection raises a clear error. Numerical rescaling
handles coefficients as small/large as 1e±300; exact floating-point threshold
parity is not claimed. Native high/low-pass flags, other extended transforms
and their file representations remain open. See the
[native compression audit](../research/wfmm-native-compression-audit.md).

## Initialize variance components from the data

`initialize_wfmm_variances` estimates random-effect and residual variances
separately for each transformed coefficient under the Gaussian mixed model.
It is an optional Python policy. The recovered native MOM/profile initializer
is distinct; its inverse-gamma prior mapping is verified in the next section.

```python
import numpy as np
from mdanderson_stats import initialize_wfmm_variances

x = np.ones((8, 1))
z = np.eye(4)[[0, 0, 1, 1, 2, 2, 3, 3]]
first_coefficient = np.array([-1.7, -1.3, -0.7, -0.3, 0.3, 0.7, 1.3, 1.7])
coefficients = np.column_stack((first_coefficient, 3 + first_coefficient / 2))
initial = initialize_wfmm_variances(coefficients, x, z)
print(initial.raw_random_variance)
print(initial.raw_residual_variance)
print(initial.optimizer_success, initial.status)
```

The model is `V = sum_g(q_g * Z_g @ Z_g.T) + diag(s[residual_strata])`,
with the same group labels as `fit_wfmm_coefficients`. A fixed-effects-only
model with one residual variance uses the direct estimate `RSS / (N-P)`.
Other models optimize a bounded restricted likelihood using Cholesky solves,
processing coefficients one at a time. Rank-deficient fixed designs,
nonpositive residual degrees of freedom, and covariance components that cannot
be distinguished in the residual space are rejected.

Use `initial.random_variance` and `initial.residual_variance` as the corresponding
inputs to `calibrate_wfmm_shrinkage` or `fit_wfmm_coefficients`. For a model
without random effects, omit `random_variance` when calling those functions.
Inverse-gamma prior shapes/scales and sampling proposal SDs remain separate
inputs. Conditioning a fit on these estimates omits variance-estimation
uncertainty; sampling the variances requires explicit priors and proposals.

Inspect `optimizer_success`, `status`, `optimizer_message`, and the lower/upper
bound flags before using an initialization. `starts_usable` means positive
starting values are available; it does not imply optimizer convergence or
adequate model fit. Finite optimization-bound estimates are retained as
estimated, rather than silently converted to zero.

`positive_floor_ratio=1e-8` supplies the minimum starting variance as that
ratio times the squared maximum absolute observation in each coefficient.
This floor changes only sampler starts: the raw estimates and floor flags
remain available. Residuals at linear-algebra roundoff scale receive an
explicit numerical-boundary status and retain their normalized residual norm.
All-zero coefficients need a caller-supplied positive `zero_data_floor` in
variance units; otherwise their starts remain zero and `starts_usable` is false.
These rules do not define a Bayesian prior.

Iteration, likelihood-evaluation, design-size and projected covariance-rank
limits are checked explicitly. The initializer may reject a large model that
is otherwise accepted by the sampler. Independent residual-only and balanced
random-intercept references are recorded in the
[initialization audit](../research/wfmm-variance-init-audit.md).

## Fit functional fixed and random effects

The following small example illustrates the complete explicit-prior workflow.
Its short chains demonstrate the interface; they are not a convergence study.

```python
import numpy as np
from mdanderson_stats import (
    WFMMPrior,
    fit_wfmm_coefficients,
    wfmm_basis,
    wfmm_transform,
    wfmm_summarize,
)

grid = np.linspace(0, 1, 8, endpoint=False)
x = np.column_stack((np.ones(6), [-1, 1, -1, 1, -1, 1]))
z = np.eye(3)[[0, 0, 1, 1, 2, 2]]
curves = (
    np.sin(2 * np.pi * grid)[None, :]
    + 0.3 * x[:, 1, None] * np.cos(2 * np.pi * grid)[None, :]
    + (z @ np.array([-0.1, 0.1, 0]))[:, None]
    + 0.2 * np.random.default_rng(7).normal(size=(6, 8))
)
basis = wfmm_basis(8, levels=2)
transformed = wfmm_transform(curves, basis)
prior = WFMMPrior(
    inclusion_probability=0.5,
    slab_variance=1,
    random_shape=3,
    random_scale=0.2,
    residual_shape=3,
    residual_scale=0.2,
)
fit = fit_wfmm_coefficients(
    transformed.coefficients,
    x,
    z,
    prior=prior,
    random_variance=0.1,
    residual_variance=0.1,
    proposal_sd=(0.05, 0.05),
    coefficient_partition=transformed.coefficient_partition,
    coefficient_scale=transformed.coefficient_scale,
    draws=64,
    warmup=64,
    chains=2,
    sample_random_effects=True,
    rng=np.random.default_rng(2026),
)
summary = wfmm_summarize(
    fit.coefficients,
    basis,
    effect_contrast=[[0, 1]],
    effect_sizes=[0, 0.2],
    confidence=0.95,
)
print(summary.mean)  # second fixed-effect curve
print(summary.simultaneous_lower)  # band over all eight grid points
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
Zero slab variance is accepted only with zero inclusion probability, an
all-spike prior that fixes the coefficient at zero.

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

## Calibrate shrinkage from the observed coefficients

`calibrate_wfmm_shrinkage` estimates mixture inclusion probabilities and slab
variances, conditional on supplied random-effect and residual variance
estimates. It pools coefficients within each fixed-effect/partition group.
For the example above:

```python
from dataclasses import replace
from mdanderson_stats import calibrate_wfmm_shrinkage

calibration = calibrate_wfmm_shrinkage(
    transformed.coefficients,
    x,
    z,
    random_variance=0.1,
    residual_variance=0.1,
    coefficient_partition=transformed.coefficient_partition,
)
print(calibration.group_converged)
print(calibration.group_inclusion_probability)
calibrated_prior = replace(
    calibration.prior,
    random_shape=3,
    random_scale=0.2,
    residual_shape=3,
    residual_scale=0.2,
)
```

Pass `calibrated_prior` as the `prior` argument to `fit_wfmm_coefficients`.
The returned prior contains only the calibrated fixed-effect mixture; the
example adds explicit variance priors for subsequent variance sampling.
For fixed variance components, `calibration.prior` is directly usable with
`estimate_variances=False`.

Calibration uses joint generalized least-squares coefficient estimates and
the source's conditional sampling variances `1/(X_i' Sigma^-1 X_i)`. The
latter differ from the diagonal of the joint GLS covariance matrix.
The EM update returns a local solution; inspect `group_converged`, iteration
counts and `group_log_marginal_likelihood`, and compare initial values when
needed. The reported objective is the log likelihood relative to the
standard-normal sampling model. If slab-to-sampling variance reaches zero,
the mixing proportion is unidentified; the API returns the equivalent
all-spike prior with zero inclusion probability and zero slab variance.
Rank-deficient designs and diagonal-normalized GLS information with condition
number greater than `1e12` are rejected to avoid unreliable coefficient estimates.

This step does not estimate random/residual variances, native inverse-gamma
hyperparameters or proposal SDs, and it does not propagate uncertainty in
the supplied variance estimates or estimated shrinkage hyperparameters.
Partition labels must be nonnegative integers; no native coarse-scale
exception, `minT` floor or `bigT` constant is silently applied.

## Verified native variance-prior mapping

`wfmm_variance_prior` combines supplied fixed-effect mixture parameters with
WFMM 3.1's inverse-gamma mapping. Supply positive two-dimensional variance
matrices, with rows for random-effect levels or residual strata and columns
for coefficients. `random_effect_counts` gives the number of columns of Z at
each level; `residual_stratum_sizes` gives the number of curves in each stratum.
The mapping is `a = delta_omega * count`, `b = a * variance`, with density
proportional to `v**(-a-1) * exp(-b/v)`. Wavelet partition size does not enter.
The native default `delta_omega` is `1e-4`.

```python
from mdanderson_stats import wfmm_variance_prior

prior = wfmm_variance_prior(
    inclusion_probability=0.5,
    slab_variance=10.0,
    random_variance=[[0.2, 0.3]],
    random_effect_counts=[4],
    residual_variance=[[1.0, 1.5]],
    residual_stratum_sizes=[8],
    delta_omega=0.1,
)
```

Pass this prior to `fit_wfmm_coefficients`, with explicit starting variances
and proposal SDs. To use empirical-Bayes fixed effects, pass the calibration
prior's `inclusion_probability` and `slab_variance` to this helper. The mapping
centers **precision**: `E[1/v] = 1/variance`. The mean of v is infinite when
`a <= 1`; the supplied variance is neither its mean nor its mode.

Eight wholly synthetic original-Linux-program initializations verify this
mapping, including two random-effect levels, unequal residual strata and
wavelet partitions. The helper does not run the native MOM/profile optimizer;
supplying Python REML estimates is an explicit alternative centering choice.
See the [native-prior audit](../research/wfmm-native-prior-audit.md).

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

## Covariance and variance functions

`wfmm_summarize_covariance` reconstructs data-time variance functions from
coefficient-variance posterior draws. It accepts either `fit.random_variances`
or `fit.residual_variances`, with shape `(chain,draw,component,K)`:

```python
from mdanderson_stats import wfmm_summarize_covariance

residual_covariance = wfmm_summarize_covariance(
    fit.residual_variances,
    basis,
    include_covariance=True,
)
print(residual_covariance.variance_function_mean[0])
print(residual_covariance.correlation_from_mean_variance[0])
```

Each component corresponds to one supplied random-effect level or residual
stratum. With synthesis matrix `S` in `Y=D*S`, its data-space covariance is
`S.T*diag(omega)*S`. For an orthogonal analysis matrix `W`, `S=W.T`, giving
the familiar `W*diag(omega)*W.T`. Supplied non-orthogonal transform pairs use
their own synthesis matrix, and do not preserve the meaning of an unchanged
coefficient-space prior across bases.
The diagonal-only default computes each draw's variance function without
materializing a time-by-time matrix for every draw. Results include mean,
sample SD and linearly interpolated quantiles in both coefficient space and
data space. Mean/SD arrays have shape `(component,K)` or `(component,time)`;
quantile arrays add a leading quantile-probability axis. These summarize
**variances**, not standard-deviation functions.

`include_covariance=True` adds `(component,time,time)` posterior mean covariance
and its plug-in correlation. Mean covariance equals the reconstruction of
mean coefficient variances by linearity. The plug-in correlation generally
differs from the posterior mean of correlations, because normalization is
nonlinear. A zero covariance diagonal makes correlation undefined and raises
an error; diagonal-only summaries remain available for zero variances.

`retain_covariance_draws=True` additionally returns
`(chain,draw,component,time,time)` covariance draws and requires
`include_covariance=True`. Full covariance products, working arrays and
transform operations are checked before expansion. For a single supplied
variance vector or a batch ending in `K`, `wfmm_covariance(variances,basis)`
performs the same reconstruction directly and returns leading dimensions
followed by `(time,time)`.

The native guide calls the plug-in correlation `rho` and the variance function
`sigma`; the latter is explicitly a variance despite its name. Python returns
descriptive fields and does not reproduce native output-file formats.

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

Native variance initialization and
proposal selection, additional transforms and
boundary rules, pass filtering and native prediction/file
formats remain open. Energy compression of explicit coefficient matrices is
now verified against the original program. The [source and implementation audit](../research/wfmm-audit.md)
tracks these gaps and the independent references.
