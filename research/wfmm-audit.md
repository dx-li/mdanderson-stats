# WFMM implementation handoff

Entry 70 is partial: the explicit-prior coefficient model, orthogonal transforms
and reconstructed posterior summaries are available. Primary mathematical source:
[Morris and Carroll (2006)](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/WFMM/Morris%26Carroll2006.pdf),
Sections 4–5 and Appendix A. An orthogonal wavelet transform yields mixed
models with location/scale-specific random-effect and residual variances and
zero/normal mixture priors for fixed coefficients. Sampling marginalizes random
effects, uses conditional mixture-normal fixed-effect updates and positive
truncated-normal Metropolis variance proposals. Optional random effects have
Gaussian conditionals; inverse transforms recover functional estimates.

For fixed effect `i`, use conditional precision `X_i' Sigma^-1 X_i`, variance
`V` equal to its inverse, and the residual subtracting other fixed effects.
This differs from the diagonal of the joint inverse information matrix.
Equations 9–13 give, with `z=beta_hat/sqrt(V)` and `U=tau/V`,

```text
O = pi/(1-pi) * (1+U)^(-1/2) * exp(0.5*z^2*U/(1+U))
G = O/(1+O)
U_new = max(0, sum(G*z^2)/sum(G) - 1)
pi_new = mean(G)
tau = V*U
```

EB updates pool locations within each fixed-effect/scale group. Conditional
normal posterior mean and variance are `beta_hat*U/(1+U)` and `V*U/(1+U)`.
The inverse factor in equation 11 is lost in extracted text; Appendix A,
equation 20, confirms the expression above. Implement boundary limits and
log-domain odds explicitly.

## Native scope and gaps

The [official guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/WFMM/wfmm_Users_Guide.pdf)
documents data matrices, fixed/random-effect designs, residual strata,
compression, wavelet and other transforms, and posterior summaries. It also
specifies an inverse-gamma variance prior controlled by `delta_omega`, but the
exact native shape/rate mapping is not established by the current audit.
Explicit user-supplied priors would be an honest initial API; guessed native
defaults would not. Output fields `prior_omega_a` and `prior_omega_b` may allow
later native comparison. The guide fixes between-function correlations to
identity matrices; `C` groups functions sharing residual covariance, permitting
different wavelet residual variances across strata without cross-function
residual correlation. The paper's covariance model is more general.

The catalog provides executable archives and examples; no source archive has
been verified. The existing `pinnacle_wavelet.py` uses a redundant image
transform and cannot be substituted for an orthogonal decimated transform.
Boundary rules, coefficient ordering and reconstruction need independent
checks before any native compatibility claim.

The implementation preserves the coefficient-level covariance and mixture
shrinkage model, with bounded sampling and explicit assumptions. Empirical-Bayes
shrinkage calibration conditions on supplied variance components; automatic
variance initialization remains open. No native example execution is claimed here.

## Independent reduced-posterior reference

`tools/reference_wfmm_coefficients.R` enumerates the four inclusion patterns
of a two-fixed-effect Gaussian mixture. There are five curves, three shared
random intercepts and two transform coefficients with different variance
components. Given an inclusion pattern, integrating the Gaussian fixed effects
gives covariance `Sigma + X*diag(tau*gamma)*X'`; ordinary Gaussian conditioning
gives its posterior mean and covariance. Mixing these results with normalized
marginal likelihoods yields exact inclusion probabilities and posterior moments.

The input and posterior CSV fixtures retain all numeric choices. Variance
components are deliberately fixed in this reduced reference; the sampler also
estimates them, checked separately below. This reference targets shrinkage and Gaussian
integration, not the native variance-prior defaults or MCMC random-number
parity. Base-R generation completed in 0.09 seconds.

## Native example availability

Catalog download 431 (version 169) advertises `wfmm_v3_1_Example.zip` with a
partial pancreatic MALDI-TOF example. The visible description names
`Pancreatic_MYO25_wfmm_example.fig`, `wfmmdemo.bat`, and
`PlotPancreatic_MYO25.m`, but exposes no archive size, full member listing or
fitted variance-prior parameters. The download endpoint was inaccessible to
the read-only source review. No example archive or native program was run;
the native inverse-gamma defaults remain unverified.

The guide's posterior workflow reconstructs each coefficient draw before
computing curve summaries. Linear effect contrasts and time-region summaries
must therefore propagate posterior draws, rather than inverse-transforming
coefficient quantiles. Simultaneous bands require a maximum standardized
deviation over the time grid for each draw, a distinct calculation from
pointwise quantiles. The posterior layer implements these operations;
native output-file parity remains separate.

## Python transform checkpoint

The public basis API now supports identity, custom square orthogonal matrices,
and periodic decimated Daubechies db1–db10 transforms. Coefficients are packed
as `[a_J,d_J,...,d_1]`, using even-index decimation with forward tap offsets;
scale and partition labels accompany the immutable result. The default level
is the largest power of two dividing the curve length. The transform uses
bounded row-wise filtering rather than constructing a dense wavelet matrix.
Custom matrices use `D=Y*W` and `Y=D*W.T` after an orthogonality check.

Three focused checks pass after integration with explicit absolute tolerances:
hand-computed Haar coefficients, a scalar db4 reference, energy conservation,
inverse reconstruction, custom/identity transforms and a 12-point curve.
These verify the stated Python convention, not native boundary extension or
coefficient ordering.

## Explicit-prior statistical workflow checkpoint

`fit_wfmm_coefficients` supports coefficient-specific random-effect variances,
multiple random-effect levels and residual strata, and mixture-normal fixed
effects. Variance sampling uses explicit inverse-gamma shapes/scales and
positive-truncated Gaussian proposals with the normalization correction.
Optional random-effect draws use their Gaussian conditional. The covariance
model fixes between-function correlations to identity, matching the available
guide; native prior/proposal initialization is not inferred.

The fixed-variance sampler matches the independent exact mixture moments and
inclusion probabilities within declared Monte Carlo tolerances. A separate
conjugate inverse-gamma case gives posterior precision mean 2.6845 versus
2.7140 analytically; zero-information fixed effects retain proper prior draws.
These checks cover different inference components rather than reproducing the
implementation's update equations as expected values.

`wfmm_summarize` reconstructs posterior draws, applies effect/time contrasts,
and computes pointwise quantiles, effect-size probabilities, strict sign-tail
scores, simultaneous bands and SiMBaS probabilities. Hand-calculated contrast
and band values, transform inversion and constant-coordinate limits provide
deterministic references. Input, retained-output and intermediate dimensions
are bounded before allocation; transforms are reconstructed in chunks.

All nine focused transform, model and posterior tests passed together in
4.99 seconds. The public observed-curves-to-bands example in `docs/wfmm.md`
ran in 0.873 seconds with 116.8 MiB process peak memory and no process swaps.
It used two sequential chains, 64 warmup and 64 retained draws, eight
coefficients and optional random-effect draws. Variance acceptance ranged
from 0.426 to 0.648 over 8,192 likelihood evaluations; variances were positive
and output bands/probabilities finite. Reconstructed posterior means agreed
with an independent inverse transform of coefficient means within `5e-15`.
This short example validates the public workflow, not convergence for
scientific use or behavior on the inaccessible native pancreatic example.

Remaining coverage includes automatic variance and
proposal initialization, the native `delta_omega` mapping, additional transform
families and boundary conventions, compression, covariance/prediction workflows
and native files. The catalog remains partial until these gaps are resolved.

The wheel and source distribution built successfully from revision `c7f3a97`.
All 490 Python modules, the catalog and source notices matched the committed
files byte for byte; raw research downloads and local compiled binaries were
excluded. This offline packaging check used 74.75 MiB peak process memory and
reported no swaps. It checks distribution contents, not additional statistical
methods or native software parity.

## Independent shrinkage-calibration reference

`tools/reference_wfmm_shrinkage.R` supplies six curves, twelve coefficients,
two correlated fixed effects, two random-effect variance groups and two
residual strata. It computes joint GLS fixed-effect estimates and the required
conditional variances independently in base R. Joint GLS covariance diagonals
are also retained to detect accidentally substituting them for the conditional
variances; in this example the ratio exceeds two.

For each of the four fixed-effect/partition groups, the script maximizes the
two-component normal-mixture likelihood directly by BFGS from nine initial
values. It does not use the Python EM recurrence. The selected solutions have
positive definite objective Hessians and improve on both all-spike and all-slab
boundary likelihoods. The `wfmm-shrinkage-*.csv` fixtures retain inputs,
conditional statistics, optimized parameters and full log densities; Python's
relative log likelihood must subtract the standard-normal baseline before
comparison. Base-R generation completed in 0.15 seconds.

The integrated Python calibration agrees with these references: maximum
absolute errors are `4.22e-15` for GLS coefficients, `5.00e-16` for conditional
variances, `2.93e-8` for inclusion probabilities, `5.72e-7` for slab-to-sampling
variance ratios and `5.68e-14` for relative log likelihood. All four groups
converged in 33–104 iterations with tolerance `1e-11`; the comparison itself
used 0.0062 seconds and the process peaked at 113.2 MiB.

Six focused calibration/model checks passed in 4.79 seconds in the implementation
checkout; Ruff and mypy passed. The calibration rejects rank-deficient designs
and diagonal-normalized GLS information condition numbers above `1e12`. It
returns per-group convergence/objective diagnostics and a ready fixed-effect
prior, canonicalizing the unidentified zero-variance mixture to all-spike.
Prior arrays are bounded before joint zero-variance/probability checks.

All public guide examples then ran in the integrated checkout, followed by a
new fit using the calibrated prior and explicit inverse-gamma variance priors.
All six calibration groups converged; twelve all-spike coefficients stayed
exactly zero during sampling, variances stayed positive and curve summaries
were finite. The combined manual-prior and calibrated-prior workflow took
1.677 seconds, with 116.5 MiB peak process memory and no process swaps.
This verifies composition of calibration and fitting, not native initialization
or convergence adequacy for an applied scientific analysis.

## Verified covariance and prediction scope

The official guide, PDF pages 8–10, describes separate random-effect and
residual-stratum covariance outputs. For `D=Y*W`, each data-space covariance
is `W*diag(omega)*W.T`, the guide's two-dimensional inverse transform of the
diagonal coefficient-space matrix. It describes `sigma` as a **variance**
function and `rho` as a correlation matrix, both reconstructed from
`omega_mean`. The covariance reconstructed from mean omega is also the
elementwise posterior mean covariance by linearity. Correlation normalization
and square roots are nonlinear, so plug-in correlations/SDs differ in general
from posterior means of draw-wise correlations/SDs.

Native univariate covariance samples are available only for fewer than 1,000
time points, and plug-in rho/sigma for fewer than 1,500; the guide explicitly
cites memory restrictions. A Python implementation should bound the full
draw/component/time-by-time product and offer a diagonal-only path or streaming
summaries. An individual time-point cap alone does not bound retained draws.

[Morris and Carroll, Section 5.1](https://pmc.ncbi.nlm.nih.gov/articles/PMC2744105/)
also describes integrating future-curve predictions over posterior draws of
fixed effects, random effects and variance components. The guide documents
fitted fixed/random functions and covariance outputs, but no predicted-curve
output. Existing-subject latent means, future replicates conditional on existing
random effects, and new random-effect levels are distinct prediction targets.
An explicit Python API separating them would be a model-consistent extension;
the available sources do not establish native prediction-interface parity.

## Remaining transform and retention contracts

The guide, PDF pages 2–4, names `none`, `wavelet`, `PC`, `custom`, `PCw` and
`wPC`. `PCw` concatenates PC coefficients with a wavelet transform of the
remaining residual; `wPC` applies wavelets before PC. It specifies custom
forward and reverse matrices as `D=Y*phi_inv` and `Y=D*phi`, but this Python
basis currently supports only square orthogonal custom matrices.

Native outputs retain `Kstar` and `DIndex`, the selected coefficient count
and original indices. That supports explicit coefficient/partition selection
and reinsertion into original positions, with zeros in omitted coordinates,
before inverse reconstruction. Source highpass removes the most detailed
wavelet levels and lowpass removes the least detailed ones. The source does
not say whether lowpass counts the coarse approximation block; Python must
make approximation/detail selection explicit rather than claim native flag
parity from an assumed convention.

The guide calls `alphaPC` and `alphawav` retained-energy fractions in `[0,1]`,
but does not establish PCA centering/scaling, the exact energy calculation or
the compression threshold `P` used with its function-count parameter `t`.
It describes descending-eigenvalue partitions using half-unit bins of log-base
eigenvalues, with singleton-bin merging, but this does not specify the PCA
transform itself. The [v3.1 release notes](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/WFMM/wfmm_v3_1_ReleaseNotes.pdf)
confirm the added transform/filter names without resolving those calculations.
Energy-threshold, PCA and hybrid-transform parity therefore remain open.
