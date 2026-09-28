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
calibration and automatic initialization remain separate gaps. No native example
execution is claimed here.

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

Remaining coverage includes empirical-Bayes shrinkage, automatic variance and
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
comparison. Base-R generation completed in 0.15 seconds. Comparison with the
Python calibration is a separate validation step.
