# WFMM implementation handoff

Entry 70 remains pending. Primary mathematical source:
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

Next implementation should preserve the coefficient-level covariance and
mixture shrinkage model, with bounded sampling and explicit assumptions.
Neither a generic mixed-model wrapper nor wavelet denoising alone completes
WFMM. No implementation or native example execution is claimed here.

## Independent reduced-posterior reference

`tools/reference_wfmm_coefficients.R` enumerates the four inclusion patterns
of a two-fixed-effect Gaussian mixture. There are five curves, three shared
random intercepts and two transform coefficients with different variance
components. Given an inclusion pattern, integrating the Gaussian fixed effects
gives covariance `Sigma + X*diag(tau*gamma)*X'`; ordinary Gaussian conditioning
gives its posterior mean and covariance. Mixing these results with normalized
marginal likelihoods yields exact inclusion probabilities and posterior moments.

The input and posterior CSV fixtures retain all numeric choices. Variance
components are deliberately fixed in this reduced reference; the future WFMM
sampler must still estimate them. This check targets shrinkage and Gaussian
integration, not the native variance-prior defaults or MCMC random-number
parity. Base-R generation completed in 0.09 seconds.
