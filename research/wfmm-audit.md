# WFMM implementation handoff

Entry 70 remains pending. Primary mathematical source:
[Morris and Carroll (2006)](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/WFMM/Morris%26Carroll2006.pdf),
Sections 4–5 and Appendix A. The orthogonal wavelet transform converts
functional observations into coefficient-specific mixed models. Each location
and scale has its own random-effect and residual variances; replacing these
with one variance per scale would change the model. Fixed-effect coefficients
have zero/normal mixture priors. Shrinkage hyperparameters can be elicited or
estimated by the empirical Bayes procedure in Section 4.4.

The sampler marginalizes random effects while updating fixed effects and
variance components. Fixed effects use conditional mixture-normal updates;
the scalar conditional estimate must use the residual after subtracting other
current fixed effects. Variance components use positive truncated-normal
Metropolis proposals. Optional random effects are then drawn from their
Gaussian conditional. Inverse transforms recover functional estimates.
The PDF's Bayes-factor equation and exact empirical Bayes iteration require
further transcription review before implementation.

## Native scope and gaps

The [official guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/WFMM/wfmm_Users_Guide.pdf)
documents data matrices, fixed/random-effect designs, residual strata,
compression, wavelet and other transforms, and posterior summaries. It also
specifies an inverse-gamma variance prior controlled by `delta_omega`, but the
exact native shape/rate mapping is not established by the current audit.
Explicit user-supplied priors would be an honest initial API; guessed native
defaults would not. The guide's independent random effects and residual strata
need to be distinguished from the paper's more general covariance structures.

The catalog provides executable archives and examples; no source archive has
been verified. The existing `pinnacle_wavelet.py` uses a redundant image
transform and cannot be substituted for an orthogonal decimated transform.
Boundary rules, coefficient ordering and reconstruction need independent
checks before any native compatibility claim.

Next implementation should preserve the coefficient-level covariance and
mixture shrinkage model, with bounded sampling and explicit assumptions.
Neither a generic mixed-model wrapper nor wavelet denoising alone completes
WFMM. No implementation or native example execution is claimed here.
