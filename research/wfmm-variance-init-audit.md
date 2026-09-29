# WFMM variance initialization audit

The recovered WFMM implementation accepts explicit starting values for each
coefficient's random-effect and residual variances. Its empirical-Bayes helper
calibrates fixed-effect shrinkage conditional on those variances; it does not
estimate them. The source audit did not recover the native automatic
initialization algorithm or the mapping from `delta_omega` to the
inverse-gamma prior parameters. That missing native formula and its scale and
boundary conventions prevent a defensible claim of native parity.

`initialize_wfmm_variances` therefore exposes a separately labelled Python
policy. For each coefficient it maximizes the restricted Gaussian likelihood
under the covariance already implemented by WFMM,

`V = sum_g(q_g Z_g Z_g.T) + diag(s[residual_strata])`.

It rejects nonpositive residual degrees of freedom, deficient fixed designs,
and covariance components that cannot be distinguished after projection away
from the fixed-effect space. A single residual component has the closed-form
REML estimate; multiple components use bounded, serial, per-coefficient
optimization with an explicit likelihood-evaluation limit. Reported starts
include raw estimates, optimizer status, boundary indicators, and residual
diagnostics. A scale-relative positive floor is applied only to sampler starts
and never changes the reported raw estimate or defines a prior. All-zero data
have no internal variance scale and require a caller-supplied absolute floor
for usable starts.

Independent checks use the closed-form residual-only estimator and balanced
random-intercept ANOVA REML equations, with likelihoods recomputed from the
resulting covariance in base R. These validate the implemented model and
Python policy, not the undocumented native initializer.
