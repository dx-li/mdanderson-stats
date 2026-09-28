# BaCIS latent classification posterior

The archived bacistool 1.0.0 `bacisThetaPosterior` returns `ModelOne`'s retained
`theta` samples. `bacisPlotClassification` applies R's density estimator to
those samples for display. Both use `model1Str` in `R/internal.R`, whose
data likelihood depends on theta only through its sign:

```text
theta_i ~ Normal(0, precision=tau2)
ind_i = step(theta_i) + 1
eta_i | ind_i ~ Normal(logit(phi_ind_i), precision=tau1)
y_i | eta_i ~ Binomial(n_i, logistic(eta_i)).
```

Conditional on its sign, theta's magnitude retains its prior half-normal
distribution. Its posterior density is therefore a negative/positive
half-normal mixture with weights from `bacis_classify`. The default native
latent precision is `.001`. It changes theta's scale but does not change the
classification probability, which is already calculated by integrating out
that sign variable.

For `z=theta*sqrt(tau2)` and posterior weights `wL,wH`, the exact density is
`2*sqrt(tau2)*normal_pdf(z)` times the weight for the corresponding half-line.
The density at zero can be represented by the right-hand value, consistent
with `step(0)=1`; its value at that single point does not alter the posterior.
The CDF is `2*wL*normal_cdf(z)` on the negative half-line and
`wL + wH*erf(z/sqrt(2))` on the nonnegative half-line. Complementary tails
should be computed directly, including near zero and for small component
weights, rather than by subtracting a rounded CDF from one.

The mean is `(wH-wL)*sqrt(2/(pi*tau2))`, the second moment `1/tau2`, and the
variance `[1-(2/pi)*(wH-wL)^2]/tau2`. Independent posterior samples can be
drawn by assigning an independently drawn half-normal magnitude the positive
sign with probability `wH`. This represents the exact statistical target of
the native MCMC output. It does not reproduce its random stream, dependence
between iterations, finite-sample histogram or smoothed density estimate.

`tools/reference_bacis_theta.R` uses the 19 independent R classification
references, three latent precisions, and a grid covering both infinities,
normal tail arguments and points within `1e-12` of zero. It calculates
near-zero normal mass by direct quadrature, avoiding cancellation between
nearly equal CDFs. Generation succeeded for 513 density/CDF/survival rows and
57 moment rows without JAGS. Agreement with the Python implementation remains
pending. Source provenance is shared with [bacis-sources.json](../docs/bacis-sources.json).
