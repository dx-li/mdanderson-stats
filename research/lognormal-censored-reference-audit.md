# Independent log-normal censored-posterior reference

This audit describes an independent base-R quadrature target for a prospective
Python right-censored log-normal fit. It validates an explicitly selected
proper prior and the observed-data likelihood; it does not recover BCSTTE's
native prior or fitting algorithm, and it does not define a censored Johnson
diagnostic.

## Contract and prior

The cached BCSTTE guide §4.5 defines `Y=log(T) ~ Normal(mu,V)` with
`V=sigma^2`. Its input contract uses indicator one for uncensored/event and
zero for right censoring. It requires at least one uncensored observation. The
guide does not state a prior or censored posterior algorithm.

The reference uses the same explicit proper Normal-Inverse-Gamma prior as the
complete-data Python workflow:

```text
V ~ InvGamma(a0=4.5, b0=1.1)       [shape/scale]
mu | V ~ Normal(m0=0.30, V/kappa0=V/2)
```

For comparison with centered Python parameters, each case uses its first
positive time as reference. This shifts only the prior location: centered
`m0=0.30-log(reference)`. For mixed data the reference is `0.4`, giving
`m0_centered=1.216290731874155`; for all-censored data it is `0.7`, giving
`m0_centered=0.6566749439387324`. The NIG variance hyperparameters and location
precision do not change.

## Direct likelihood quadrature

`tools/reference_lognormal_censored.R` integrates the *observed-data*
likelihood directly, without latent augmentation. It changes coordinates to
`q=sqrt(kappa0)*(mu-m0_centered)/sqrt(V)` and `z=log(V)`. Under the NIG prior,
`q` is standard Normal and independent of `z`; the inverse-Gamma density in
`z` includes the `dV/dz=V` Jacobian. For an event at `t`, the likelihood is
the Normal density of `log(t)` under `(mu,V)` with the original-time density
Jacobian `1/t`. For a right censor at `c>0`, it uses
`Phi((mu-log(c))/sqrt(V))`. A zero-time censor contributes exactly one.

Tensor Gauss-Legendre quadrature compares 81-, 121- and 151-point grids over
the same `q in [-7,7]`, `z in [-11,7]` domain, then a 151-point grid over the
wider `q in [-8,8]`, `z in [-13,9]` domain. The first three resolutions check
grid convergence at fixed domain; the last checks tail-domain sensitivity at
fixed quadrature order. The fixture records posterior means, variances and
covariance for centered location and log variance, plus posterior means of the
CDF at every positive observed time.

The mixed case has times `[0.4, 1.2, 1.2, 2.6, 5.0, 0, 3.5]` and event flags
`[1, 0, 0, 1, 0, 0, 0]`; the repeated `1.2` censor threshold and zero-time
censor check both edge conventions. The second case has all times censored at
`[0.7, 1.5, 2.0, 4.0, 4.0]`; the proper NIG prior makes this posterior proper.
This all-censored case is a Python mathematical extension, not a claim of
native application parity with the guide's one-event minimum.

## Validation status

The generator ran in base R in about 0.33 seconds. From the 81- to 121-point
same-domain grids, the largest absolute metric changes were `9.14e-8` for the
mixed case and `2.03e-8` for all-censored. Refining from 121 to 151 points at
the same domain reduced the maximum changes to `9.65e-13` and `2.08e-13`,
respectively. Expanding the domain at order 151 changed metrics by at most
`3.86e-13` (mixed) and `7.60e-10` (all-censored). These comparisons support
both grid and domain convergence for these two bounded cases. The fixture is
ready for independent comparison with the Python latent log-time Gibbs fit.
No native prior, native censoring algorithm, or censored Johnson calibration
is inferred.
