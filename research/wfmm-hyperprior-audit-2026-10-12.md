# Optional pi/tau prior conversion: source audit, not production acceptance

`tools/reference_wfmm_hyperprior_audit.py` pins WFMM1/2 checksums, creates a native
Init file from synthetic data, then traces its own WFMM2 child. It supplies a
numerical centering matrix at the entry to the original beta (0x42b6e0) or
inverse-gamma (0x42be30) parameter conversion, retaining the configured variance.
The functions are named GetBetaParametersGivenMeanAndVariance and
GetInvGammaParametersGivenMeanAndVariance. Outputs are read at return, before
MCMC. The 16 probes regenerate byte-identically. No source code/binary bytes,
posterior samples, RNG equivalence or clinical claims enter this evidence.

The function names are insufficient mathematical specifications. Beta output at
center 0.2 and variance 0.06 is approximately a=1.14882019, b=1.80985587: variance
is 0.06, mean is about 0.3883 and mode about 0.1552. At smaller variance, the
mode is much closer to the supplied center; this does not establish intended
centering or justify copying a conversion into a scientific default.

Inverse-gamma outputs satisfy b=(a+1)*center, hence mode-centered under the
standard shape/scale definition. Their requested variance is not generally
attained. For center 0.1/variance 1, a≈1.81807; center 0.1/variance 1000,
a≈1.67036; center 1/variance 1000, a≈1.70756. These three shapes have infinite
variance. Other examples also disagree with requested finite variance. This
is an audit of these inputs, not an assertion about every native prior.

Independent mathematical references in the JSON solve explicit MODE/variance
constraints, without claiming this was the native intended convention. For a
beta distribution use a=1+m*c, b=1+(1-m)*c, with c>0 chosen to satisfy
v=(1+m*c)(1+(1-m)*c)/((2+c)^2(3+c)). For an inverse-gamma distribution, a>2,
b=m(a+1), and v=b²/((a-1)²(a-2)). Bracketed roots avoid relying on the original
conversion. These references identify a possible proper-prior route; they do
not yet add optional pi/tau updates to the production sampler. Explicit prior
shape/scale inputs and mathematically validated conditional updates are next.

Initial experiments rewriting an original MAT structure failed before reaching
these routines. The accepted audit instead lets native WFMM1 write its own
Init file, then supplies controlled numbers only at conversion entry. Those
rewriting failures are not claimed as native hyperparameter failures.

Subsequent local implementation adds explicit proper priors and independently
validated conditional updates; see
[the hyperprior checkpoint](wfmm-hyperprior-checkpoint-2026-10-12.md).
The native conversion findings above remain unchanged and are not copied into
the optional production calibration policy.
