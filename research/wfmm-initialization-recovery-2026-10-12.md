# WFMM initialization: conditional likelihood recovered

The checksum-verified original WFMM1 3.1 executable is now traced at
`WFMM::loglik1` (`0x49d960`) using child-only Linux x86-64 ptrace.
`tools/reference_wfmm_initialization.py` generates four wholly synthetic designs,
records actual variance inputs, coefficient inputs and returned likelihoods, and
reads the original initialization outputs. The committed numerical fixture
contains969 likelihood evaluations:125 identity,325 heterogeneous residual
strata,195 random-intercept and324 two-random-level evaluations. Regeneration
is byte-identical. No original binary, source listing or patient data is included.

Independent dense covariance calculations match

`logL = -0.5 * [logdet(V) + (y-X beta)' V^-1 (y-X beta)]`

with `V = sum_h omega_h Z_h Z_h' + diag(C omega_residual)`. Maximum absolute
errors are8.89e-15,1.78e-13,1.57e-13 and3.20e-14 respectively. The five tests in
`test_wfmm_native_initialization_reference.py` calculate this directly without
calling a production WFMM implementation.

## The guide's profile label is not the executable's full contract

The beta passed to every traced likelihood call remains **fixed at the shrunken
MOM initialization**. It is not refit as the variance changes. The trace also
records the separate unshrunk GLS coefficient input. The shrunken coefficient is
`beta_MOM_GLS * alpha_MOM * tau_MOM / (1+tau_MOM)` on these one-partition designs.
The saved final `betastar_ns` can differ from the initial GLS value for heterogeneous
strata; it cannot be substituted into this earlier step.

Thus the guide's `omega_MLE` label does not justify ordinary profile ML or REML
parity. For identity coefficient1, unshrunk profile ML is2.484375, the exact
fixed-shrunken-beta optimum is2.4944369145956484, and the native optimizer returns
2.494323656093732. The latter discrepancy is numerical optimization tolerance;
the former difference is a distinct conditional statistical objective. The
existing explicit Python REML initializer remains unchanged.

## Proposal-scale lead sharpened

Original initialization46c5cc–46c6d2 subtracts the upper/lower likelihood bounds
and multiplies by0.5. On all four identity coefficients, independently solving
for log-likelihood drop1 around the native optimum reproduces `se_omega` within
3e-5 (native root tolerance). It is a likelihood-width scale in variance units,
not the Hessian variance of an MLE. This is not yet the downstream Metropolis
proposal mapping: inspect WFMM2 `MCMC_Update::UpdateTheta` at0x4936a0 before
choosing how `propvar_omega` and this scale enter proposals.

## Precise next contracts

Recover the general MOM iteration and flooring/zero-component policy; verify
native empirical-Bayes floors/coarse-scale exceptions and its handoff to the
fixed-beta conditional optimizer; validate conditional optimization/bounds across
all supported designs; trace WFMM2's proposal mapping. Then integrate saved
initialization/MCMC/postprocessing studies and the remaining transform/prediction
contracts. Original guide workflows are split across wfmm1, wfmm2 and wfmm3.
Entry70 remains partial; this source-recovery checkpoint changes no production
Python and does not promote it.
