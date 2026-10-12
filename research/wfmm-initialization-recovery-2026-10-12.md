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

## MOM and EB integration; expanded source evidence

The new `wfmm_mom_initialization` implements iterative trace equations with joint
GLS updates. Nineteen synthetic designs provide166 actual native normal-equation
systems, including multiple random levels, crossed/unbalanced designs, fixed
trends, two/three residual strata, scaled responses, iteration exhaustion,
changed floors/thresholds and a signed-only stop. Every recorded A/b/raw solution
is also checked by independent dense matrix calculations. Floored native
Woodbury systems can have condition numbers near1e8; their comparisons use a
separate2e-6 relative bound, not a machine-precision parity claim.

The complete mathematical equation is `A_ij=tr(P_i Q_j)` where random P is the
column-space projector and Q is ZZ', and residual P/Q are stratum indicators.
The source off-diagonal loop488970–4889a8 runs only through columnH, so later
residual columns incorrectly retain zero off-diagonals. This matters when random
components and heterogeneous residual strata coexist. Native GLS additionally
omits random components at/below VC0_thresh, and native stopping is the signed
maximum of new-minus-old variance. These behaviors require `legacy_3_1`.
Default stable mode uses complete trace equations, all floored components and
absolute stopping. Both profiles report absolute convergence separately.

The observed signed-only native stop takes7 iterations, with absolute change
0.0176823 despite the native1e-3 tolerance. It is explicitly not marked converged.
No empirical MOM iteration convergence theorem or optimality claim is made.

Twenty original EmpBayes calls across10 synthetic cases establish native floors,
odds cap, partition coupling, coarse exceptions and returned gamma. Native starts
pi=.5,T=1; updates partition pi to clipped mean gamma; computes T as weighted mean
squared standardized estimates divided by the **clipped/overridden** new pi,
minus1, then floors at minT. Unsmoothed partitions override pi=1-minp,T=bigT.
Exponent ceiling250 and odds ceilingmaxO apply. The global stop is signed maximum
change<=1e-4, not absolute/relative convergence. Gamma is the preceding E-step,
not recomputed from the returned pi/T. The explicit `wfmm_native_empirical_bayes`
API retains these semantics and reports absolute convergence independently.
Zero signal stops after2 iterations atpi0.17573408380111574,T10, while the absolute
change is0.2384794785719794. Existing mathematical EM remains separate.

Four actual WFMM2 proposal matrices at UpdateTheta4936a0 prove
`proposal_sd = propvar_omega * se_omega` at multipliers.5,1,1.5,4. Each includes
24 positive and8 zero scales. No RNG-stream or MCMC equivalence is claimed.

## Bound fallback is not a likelihood interval

Four original Bisect return matrices now preserve MLE/upper/lower together.
Independent dense likelihood roots confirm the positive roots (absolute native
root tolerance5e-5 on these inputs). Native MLE-floor components return **both
bounds zero**. Where no nonnegative lower drop1 root exists, native returns
**lower=-upper**: this is an artificial symmetry fallback, not a valid variance
bound. Subtracting these bounds and halving doubles the scale relative to using
zero as a truncated lower endpoint. This qualifies the earlier identity-only
half-width result; never present all native scales as confidence intervals.
The native return uses a hidden structure-result pointer, explicitly handled by
the reference tracer. The fixture exports only synthetic numeric observations.

Scope closed here: bounded MOM and native standardized-input EB components,
source proposal mapping and conditional-bound semantics. Scope still open:
compose GLS/EB with the conditional optimizer, safe/native scale usability,
complete transform/extension coverage, saved WFMM1/WFMM2/WFMM3 studies and
postprocessing/prediction integration. Entry70 remains partial.

A fifth bound fixture sets maxO=2 in the balanced random-intercept design.
Native components starting at the MOM floor remain frozen there during MLE.
Independent balanced-eigenvalue formulas show conditional optima q=.533729 and
q=1.229817 for two such coefficients, with log-likelihood improvements.0401964
and.0943257. This is a **restricted optimizer**, not global conditional ML over
all positive variance components. Native bound tolerance is1e-4 for this extra
case (maximum observed root error5.15e-5), versus5e-5 on the original four.
The next initializer must expose the floor freeze in its compatibility profile;
a mathematical optimizer must be free to leave that floor. A log-parameterized
simplex started exactly at its lower bound can also falsely stick there, so it
is not an adequate independent optimization check without analytic or multi-start
validation.
