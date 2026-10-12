# WFMM conditional initializer integration

`initialize_wfmm` composes bounded MOM, GLS/shrinkage, fixed-shrunken-beta
conditional covariance optimization, final GLS/shrinkage, native-count-centered
inverse-gamma priors and explicit proposal scales. `fit_wfmm_conditional_variances`
also exposes the conditional fitting stage independently.

The default stable profile uses complete moment equations and mathematical EM,
allows initial floor components to move, and truncates absent lower roots at the
positive floor. Explicit legacy_3_1 retains recovered MOM/EB policies, the MOM-floor
freeze and native zero/artificial bounds. Its analytic-gradient optimizer is an
independent implementation; native Brent iteration and rounding are not claimed.
All components expose diagnostics rather than silently accepting signed-only
stops. Strict readiness requires actual convergence and positive usable proposals.
MCMC mixing and precision remain separate checks.

## Independent evidence

Nine original initialization studies preserve D, MOM/conditional estimates,
first and final pi/T/alpha, unshrunk GLS estimates/conditional variances, width
scales and inverse-gamma parameters. They include residual strata, random levels,
multiple fixed effects, zero signal, periodic Haar transforms, coarse exceptions
and an odds-cap case exposing the native floor freeze. Six related native JSON
files regenerate byte-identically.

First shrunken coefficients match native within1e-6 on these inputs. The tighter
independent optimizer differs from native variances by at most.0011, final
conditional sampling variances by.00022, inclusion probabilities by1.1e-5 and
slab-to-sampling ratios by.006. These are fixture-specific tolerances, not general
accuracy bounds. Independent exact RSS/N and balanced covariance-eigenvalue
solutions establish mathematical optima in the analytic cases. Direct likelihood
calculations verify roots and the restricted source objective. Two starts avoid
the demonstrated log-simplex boundary-start failure; projected gradients and
covariance condition numbers support a **local**, not global, convergence claim.

The55 new focused tests include invalid controls, iteration holds, ownership,
source composition, analytic checks and an automatic prior/proposal handoff into
MCMC and curve reconstruction. The2-chain32-draw example is a workflow smoke
check, not a precision or clinical-validation claim. Stable initialization is
ready in all9 reference studies. Explicit native modes flag zero proposals and
signed stops; the odds-cap case also flags a blocked likelihood improvement.

## Scope closed and next contracts

Closed: bounded automatic coefficient-domain initialization with audited
scientific/native differences and sampler-ready priors/proposals when qualified.
Not closed: native per-component frozen variance sampling, saved whole-study
initialization/MCMC/postprocessing/prediction orchestration, broader native
wavelet/boundary/multidimensional transforms, PCA/PCw/wPC and full reports.
Entry70 remains partial. Counts stay103 implemented/31 partial/4 pending.

Native PC probes during validation show uncentered singular-vector transforms,
PCw concatenation with wavelet-transformed residuals, and wPC after wavelet
analysis. A simple alphaPC=1 input exits with a zero-column resize error; the
.9/.5 probes retain2/1 components. These are leads requiring formal fixtures,
partition/boundary evidence and explicit defect handling before integration.
