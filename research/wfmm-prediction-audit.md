# WFMM posterior prediction audit

## Source-supported target

Morris and Carroll (2006), Section 5.1, defines the future-curve posterior
predictive distribution as

```text
f(Y* | Y) = integral f(Y* | B, U, Omega) f(B, U, Omega | Y) dB dU dOmega.
```

It states that Monte Carlo integration over posterior samples propagates
uncertainty in fixed effects, random effects and variance components. The
functional mixed model in Section 3 is `Y = X B + Z U + E`; Sections 4.1–4.2
put independent coefficient-specific Gaussian variances on random effects and
residuals. Section 5, step 4, gives the Gaussian conditional for observed random
effects. Source: [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC2744105/).

## Python prediction convention

`wfmm_predict_coefficients` computes future coefficient draws conditional on
retained `WFMMCoefficientFit` posterior states. It makes the target explicit:

- Fixed-only latent mean: `X_new B`.
- Existing levels: add `Z_existing U`, requiring the fit's retained conditional
  random-effect draws and exactly matching its level columns.
- New levels: add `Z_new U_new`, drawing each level independently with
  `U_new[l,k] ~ Normal(0, q[group(l),k])` for each posterior variance state.
  A level is one design column and is shared by every row loading that column.
- Future replicate: optionally add independent row residuals
  `E_new[i,k] ~ Normal(0, s[stratum(i),k])`.

The returned coefficient draws can be inverse-transformed and summarized by the
existing `wfmm_summarize` workflow. New random levels and residuals require an
explicit NumPy Generator; fixed-only and existing-level latent means are
calculated deterministically from retained draws. Separate existing/new design
matrices allow a prediction set to mix known and new levels without ambiguous
index inference.

This is model-consistent posterior prediction under the implemented
independent-level, diagonal-coefficient covariance model. The paper gives no
native prediction input/output schema, and no native executable prediction
parity is claimed. Repeated levels are represented by shared design columns;
residual rows are independent conditional on the posterior state.

## Validation

Focused tests use a small synthetic `WFMMCoefficientFit` to verify exact fixed
and retained-existing coefficient predictions, shared new-level contributions,
residual mapping, read-only outputs, RNG requirements, malformed mappings and
preflight failures. Root's independent base-R fixture compares mixture
predictive means/covariances and conditional means for fixed-only, existing,
new-latent and replicate targets under varying posterior draws and a
nonorthogonal synthesis matrix. The integrated comparison executes the existing
inverse-basis reconstruction and checks its resulting predictive curves; it
does not claim native runtime parity.

The five focused tests pass, as do targeted Ruff, formatting and mypy checks.
The worker's test run peaked at 132.11 MiB RSS and reported zero swaps. A
direct-module guide smoke run also passed; package-level examples are checked
after public export integration.

The independent base-R comparison passes 192 deterministic conditional curve
values and all 168 predictive mean/covariance checks. Stochastic targets each
use 8,192 draws from eight equally weighted supplied posterior states, with
shared new levels, distinct variance strata and nonorthogonal reconstruction.
The maximum discrepancy is 2.081 estimated Monte Carlo errors. The comparison
took .266 seconds, peaked at 125.23 MiB RSS and reported zero swaps. This checks
propagation conditional on supplied posterior states; it is not a new
validation of the original fitting posterior. Reproduce with
`tools/reference_wfmm_prediction.R` and `tools/check_wfmm_prediction.py`.
