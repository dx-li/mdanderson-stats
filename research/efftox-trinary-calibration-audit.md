# Trinary EffTox prior calibration audit

The continuation-ratio model source uses two independent Gaussian coefficient
pairs. Toxicity is `T=logistic(mu_T+beta_T*x)` and conditional efficacy among
non-toxic patients is `Q=logistic(mu_Q+beta_Q*x)`, with both slopes conditioned
positive. The three mutually exclusive cell probabilities are
`((1-T)(1-Q), (1-T)Q, T)`. Consequently, an elicited marginal efficacy mean is
`E[(1-T)Q]`; it is not the conditional mean `E[Q]`.

The available source does not specify a trinary prior calibration procedure or
its objective. The Python implementation therefore documents its own policy:
first fit the toxicity coefficient pair to toxicity means and its beta-moment
ESS target; then fit the conditional-efficacy pair to marginal efficacy means
and marginal efficacy beta-moment ESS, integrating over the already calibrated
independent toxicity prior. The efficacy second moment uses
`E[(1-T)^2] E[Q^2]`, preserving uncertainty in both blocks. Separate efficacy
and toxicity ESS targets are accepted. The result reports both mean residuals
and optimizer diagnostics. This sequential objective is a stated Python
elicitation choice, not a native Windows default or parity claim.

The input feasibility check requires elicited efficacy to be less than one
minus fitted mean toxicity at every dose. This is necessary for the requested
conditional mean `E`-target divided by `1-E[T]` to lie below one; actual fitted
marginal mean targets and residuals are retained because the nonlinear prior
fit need not interpolate all elicited points exactly.

`tools/reference_efftox_trinary_calibration.R` independently integrates
logistic-normal means and second moments with base R, then applies the product
moment identities. It emits fixed-slope cases for testing the induced
toxicity/conditional-efficacy uncertainty and marginal beta-moment ESS. It does
not test optimizer parity. Native Windows trinary calibration behavior remains
unverified.

## Numerical validation and limits

Independent base-R fixtures cover twelve dose/prior combinations, including
near-fixed intercept distributions. Integration of centered squared deviations
and the nonnegative product-variance identity avoid cancellation in those
references. Seven focused tests cover fixed/near-fixed moments, positive and
negative extreme logits, separate ESS targets, incompatibility rejection and
independent recomputation of the returned objective. The tests do not establish
Windows calibration parity or global optimality of the nonlinear fit.

The default three-dose public example converges in both stages: toxicity uses
431 evaluations/254 iterations and conditional efficacy uses 494/283. Achieved
mean efficacy/toxicity ESS values are 0.8000035097/1.1000007421 for targets
0.8/1.1; the largest marginal efficacy mean residual is 0.00122147. Gaussian
slope means retain their supported bounds before positive-slope conditioning.
The calibration result's `moments` field includes both outcome blocks.

Root integration ran all seven tests and both public guide blocks in 29.749
seconds with 145.17 MiB peak RSS and zero swaps. Targeted lint, formatting and
type checks pass. The numerical work ran serially with one BLAS/OpenMP thread;
no dependencies or CI workflows were added. Quadrature errors and optimizer
convergence remain distinct, and callers must assess residuals and bounds.
