# BARD BF-BLRM stochastic-generation audit

## Source boundary

The cached BARD paper's BF-BLRM model definition (Section 2, printed pp. 9–10)
gives `logit(p_j)=log(alpha)+beta*(d_j/d*)`, positive `alpha,beta`, and
independent Normal priors on `log(alpha), log(beta)`. Its simulation settings
(Section 3, printed p. 16) specify five doses, a 30-patient BF-BOIN escalation
cap, cohorts of three, accrual rate 3/month, one-month DLT window and backfill
cap 12. BF-BLRM instead uses a calibrated escalation cap whose value and
general calibration procedure are not recovered. The source also does not
specify the stochastic arrival-gap law or exact outcome-time generation. The generated
workflow therefore makes arrival law/rate, DLT window, explicit patient cap,
prior, target, and sampler settings caller inputs; it makes no native RNG
claim. Existing `run_bard_blrm_trial` remains the sole conduct and inference
engine.

The paper's simulation prior/settings (Section 3, printed pp. 17–18) use
target interval `(0.16, 0.33)`, doses `(10,20,50,100,200)`, reference dose
50, `eta=0.30`, and `(log(alpha),log(beta)) ~ Normal((-1.1,0), diag(4,1))`.
Under the paper's printed raw-ratio curve and positive-beta prior, every dose
probability is above `expit(log(alpha))`; hence prior `POD` at the 0.33 cutoff
is at least `Pr(log(alpha)>=logit(0.33))`, about 0.42, which exceeds eta.
Exact prior probabilities therefore trigger the Python all-overdose initial
screen; finite Monte Carlo estimates can vary. The
generation layer does not change the source model equation, prior, or screen
to force a simulated enrollment; examples use an explicitly illustrative
prior that passes the implemented screen.

## Implemented generation contract

`BARDBLRMSimulationDesign` snapshots bounded doses, target and optional
titration inputs and validates the fit-retention, outcome-cell, and work
limits. `simulate_bard_blrm_stage_one` validates response calibration and
optional joint endpoint probabilities before drawing. Profiles are drawn from
the model's explicit joint weights. The helper generates all potential dose
columns with independent counterfactual outcomes, then the calendar replay
observes only the assigned dose. This counterfactual independence is an
explicit Python convention, not a paper-defined dependence structure.

DLT times use the shared Weibull calibration `F(w)=p`, `F(w/2)=p/2`; endpoint
probabilities 0 and 1 are handled explicitly. Assessment delays are `w` for
non-DLT and the sampled DLT time for DLT. Response assessment occurs at `w`.
When `q=P(DLT,response)` is supplied by dose/profile, the shared BF-BOIN
conditional-probability helper governs response generation; absent `q`, the
Python convention is endpoint conditional independence. Grade-2 titration
probability and delay are explicit inputs, with grade-2 generated only for
no-DLT potential outcomes.

The output keeps immutable arrival-indexed profile, probability, event, and
timing tapes beside the exact `BARDBLRMTrial`. Accepted patient records point
back to their original arrival index, so declined arrivals are preserved and
not recycled. Integer seeds are captured; caller-owned generators advance
in-place and cannot be reconstructed from the result.

The workflow does not claim native calibrated cap/defaults, native arrival or
outcome RNG identity, dependence among counterfactual doses, or exact native
calendar behavior. The source's model equation and prior are used as currently
implemented by `fit_bard_blrm`; generation does not switch to a different
dose parameterization to force a published example through initial safety
screening.
