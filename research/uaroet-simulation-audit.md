# UAROET complete-outcome trial simulation contract

Current source inspection on September 28, 2026 confirms three implemented
layers: ordinal/correlated probabilities in `uaroet.py`, explicit-prior posterior
fitting in `uaroet_fit.py`, and the paper's allocation equations in
`uaroet_decision.py`. Full trial simulation is not yet implemented. The
[existing audit](uaroet-audit.md), [next-method source notes](uaroet-next-audit.md)
and [public guide](../docs/uaroet.md) establish the method and its limitations.

## Trial inputs and progression

A truth tensor has axes `(dose, efficacy_category, toxicity_category)`, matching
posterior counts. Each dose's joint table must be a probability simplex. Draw
one joint ordinal cell per assigned patient; independent marginal draws would
discard the specified endpoint association.

Require explicit priors, utility table, toxicity tail/limit, probability cutoffs,
good-outcome threshold, starting dose, maximum enrollment, analysis looks and a
nonincreasing utility-tolerance schedule. Retain the existing monotonicity flags
and optional fixed association. Neither universal cohort/analysis schedules nor
calibrated study-specific priors are identified by the general paper; these
choices must stay visible in the API.

The first assignment uses the physician-specified starting dose without posterior
filters. Subsequent scheduled analyses fit the accumulated complete ordinal
counts and call `uaroet_allocation`. Interim allocation uses posterior predictive
good-outcome weights and the existing no-skip escalation rule. An empty eligible
set stops enrollment with no selected dose. Do not add a separate efficacy
futility test or replace the existing joint decision rule by a toxicity-only stop.

At maximum enrollment, apply the existing final rule without the interim
escalation restriction. Preserve explicit `acceptable` versus `paper_global`
interpretations: the paper's global-maximum notation can identify a dose outside
the acceptable set. These must not be silently merged or labeled equivalent.
An analysis schedule must include the final sample size; its tolerance applies
to final selection as well as interim decisions.

## Precision, randomness and bounded work

Patient outcomes, randomization coins and posterior sampling use separate random
streams. Changes to sampler effort must not consume patient-generation draws.
Fixed draws/warmup/chains remain visible, along with per-look convergence and
Monte Carlo diagnostics. A completed sampler run does not establish convergence.
Do not silently replace a failed fit or conceal a failed configured diagnostic
criterion. Native random-stream and trajectory parity are not established.

Run trials and fits serially, discard posterior draws after computing each
analysis summary, and retain full path histories only when requested for one
trial. Reuse the posterior's two-million-cell retention limit and per-fit
likelihood/rectangle evaluation budget. Bound trial count, planned analysis
work and total retained outputs before allocation. Charge actual fit work to a
whole-simulation budget and report trial/look context if it is exhausted.

## Useful outputs and verification

A replay should retain assigned dose, assignment probabilities, paired ordinal
outcomes, observed counts, decisions and compact sampler diagnostics. Simulation
summaries should include per-dose selection probabilities plus no selection,
stopping/enrollment summaries, per-dose allocations and observed outcome/utility
summaries with appropriate Monte Carlo uncertainty. Do not retain every
posterior tensor for every trial.

Existing independent posterior and allocation evidence can be reused. New checks
should target complete trial progression, joint-outcome generation, independent
random streams, terminal selection and work limits. Small deterministic outcome
scenarios or explicit random-number tapes can test those behaviors without a
large simulation study or a duplicate numerical test matrix.

Native prior files, pseudo-trial/ESS elicitation, adaptive posterior precision,
reports and native executable parity remain distinct gaps after this extension.

## Integrated complete-outcome checkpoint

Luna's `d7bc88b` was integrated as `43eb23b`. The public trial/simulation APIs
follow the contract above. Review corrected assignment of one sampled dose to
a whole interval: every patient after the first look now has its own allocation
coin, with the latest probabilities retained in the patient record. Reported
stream seeds instantiate the actual generators. Fixed association is excluded
from aggregated sampling diagnostics because its R-hat is undefined by design.
Utility means use integer joint-cell totals and scaled weights to avoid raw-sum
overflow. Failures retain both trial and analysis-look context.

`tools/reference_uaroet_trial.R` generated five analysis rows and fourteen
patient rows across a final selection, early stop and final no-selection path.
Independent scalar posterior integration at fixed zero association supplies
mean-utility references. Root replay matched all assignments, paired cells,
counts, eligibility, final decisions and early/final stopping distinctions.
With 800 retained draws per chain, two chains and 200 warmup iterations, the
largest utility-mean difference was 0.01102 and largest sampled-coordinate
split R-hat was 1.01124. These are bounded reference checks, not a guarantee
of sampler precision in every study.

The warning-free root check took 1.366 seconds, peaked at 109.12 MiB and reported
zero swaps. Two focused worker tests cover between-look independent patient
randomization, seed replay, count conservation and large finite utility scaling;
Ruff and targeted mypy passed. Both public guide examples and deliberately tight
runtime work budgets passed, including trial/look context on exhaustion. No
large simulation, full-suite rerun, new dependency or CI expansion was used.
