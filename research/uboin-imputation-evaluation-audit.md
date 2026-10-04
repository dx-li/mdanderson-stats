# U-BOIN multiple-imputation evaluation source audit

Audit date: 2026-10-04. This record covers only the Stage-II evaluation layer
for already-supplied efficacy predictions; it does not claim the delayed
efficacy predictor has been recovered.

## Source-to-code crosswalk

The cached Zhou, Lee and Yuan paper text (`research/raw/UBOIN/paper.txt`,
§2.5, including its discussion around Eqs. 5 and 9) describes multiple
imputation for pending efficacy. For each posterior draw of the prediction
model, binary pending responses are completed, the complete-data posterior
utility and low-efficacy posterior probability are evaluated, and those
functionals are averaged before the §2.4 conduct rule is applied. Therefore
the implementation evaluates `uboin_posterior` on each completion and averages
its `mean_utility` and `low_efficacy_probability`; it never merges imputed
counts into one pseudo-dataset. Observed toxicity is retained in every
completion, so the overdose posterior is invariant to efficacy draws.

`src/mdanderson_stats/uboin_conduct.py` provides the shared `_stage2_action`
ordering used by both complete-outcome decisions and this evaluator:
lowest-dose safety stop, total and per-dose stopping, B1 escalation, then B2
admissibility and allocation. `src/mdanderson_stats/uboin.py` supplies the
Dirichlet posterior summaries and allocation-by-summary primitive.

## Explicit boundary

The cached primary text mentions a scaled logistic model, standardizes the
quickly observed response to mean zero and standard deviation 0.5, and gives
priors for its parameters. However, the actual scaled-logistic equation and
complete observed-data likelihood/standardization recipe were not recoverable
from the inspected paper text or cached application metadata. The cached app
labels the immune-response option as under development. The protected PMC
route was challenged and the publisher supplement route returned access
denial; neither was retried or bypassed. Accordingly,
`efficacy_probabilities` is an explicit caller input and must not be described
as a fitted U-BOIN model or as a reconstructed native prediction.

The paper says to use at least five imputations and uses 20 in its simulation.
This API enforces H≥5 but leaves the predictive draws and their dependence
across patients to the caller. It is a bounded Stage-II evaluator, not a
time-to-event model, interim accrual simulator, or full delayed-efficacy
workflow.
