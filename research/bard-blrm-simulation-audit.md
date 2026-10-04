# BARD BF-BLRM operating-characteristic workflow audit

The cached paper reports a full stochastic two-stage BARD-BLRM evaluation,
distinct from the BARD guide's app workflow, whose stage-one design is BF-BOIN.
The study uses 30,000 trials and reports sample size, duration, prognostic
factor and allocation imbalance, and correct selection for both final OBD
methods (`research/raw/BARD/paper.txt:691–823`, Table 4 at `:1294–1385`).

`simulate_bard_blrm` composes the explicit-prior BF-BLRM calendar replay and
the complete-outcome BARD stage-two continuation into bounded serial trials.
Each replicate is generated and reduced immediately; individual patient
ledgers are not retained. All valid trials, including no-MTD and unavailable
pair outcomes, remain in unconditional selection and PCS denominators. The
selected-only accuracy and pair/factor imbalance use separate reported
denominators. Every modeled factor is summarized, including factors omitted
from minimization.

The paper supplies the five-dose settings, prior, toxicity target interval,
EWOC cutoff, response model coefficients, response intercepts, stage-two
target and allocation probability (`paper.txt:695–709`, `:739–777`,
`:1403–1428`). It does not give the BF-BLRM stage-one cap: instead it says
that this cap was calibrated to match BARD-BOIN's mean stage-one enrollment.
The API therefore requires an explicit maximum. The paper does not specify a
stage-two arrival/assessment schedule or a joint DLT-response distribution;
the implementation exposes timing settings and requires/records an explicit
joint-outcome policy. These choices do not claim exact paper table replication.

The weakly informative paper prior and raw-dose-ratio model can trigger the
prior overdose screen before enrollment. That outcome is retained as a valid
trial result; the simulator does not modify the prior to force enrollment.
Posterior MCSE and split-R-hat summaries are diagnostic estimates only and do
not guarantee convergence. Aggregate patient, likelihood-evaluation and work
limits are checked/enforced across the request. Arrival-schedule exhaustion
raises rather than becoming a completed truncated replicate. The 30,000-trial
paper run is not an appropriate default for the repeated-MCMC Python workflow.

This is the paper's BF-BLRM numerical study as a configurable community Python
workflow. The cached BARD app guide itself describes BF-BOIN in stage one;
this feature does not claim the app exposes BF-BLRM simulation or that its
hidden defaults, calendar, joint outcome law, posterior stream, or tables are
reproduced byte for byte.
