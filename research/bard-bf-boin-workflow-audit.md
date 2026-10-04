# Integrated BARD BF-BOIN workflow

The current BARD application guide uses BF-BOIN for stage one. Its cached
Section 2 (printed pages 12–14) accepts scenario toxicity/response rates and
categorical factor inputs, then reports operating characteristics. The paper's
stage-two section defines the selected dose pair, eligibility-filtered
carryover, the combined sample-size target, minimization, and final OBD rules.
Its simulation section (printed pages 18–19) reports enrollment, duration,
arm-count imbalance, factor-proportion imbalance, and correct OBD selection
under both noninferiority and utility. Source versions and hashes remain in
[the provenance record](../docs/bard-sources.json).

The new trial runner composes the existing BF-BOIN calendar, response-model
calibration, minimization, and final-selection implementations. A generated
patient retains the actual profile and outcomes through the stage transition.
Only patients both clinically eligible under the supplied profile policy and
treated at one of the chosen doses enter carryover. All such patients remain
included, even if they already exceed the combined target. The stage-one
cumulative elimination mask prevents a later dose-pair choice from reopening
a dose eliminated earlier in the calendar.

The streaming wrapper accepts user-defined scenarios rather than requiring a
published three-factor scenario. Both OBD methods use caller-supplied true dose
indices. Unconditional correct-selection probabilities use every simulated
trial, including trials without an MTD or OBD. Selected-only accuracy is
reported separately. Arm-count imbalance includes empty-arm trials when a
usable pair exists; factor-proportion imbalance requires both arms to contain
patients and reports its own denominator. Every modeled factor is summarized,
including factors omitted from the configured minimization subset.

## Explicit policies and limits

The paper and guide do not fully specify the native joint endpoint generator,
calendar convention, or interpretation of per-arm enrollment targets. Python
uses an explicit joint profile distribution, an optional dose/profile table
of joint toxicity-response probabilities, and conditional independence when
that table is absent. Stage two begins after complete stage-one follow-up,
uses the configured renewal arrival law, and targets the paper's combined
eligible count. Stage-two profile weights are conditioned on the eligibility
mask. These choices do not claim hidden native defaults or random-stream
equivalence.

The wrapper retains one trial at a time. The existing stage-one record cap
and the stage-two target cap each remain 1,000; an aggregate patient-work
budget rejects oversized requests before draws. A five-dose design with 30
escalation patients, a backfill cap of 12 per dose and a combined stage-two
target of 40 has a conservative 130-patient bound per trial. The paper's
30,000-replicate scale therefore fits within
the default 100,000,000-patient-work budget without collecting all histories.
No large local simulation was used for validation.

No further advertised statistical calculation was identified in this bounded
cached-source review. A portable saved BARD protocol/OC report remains open,
and the source-contract choices above remain explicit. This checkpoint does
not promote the catalog to complete or claim native application parity.

## Validation

Fourteen distinct focused checks pass: nine affected BF-BOIN simulation/profile
checks, three complete-trial checks and two OC reducer checks. The profile
checks include a joint-probability rejection against canonical probabilities
when a manually constructed model contains a slightly perturbed saved table.
Scoped Ruff, formatting and mypy checks pass. Ordinary escalation, expansion
and titration examples reproduce every legacy result field and the subsequent
RNG draw from published base `06e86545` when the response model is omitted.

An independent four-trial audit reconstructed 55 patient records, all 20 new
stage-two minimization decisions, four-cell outcome tables, inherited joint
response probabilities and final posterior utilities. Maximum utility
difference was `1.42e-14`. The audit used three modeled factors and balanced
only two. Its process peak was 133.16 MiB with zero reported swaps. The trial
worker reported 142.88 MiB for its focused tests; the OC worker's 100-trial
guide example reported 131.34 MiB. These are observed small-run process peaks,
not global memory guarantees.

A separate read-only source/code review found no material issue in eligible
carryover, persistent elimination, NI with zero carryover, shared endpoint
truth, all-factor summaries or no-selection denominators. Numerical work was
serialized with single-thread BLAS; no full local test suite was run.
