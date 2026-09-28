# BARD accelerated-titration replay boundary

## Review and bounded validation

Review identified and corrected singleton-cohort event ordering: ordinary
cohort decisions must wait while titration remains active, and a grade-2-only
exit after DLT assessment must immediately make an eligible cohort decision.
The latter uses the current posterior without an extra fit. The reproduced
calendar now assigns patients at times 0, 2 and 4 and uses exactly four fits
(prior plus three DLT updates). Safety boundaries also cover titration and
top-up roles. A separate source review found no remaining material allocation
or posterior-logic issue.

Nine focused replay/titration tests pass in 1.53 seconds, with targeted Ruff,
formatting and mypy checks passing. Root executes the titration, ordinary
calendar and stage-two continuation guides together. Their three examples pass
in 1.492 seconds including imports, at 114.73 MiB peak RSS with zero swaps.
The titration example matches the independent hand ledger: doses 1, 2 and 3
at times 0, 2 and 4, then a same-dose top-up at 4.5 before the previous patient's
assessments. Existing model references continue to validate the reused
BF-BLRM posterior. No new CI workflow or large trial simulation was run.

## Source and scheduling contract

The cached BARD guide, `research/raw/BARD/Guide.txt`, Remarks 3 (pp. 3–4), describes a one-patient-per-dose-level accelerated titration. It ends titration at the first DLT, the second grade-2 toxicity among titration patients, or the dose cap. Reaching the highest dose itself exits titration and starts a top-up of `cohort_size - 1` patients at that dose. Reaching a lower cap without a toxicity trigger advances to the next dose and starts a full cohort there. Backfill is not part of titration.

`run_bard_blrm_trial` accepts explicit per-arrival/per-dose grade-2 outcomes and assessment delays when `accelerated_titration=True`. Titration proceeds sequentially: only one titration patient can be unresolved, and advancing to another level waits for that patient's DLT and grade-2 assessments. A DLT or second grade-2 event exits as soon as observed; a positive DLT need not wait to the end of the DLT window. The highest-dose reach branch starts top-up at the next supplied enrollment arrival without waiting for that patient's assessments. These are explicit Python calendar-scheduling conventions; no native timing distribution or native calendar parity is claimed.

If highest-dose reach and the overall escalation-patient cap coincide, the overall cap takes reporting precedence: no top-up is enrolled, and `titration_exit_reason` is `escalation_patient_cap`. While one titration patient remains unresolved, intervening supplied arrivals are declined with reason `titration_assessment_pending`; there is no retroactive assignment. Top-up patients enter the ordinary escalation cohort and count toward the escalation-patient cap, but their grade-2 outcomes do not contribute to the titration trigger count. Once titration exits, BF-BLRM cohort decisions and response-based backfill follow the existing replay. The existing BF-BLRM unsafe-dose boundary and permanent all-overdose stop can preempt the guide's titration sequence; this composes the BARD sequencing with BF-BLRM policy and is not native BF-BOIN titration parity. With `accelerated_titration=False`, grade-2 inputs are not accepted and the existing replay path is unchanged.
