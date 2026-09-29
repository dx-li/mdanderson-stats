# Dose Schedule Finder toxicity adjudication helper

## Source rule

Braun, Thall, Nguyen and de Lima (2007), [*Simultaneously optimizing dose and
schedule of a new cytotoxic agent*](https://odin.mdacc.tmc.edu/~pfthall/main/ClinTrials%20dose-sched%202007.pdf),
author-hosted paper, page 4 (printed page 116), specifies that a grade-2
toxicity that is not therapeutically resolved within two weeks of onset, or
one that necessitates dose reduction, qualifies as a dose-limiting toxicity
dated at its initial onset. The source does not provide a complete
patient-by-patient grade-2 generation or adjudication process for the Python
calendar replay.

The helper therefore accepts an onset and the caller's adjudication time and
decision. It does not infer persistence, dose reduction, a two-week clock, or
within-patient adaptation. Known qualifying onsets are backdated for the
likelihood; the actual administration ledger remains separate and retains
post-onset administrations that occurred by the requested as-of time.

## As-of information boundary

Episode onsets after `as_of` are omitted from returned statuses. Adjudication
times and decisions after `as_of` are masked as pending. A known qualifying
onset by the horizon is used as the event time, but unresolved earlier onsets
prevent `final_ready`. With no known event, final readiness requires horizon
follow-up and adjudication of every episode beginning by the horizon. Pending
episodes after the horizon can be reported without blocking risk-horizon
finality. These rules make the result usable for an interim fit while exposing
when later clinical adjudication could revise the analysis.

The returned `DoseSchedulePatient` follows the existing replay tie convention:
administrations strictly before a known event onset enter the event likelihood;
an administration tied with the onset is retained only in the actual-delivery
ledger. Censored records include administrations through their observation
end. This helper adds no stochastic low-grade-to-DLT process and no allocation
or dose-modification policy.
