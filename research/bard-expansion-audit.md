# BARD BF-BOIN expansion source audit

## Source-backed conduct

The cached author guide (`research/raw/BARD/Guide.pdf`, text in
`research/raw/BARD/Guide.txt`) says backfilling ends when escalation ends by
default. Its separate `Expansion` help sheet says that when this option is
unchecked, let `c` be the dose of the latest dose-escalation cohort and treat
additional patients at dose `c - 1` until either the assigned count there
reaches `n_cap` or that dose closes for toxicity. The local help PDF SHA-256 is
`00bcf52267a653b36e6a7ef6b51103442667cbfd23f2c5c30dde409e2cbeb8f2`.

The primary paper's BF-BOIN section adds the existing eligibility condition:
at least one response must be observed at dose `b` or a lower dose. The
backfill cap counts assigned patients, including escalation and backfill
patients. The existing `BFBOINDesign.backfill_eligibility` already implements
that response condition, the per-dose cap and toxicity closure. Expansion
therefore reuses it and does not target a different dose if `c - 1` closes.

## Python calendar choices

The author sources specify the expansion target and stopping rules, not event
ordering, arrival timing, or how to represent an activity-ineligible target
with pending responses. The simulator uses its existing renewal-arrival
process, processes assessments before tied arrivals, waits only for pending
observations that can update eligibility or toxicity closure, and consumes
intervening arrivals rather than moving the clock directly past them. It then
either starts expansion or returns `activity_unavailable`. Because this
simulator observes response at the same full window used for a negative DLT,
activity at `c - 1` is ordinarily known when its escalation cohort has
completed; the delayed-activity branch remains correct for any pending
response records already in the ledger. Safety stopping remains terminal; a
precision stop or configured escalation-cohort exhaustion can transition into
expansion. These are documented Python calendar decisions, not native timing
parity.

The continuation is specifically the BF-BOIN guide feature. No corresponding
BF-BLRM expansion rule was located, and the feature is not applied to
BF-BLRM. BARD's primary paper describes a combined stage-two total target but
does not specify assessment delays, interim stopping or a stage-two calendar;
this audit makes no such extension.

## Validation

Focused tests cover dose targeting at the last actually treated escalation
dose even when its post-cohort recommendation points higher, assigned-cap
accounting, activity-unavailable and no-lower-dose outcomes. The no-expansion
default is intended to leave earlier simulator draws and results unchanged;
the root integration check compares those outputs against its published
baseline fixture. No broad test suite or native application execution is part
of this increment.
