# BARD BF-BLRM calendar contract

The [model and decision audit](bard-blrm-audit.md) records the primary paper's
rules and ambiguities. This replay targets the full stage-one calendar with
explicit arrival-by-dose potential outcomes and assessment delays, while
retaining caller-specified priors and posterior sampling precision. It does
not invent a toxicity onset or response timing distribution.

All assessments available at a time precede allocation at that time. Positive
DLTs may complete early; pending nontoxic patients cannot enter the likelihood
before their full DLT window. Response timing is separate. A fit is reusable
until its evaluated/DLT count vectors change. Escalation cohorts have priority
over backfill; waiting for every current cohort member's DLT assessment is an
explicit Python timing convention. The backfill cap counts evaluable patients,
so several outstanding patients may be assigned before it closes.

Initial prior screening, permanent no-MTD handling after a boundary stop and
an explicit `raise`/`stop` policy for unresolved equality or unsafe intermediate
steps are Python policies. They must be visible in the API and documentation;
none is claimed to resolve the paper's omissions. All enrolled patients finish
follow-up after enrollment ends. No later posterior recovery reopens a stopped
trial or silently restores its MTD selection.

## Independent calendar references

`tools/reference_bard_blrm_calendar.R` derives fixed-prior posterior indicators
with base-R `plogis`. It uses hand-enumerated assignment ledgers rather than
duplicating an adaptive simulator. Its three cases cover staggered cohort
completion and simultaneous assessments, pending backfill assignments beyond
the evaluable cap, early DLT completion, delayed response follow-up and arrival
tape exhaustion with a partially filled cohort.

The main case has 11 assigned patients, including three backfill patients,
six declined arrivals, eight escalation assignments and final MTD dose two.
Enrollment ends at time eight and follow-up at time ten. The other cases
verify that early DLTs can complete a cohort before the window and that late
responses still contribute to total follow-up duration.

The R generator completed with warnings treated as errors, producing 16
patient rows, 129 as-of count rows, three fixed posterior rows and three
trial summaries. The completed Python replay matches all three hand-ledger
paths, all 16 assigned-patient records and the as-of evaluable, toxicity and
response counts at every recorded event. Fixed-prior target and overdose
indicators agree exactly. The paths require 12, 5 and 2 fits respectively,
including their prior-only fits; response-only assessments reuse the fit.

The bounded comparison also verifies separate rejection of a positive
response delay lost at a large calendar origin, preservation of the declared
three-unit duration at origin `1e16`, and a prior all-overdose stop using only
its initial prediction budget. It took 0.024 seconds after imports, with
113.94 MiB peak process RSS and zero reported swaps.

Luna implementation checkpoint `f2e53d1` was integrated as `b05c730`.
Four focused tests passed in 1.43 seconds; module lint, formatting, typing,
compilation and whitespace checks passed. Earlier model integration
references validate the nondegenerate posterior sampler separately; the
calendar references deliberately fix the prior to isolate scheduling from
Monte Carlo variability. These checks do not establish native simulator or
random-stream equivalence.

A separate three-patient replay with two free prior coordinates also matches
four sequential calls to the actual fitter exactly under the same random
stream: posterior probabilities, parameter means and cumulative work counters
agree. Response-only events leave that stream untouched. This checks the
calendar/fitter boundary without substituting a mock posterior.
