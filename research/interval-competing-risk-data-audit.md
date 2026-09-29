# Interval competing-risk visit conversion audit

The converter in `interval_competing_risk_data.py` prepares longitudinal visit
records for `fit_interval_competing_risk`. It is a data transformation; it does
not fit a model, estimate visit-time distributions, infer causes, or add
left-truncation support.

## Pinned source behavior

The source is `intccr` 3.0.4 at commit
`252644c0d347a663ea5d7bef88fa2ab04f114b1e`. The relevant files are
`R/dataprep.R` (blob `371ac55900e8434a264c557e1933b749f3918576`) and
`R/Surv2.R` (blob `c32f52d7c8c275cda51dd23e9868d9d7e2615fd9`). The helper emits
one record per ID. A singleton event creates `(0, visit_time]`; a singleton
censor creates `(visit_time, Inf)`. With repeated visits, every status-0 row
updates the lower endpoint, and the first status-1/2 row supplies the upper
endpoint and event cause; later rows are ignored. If no event occurs, the last
visit is the right-censoring time. Covariates come from the first row selected
for that subject. Rows with missing visit times are dropped, and the final
`na.omit` drops constructed records containing missing values.

## Explicit Python conventions and source corrections

The source's `order(data[, ID] & data[, time])` uses logical AND rather than
lexicographic ID/time ordering. The Python converter sorts by ID and then visit
time, and rejects tied visit times because the source does not establish a
stable event precedence for ties. Numeric IDs and string IDs are supported
separately; the string case is an extension because R's logical-AND ordering
is not defined for character IDs.

The source handles a singleton first-visit event as `(0,t]`, but a first event
in a multi-visit history leaves the lower endpoint missing and is then removed
by `na.omit`. Python consistently uses `(0,t]` for a first-visit event. This is
a correction to the source defect and follows the documented interval meaning.

Missing-time row indices and post-event row indices are returned so exclusions
are visible. IDs and status codes through the first event are validated;
post-event status codes are not interpreted (the input remains a real numeric
vector). Baseline covariates are taken from the chronologically earliest
retained visit and must be finite there. The supplied covariate array must be
real numeric; later values are otherwise ignored, matching the source's use of
one row per subject. They are not interpreted as time-varying covariates. The
result retains source row indices for the lower boundary, upper event, and
baseline covariates.

`Surv2.R` rejects `v >= u`, so zero-time or otherwise zero-length event
intervals are rejected. Causes must be coded 0, 1, or 2. The conversion helper
can return data with one observed cause; the existing fitter independently
requires both causes and remains responsible for that model-level check.

## Scope

The public preparation function is
`prepare_interval_competing_risk_visits(subject_id, visit_time, status,
covariates=None)`. It returns read-only arrays aligned by sorted subject ID,
plus zero-based provenance/exclusion indices. IDs are limited to homogeneous
finite numeric or string values. Visit rows, covariate columns, and total
covariate cells are bounded before numeric conversion. Numeric list IDs retain
their original scalar identity before conversion, so adjacent values above
2^53 cannot collapse; mixed numeric/string IDs are rejected. Nested two- and
three-dimensional row containers are rejected before numeric materialization.

## Validation record

The six focused converter tests pass, including chronological first-event
conversion, censoring, missing-time and post-event provenance, tied/zero-length
interval rejection, numeric-ID precision, mixed-ID rejection and nested-shape
preflight. They ran in 1.39 seconds with a child peak RSS of 134,135,808 bytes
and zero swaps. Ruff format/check and mypy for the new module pass.

`tools/reference_intccr_visits.R` verifies the pinned source hashes and was run
against the cached author files. Its sorted/unsorted `dataprep` outputs and
`Surv2` outputs (including the exact-event error) matched the checked-in native
fixtures byte-for-byte. The run took 0.275 seconds, peaked at 87,162,880 bytes
child RSS and reported zero swaps. The corrected Python ledger is a separate
explicit expectation: it sorts chronologically and retains the first-event
subject that native `dataprep` accidentally drops.

Root integration reconstructs the documented 120-subject two-cause study from
203 visit records supplied in reverse order. The converter recovers all lower
and upper endpoints, causes and baseline covariates exactly; the resulting
arrays fit successfully through the public competing-risk API. This check,
together with bootstrap replay and unit checks, takes 1.365 seconds, peaks at
120.12 MiB RSS and reports zero swaps. The twelve affected focused tests also
pass together (2.437 seconds, 140.95 MiB RSS, zero swaps).
