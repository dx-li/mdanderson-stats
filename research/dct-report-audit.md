# DCT report source audit

The cached DCT Shiny page is pinned in `docs/dct-sources.json`. Its result
panel (cached `research/raw/DCTs/index.html`, lines 1070–1087) contains dynamic
placeholders for sample size, caption, table and citation, but does not include
their server-side rendering rules. The help example around lines 1123–1145
does define the user-facing summary fields: total sample size, control and
experimental allocation, target power, significance level and a weighted
z-test description.

The report captures the existing `DCTNormalSampleSize` fields
`unrounded_total`, `allocation`, `achieved_power`, `target_power`, `alpha` and
`sides`. `dct_binary_sample_size` returns that same result type. The report
requires the caller to label allocation as participants or clusters because
the planner's repeat parameters can represent repeated measurements or cluster
sizes. It does not infer participant counts from a cluster allocation.

The source-help continuous example with effect 10 and SD 20 lists 128 total
participants; the existing Python equation and independent cellwise ceiling
produce 126. The report displays the computed Python allocation without a
native-matching adjustment and calls out the difference only for that exact
example input. No sample-size formula or rounding behavior is changed by this
workflow.
