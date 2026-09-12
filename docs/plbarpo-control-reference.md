# PLBARPO control references

Primary source: [PLBARPO](https://biostatistics.mdanderson.org/shinyapps/PLBARPO/),
J. Jack Lee, Ying-Wei Kuo and Nan Chen, PID 1002, version 2.0.3.0,
updated January 6, 2026. The [official support guide](https://biostatistics.mdanderson.org/shinyapps/PLBARPO/PLBARPO.pdf)
allows entire-trial or concurrent control response calculations and specifies
the same posterior stopping comparisons as BARPO.

The PDF retrieved September 12, 2026 has SHA-256
`86d6267b285b5cff148e0069af96f730f06b621b00f03e4565c2985324359455`.
The PDF and HTML snapshots are retained in ignored `research/raw/PLBARPO`;
original assets are not redistributed.

The guide does not fully specify the boundary or data-availability conventions
for concurrent controls. The Python helper takes explicit half-open enrollment
windows and an observation cutoff. This is a documented input contract, not
a claim of recovered app-internal behavior.

`tools/reference_plbarpo_control.R` independently selects control observations
and integrates Beta densities in base R. Six control patients enroll at times
0 through 5; their responses are `(1,1,0,0,1,0)` and observation times are
`(0.5,1.5,2.5,3.5,6,5.5)`. Treatment windows are `[0,4)` and `[3,infinity)`.
The reference evaluates cutoffs 5.75 and 7, so one control outcome becomes
available between analyses. It verifies that the first treatment's closed
window remains unchanged while the second treatment gains that observation.

Treatment response counts are `(3,4)` and nonresponse counts are `(1,2)`.
All priors are Beta(1,1). Eight posterior comparisons in
`tests/fixtures/plbarpo-control-posterior.csv` cover both control modes and
both observation dates; selection counts are recorded separately in
`tests/fixtures/plbarpo-control-counts.csv`. These are independent mathematical
references, not captured app outputs.
