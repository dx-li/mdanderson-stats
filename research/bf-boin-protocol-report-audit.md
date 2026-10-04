# BF-BOIN operating-characteristic report audit

## Advertised outputs and existing implementation

The cached user guide `research/raw/BF-BOIN/Guide.txt`, page 11, says to enter
per-dose true toxicity and response probabilities and run an operating
characteristic simulation. Figure 15 on printed page 12 shows columns for the
per-dose truth vectors, dose-specific selection percentages, `% Pts treated`,
total number of patients, `% Early Stopping`, and trial duration in months.
The image is embedded in the cached `Guide.pdf`; those labels are not present
in the extracted text. The independent cached CRAN R implementation documents
the same named summary categories in `research/raw/BF-BOIN/bf_boin_oc.R` around
lines 542–548, but is not treated as the MD Anderson app backend.

`BFBOINSimulation` already contains the inputs needed to compute Python
summaries: `selection_probability`/`selection_mcse`, `mean_assigned`,
`mean_patients`, `mean_toxicities`, per-trial stop reasons, `trial_duration`,
and expansion/titration outputs. No additional operating-characteristic
calculation kernel is introduced. The report preflights its sum-of-scenario
record bound, runs serially, and drops each simulation object after distilling
it to compact tuples.

The native figure does not reveal whether `% Pts treated` is
`E[N_j]/sum_k E[N_k]` or `E[N_j/N]`; the report explicitly labels its choice as
the ratio of mean assigned counts. Similarly, although the displayed
`% Early Stopping` happens to complement the displayed selection percentages
in the pictured scenarios, the source does not define that identity for all
trials. The report shows no-MTD probability and stop-reason frequencies
separately instead of claiming an inferred formula.

## Primary-source and execution differences

The BF-BOIN primary method contract, pending-outcome rules, pooled conflict
movement, expansion target, and comparison with the independent CRAN package
are documented in `docs/bf-boin-reference.md`. The user guide's accelerated
titration rules are in Remarks 2 (printed page 2); the fixed grade-2 probability
and delay are explicit Python inputs because the guide does not specify a
probability or assessment-time model. In the Python simulator grade-2
probability is conditional on no DLT.

The BF-specific core now implements the Guide's optional one-DLT-of-three
current-dose stay modification at target 0.25, strict extra-safe `n > 3`
stopping, and strict `<` isotonic bound for final MTD selection. The guide does
not spell out how the 1/3 modifier composes with a conflicting backfilled dose;
Python applies it to the individual action before the otherwise unchanged
conflict-pooling rule, and does not claim native parity for that interaction.
The ordinary BOIN implementation retains its separate thresholds and options.
The asynchronous Python expansion timing remains an explicit policy rather
than a recovered native calendar rule.
