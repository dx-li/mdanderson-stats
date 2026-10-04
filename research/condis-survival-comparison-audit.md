# CondiS censored-versus-imputed survival comparison

The pinned CondiS 0.1.2 vignette (`research/raw/CondiS/CondiS/vignettes/introduction.Rmd`,
lines 42–72; archive SHA-256 is recorded in `docs/condis-sources.json`) shows
the base imputation followed by two `survfit` curves: original follow-up with
its event indicator, and imputed follow-up with every status set to one. It
combines those curves with censor marks and a risk table. This is an advertised
community workflow with an explicit method definition, unlike unobserved app
controls.

`condis_survival_comparison` composes the existing `condis_impute` result and
`exploratory_survival` Kaplan–Meier implementation. It makes the risk-table
times an explicit input, counts observations with follow-up `>=` each time,
and places censor marks at the right-continuous original-curve survival value.
The optional Matplotlib view is a Python visualization, not a claim of
`survminer` style or byte-level parity. The all-event imputed curve is a
descriptive display only; the vignette does not establish standard
censoring-based uncertainty for it, so no confidence intervals are reported.

The feature closes the cited vignette's comparison/risk-table workflow. It
does not complete CondiS Shiny upload/download behavior or establish a
leakage-free future-patient prediction contract for the later illustrative
CondiS-X example.
