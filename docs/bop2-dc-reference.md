# BOP2-DC numerical references

Primary method: Zhao, Li, Liu and Yuan, *Bayesian optimal phase II designs
with dual-criterion decision making*, Pharmaceutical Statistics 22 (2023),
605–618, [doi:10.1002/pst.2296](https://doi.org/10.1002/pst.2296).
The authors' [preprint](https://arxiv.org/abs/2112.10880), version 1 dated
20 December 2021, supplies accessible equations. It is distinguished from
the subsequently published article. The
[MD Anderson app](https://biostatistics.mdanderson.org/shinyapps/BOP2-DC/)
identifies version V1.0.9.0, updated 23 March 2026.

The two criteria compare the same treatment effect with a lower reference
value and a clinically meaningful value. Final decisions distinguish go,
no-go and consider. Interim decisions use sample-size-dependent cutoffs and
ordinarily permit only no-go or continue. Equality is not sufficient for
either strict final go or strict no-go. Threshold calibration is a separate
task from calculating these decisions for supplied parameters.

`tools/reference_bop2_dc.R` independently evaluates the binary Beta posterior
using base R. The five rows in `tests/fixtures/bop2-dc-binary.csv` cover both
interim outcomes and all three final outcomes. Every row includes the counts,
prior, clinical thresholds, tuning parameters, posterior tails and cutoffs.
The generator ran successfully with R 4.4.1. These are equation-level
references, not outputs from the Shiny app's backend.

The downloaded preprint is preserved in ignored `research/raw/BOP2-DC`.
Its SHA-256 is
`a344f088f564a218e29af1abb0c22e1799a344dba6820a41048769905705ad7a`.
Original paper content is not redistributed in the package.
