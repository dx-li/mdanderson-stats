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

`tools/reference_bop2_dc_grid.R` enumerates all 16 response paths for a
four-patient trial with an interim at two patients. Six cutoff pairs are
evaluated under futile and effective response rates. The resulting
`bop2-dc-grid.csv` includes go, total no-go, consider and expected sample
size, with all design inputs. This oracle uses path enumeration rather than
the Python recursion and distinguishes maximum-go and minimum-sample-size
objectives. Its base-R run also checks conservation of total probability.

`tools/reference_bop2_dc_paired.R` generates nine Dirichlet-marginal reference
cases for multiple efficacy and efficacy/toxicity. Cell order is both events,
first only, second only, neither. The four prior shapes are 0.25 each.
The CSV records each clinical threshold and both posterior tails for each
endpoint. Toxicity uses lower tails, with its clinically meaningful threshold
below its reference threshold. The cases cover all final outcomes and interim
continue/stop decisions. They validate marginal decision composition without
assuming independence of the paired endpoints.

The downloaded preprint is preserved in ignored `research/raw/BOP2-DC`.
Its SHA-256 is
`a344f088f564a218e29af1abb0c22e1799a344dba6820a41048769905705ad7a`.
Original paper content is not redistributed in the package.

## Time-to-event reference

`tools/reference_bop2_dc_survival.R` uses base-R gamma probabilities to
independently evaluate the exponential/inverse-gamma model in Section 2.1.3.
The prior shape and scale describe the exponential **mean**, while LRV and
CMV describe the **median**. Seven cases in
`tests/fixtures/bop2-dc-survival.csv` cover interim stopping, all three final
outcomes, zero events and the prior before enrollment. These are mathematical
reference calculations, not captured outputs from the app backend.

## Exact paired operating characteristics

`tools/reference_bop2_dc_paired_oc.R` enumerates all 256 four-patient outcome
paths with a look after two patients. It computes each path's Dirichlet
posterior decisions independently with base R, retaining the first stop.
`tests/fixtures/bop2-dc-paired-oc.csv` contains ten scenarios spanning both
paired modes, positive/negative association, independence and deterministic
outcomes. The three association scenarios have identical marginal event
rates of 0.5, so they also test that the Python recursion preserves the
supplied joint distribution. All inputs are explicit in the reference script.
