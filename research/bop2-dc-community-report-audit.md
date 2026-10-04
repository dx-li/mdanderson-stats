# BOP2-DC community report source crosswalk

The report module is a presentation adapter over this repository's supported
numerical APIs, not an independent implementation of BOP2-DC. The cached
source material available for this task included the BOP2-DC paper text/PDF
and the existing source/reference notes, but no capture of the BOP2-DC app's
generated-report screen, report file, or save schema. Therefore this adapter
does not claim native report parity or attribute unobserved input controls to
the app.

| Report family | Existing source/API reused | Captured in report |
| --- | --- | --- |
| Single binary | `BOP2DCDesign.operating_characteristics` | thresholds, prior, looks, futility parameters, exact stop and final go/consider/no-go probabilities, expected sample size |
| Single Normal | `simulate_bop2_dc_normal` | NIG prior, thresholds, looks, truth, seed, Monte Carlo actions and enrollment |
| Single survival | `simulate_bop2_dc_survival` | inverse-gamma prior, median thresholds, look schedule, truth, accrual/follow-up/arrival, seed, decision and calendar summaries |
| Paired exact | `BOP2DCPairedDesign.operating_characteristics` | endpoint mode, joint truth order, prior and cutoffs, exact outcomes and sample-size distribution |
| Joint categorical | `simulate_bop2_dc_categorical` | endpoint indicator matrix/category mapping, directions, any/all rule, arm allocation and priors when randomized, truths, terminal/per-look actions, master and trial seeds |
| Randomized binary | `BOP2DCRandomizedBinaryDesign.operating_characteristics` | both arm priors, assignment, thresholds/cutoffs, interim graduation setting, exact stop/graduate/final outcomes and expected enrollment |
| Randomized Normal | `simulate_bop2_dc_randomized_normal` | independent arm NIG priors, assignment, thresholds/cutoffs, truths, seed, Monte Carlo actions/enrollment and quadrature error |
| Randomized survival | `simulate_bop2_dc_randomized_survival` | arm priors, assignment, thresholds/cutoffs, truth, arrival/accrual/follow-up, seed, decisions and arm event/exposure summaries |
| Randomized paired | `simulate_bop2_dc_randomized_paired` | endpoint mode, arm Dirichlet priors, assignment, cutoffs/graduation, arm joint-category truths, terminal/per-look actions, child seeds and quadrature error |

Numerical evidence and method limitations for these APIs are maintained in
[`bop2-dc-source.md`](../docs/bop2-dc-source.md),
[`bop2-dc-reference.md`](../docs/bop2-dc-reference.md), and the focused
categorical audit. Simulation MCSEs are those computed by the corresponding
core function; exact-recursion cases intentionally have no MCSE.
