# PoP protocol report source audit

The report wraps existing Python calculations rather than duplicating the
design or simulation. It validates and snapshots a `PoPDesign`, executes one
`simulate_pop` call per truth scenario, records the deterministic scenario seed
and request order, and retains compact summaries.

| Source-defined operation/output | Python source | Report field |
| --- | --- | --- |
| Full integer transition/exclusion cutoff table | `PoPDesign.boundaries(max_patients, cohort_size=1)` | Every count from 1 through planned maximum, with impossible values preserved |
| Dose selection including no admissible dose | `simulate_pop` | Probability and Bernoulli MCSE; vector order is no selection, then doses |
| Mean allocation and observed DLT counts, plus total means | `simulate_pop.mean_patients`, `mean_toxicities` | Per-dose tuples and their source-defined total means |
| All-dose exclusion early stop | `simulate_pop.early_stop_probability` | Probability and Bernoulli MCSE |
| Underdosing/overdosing relative to the true MTD | `simulate_pop.risk_under`, `risk_over` | Risk and Bernoulli MCSE using the requested `risk_cutoff` |
| True MTD definition | Native `which.min(abs(skeleton-target))`; Python `np.argmin` | First closest dose, including first-index ties |

The cached CRAN `get.oc.pop.R` returns selection percentage, mean patient and
toxicity counts, early-stop probability, and under/over risk. Its internal
`true.mtd` is the first minimum-distance dose. Risk events use strict
`>` against the configured fraction of planned enrollment. Although the
wrapper declares `risk.cutoff`, its call to the inner trial omits that argument;
the inner default is 0.8. This Python report captures and uses the explicit
Python cutoff rather than silently reproducing the wrapper omission. Risk
calculations are taken directly from `simulate_pop`, which supports non-exact
target skeletons; the unrelated shared BOIN/Keyboard helper has an exact-target
availability condition and is intentionally not used.

The manual describes early stop as occurring without MTD selection, but the
executable `trial` sets `early=1` when all doses are excluded, then attempts
the final selector after the trial loop. The report labels the Python metric
as all-dose exclusion stop rather than asserting that no final MTD was selected.

At cutoffs below 0.5, a trial can cross both directional allocation thresholds.
The native R code uses an `if/else if`, making those risk flags mutually
exclusive; the current Python simulation computes under- and over-risk
independently. They agree at the recommended/default cutoff 0.8, and the report
faithfully records the values the Python simulation actually computes rather
than claiming parity for low custom cutoffs. At exactly 0.5, strict greater-than
comparisons make simultaneous crossing impossible.

The native `select.mtd.pop.R` returns only target, selected MTD, and isotonic
estimates. Its executable `plot.pop.R` plots the estimates and target line; it
does not compute credible intervals, although the plot help text advertises
95% intervals. Cached app HTML has conditional `plus3` result panes but no
matching input control. Its comparison behavior is unresolved in the inspected
cache; no behavior is inferred in this report.

Final selection uses the executable selector's Beta(0.05,0.05) posterior means,
inverse-variance-weighted increasing isotonic fit, `1e-10` treated-rank
perturbation, closest admissible estimate, and higher-dose remaining-distance
tie rule. The safety screen is a separate Beta(1,1) posterior tail above 0.95,
after the configured minimum patient count, excluding the triggering and
higher doses.

Source anchors in the pinned cache: `research/raw/PoPdesign/source-1.1.0/PoPdesign/R/get.oc.pop.R`
(`true.mtd`, strict risk event and returned fields); `select.mtd.pop.R` (selector
return object); `plot.pop.R` (executable plotted fields); `man/get.oc.pop.Rd`
(reported OC list); `man/plot.pop.Rd` (interval wording); and
`research/raw/PoPdesign/app.html` (scenario simulation controls, risk cutoff,
conditional `plus3` panes and MTD report download button). No app server-side
implementation or native document/plot rendering is claimed.
