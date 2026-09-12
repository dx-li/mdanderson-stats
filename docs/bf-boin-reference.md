# BF-BOIN reference audit

This audit records the public BF-BOIN contract and bounded numerical
references.  The executable reference is the CRAN package `bfboin` 0.1.1,
which is an independent GPL-3 implementation by Magirr and Zhang; it is not
the source or backend of the MD Anderson Shiny application.  The app landing
page identified in this audit is BF-BOIN V1.0.14.0 (updated 2026-08-04), by
Yixuan Zhao, Ying Yuan, and Ying-Wei Kuo.

Primary sources:

- Zhao, Yuan, Korn, and Freidlin, “Backfilling Patients in Phase I
  Dose-Escalation Trials Using Bayesian Optimal Interval Design (BOIN),”
  *Clinical Cancer Research* 30 (2024), 673–679,
  [DOI 10.1158/1078-0432.CCR-23-2585](https://doi.org/10.1158/1078-0432.CCR-23-2585).
- [BF-BOIN app](https://biostatistics.mdanderson.org/shinyapps/BF-BOIN/) and
  its [Guide.pdf](https://biostatistics.mdanderson.org/shinyapps/BF-BOIN/Guide.pdf).
- App help: [Backfilling.pdf](https://biostatistics.mdanderson.org/shinyapps/BF-BOIN/Backfilling.pdf),
  [Safety.pdf](https://biostatistics.mdanderson.org/shinyapps/BF-BOIN/Safety.pdf),
  and [Expansion.pdf](https://biostatistics.mdanderson.org/shinyapps/BF-BOIN/Expansion.pdf).

## Conduct rules

The primary paper opens a lower dose for backfill when a response has been
observed at that dose or below it. It closes a dose for toxicity only when both
its own evaluable DLT rate and the pooled evaluable rate at that dose and the
next dose exceed the de-escalation boundary. This also closes higher backfill
doses. Such closures can reopen when later data no longer meet these conditions;
posterior safety elimination is a separate rule. Pending outcomes must not be
counted as completed non-DLTs.

The assigned count (escalation plus backfill) closes backfill at `n_cap`, but
does not prohibit later escalation-component enrollment at that dose. The CRAN
source instead tests `assigned <= n_cap` before assignment and can exceed that
cap by one. It also gates response at the exact dose, unlike the paper's
at-or-below rule. The Python implementation follows the primary paper on these
points. Highest-open allocation is the paper's default; lowest and random are
discussed alternatives.

The current escalation cohort is not used for a movement decision until all
its DLT assessment times have passed the arrival clock.  Backfill DLTs are
likewise counted only after their assessment time.  The CRAN source calls the
resulting count “recruited” in a comment but actually uses completed
backfill observations (`calendar < clock`) for that count.  An implementation
must preserve the app's assigned-count cap separately from this pending-outcome
detail.  Its final aggregation also excludes backfill events whose assessment
is after the last escalation-cohort assessment, so the CRAN output is an
evaluated-data estimand rather than an all-recruited-patients tally.

After a complete escalation cohort, the paper compares BOIN actions at the
current dose and doses that have received backfill. A lower-dose stay conflicts
with current-dose escalation; a lower-dose de-escalation or elimination conflicts
with every current-dose action. Pool from the highest conflicting backfilled
dose through the current dose. Escalate for pooled rate at or below the escalation
boundary, and de-escalate for pooled rate above the de-escalation boundary.
For de-escalation, search cumulative pools from that starting dose through each
lower candidate dose; choose the highest candidate with rate at or below the
de-escalation boundary, or move below the pooling start if none qualifies.
The CRAN implementation uses different strictness at exact boundary equality.
These distinctions preclude treating its trial outputs as exact Python parity
fixtures. The app also exposes optional 1/3-stay and 2/6-de-escalation modifiers.

The app's optional early stop is “assigned patients at the current dose >=
`n_stop` and the next action is stay.”  The optional extra-safety rule requires
more than three patients at dose 1 and `Pr(p1 > target) > P_E - delta` in the
guide.  The CRAN implementation uses `n >= 3` in its corresponding conditional;
this is a concrete backend/reference discrepancy at the boundary and should
not be silently normalized.  Final MTD selection uses all observed dose data,
the BOIN elimination rule, and optionally requires the isotonic estimate at
the selected MTD to be below the de-escalation boundary.

If backfill is allowed after escalation ends, the app expansion rule treats
patients at one dose below the latest escalation dose until that dose reaches
`n_cap` or is closed for toxicity.  The CRAN `get.oc.bf` implementation rejects
`end.backfill = FALSE`, so its operating-characteristic function cannot validate
that app mode.

## Fixtures and provenance

`tools/reference_bf_boin.R` runs three deterministic `sim.one.trial` cases and
writes:

- `bf-boin-trials.csv`: complete dose-level counts, selected MTD, seed, and
  simulation settings.  The first two cases contain more than the escalation
  cohort count at lower doses, demonstrating backfill; the third contains an
  upper-dose toxicity conflict.
- `bf-boin-boundaries.csv`: BOIN boundaries used by the reference package at
  target 0.25.

The runner asserts `bfboin` 0.1.1 and `BOIN` 2.7.2 and checks `0 <= y <= n`.
It uses seeded Weibull DLT times and response probabilities of one; these are
simulation fixtures, not claims about a native app random-number stream.

Reference artifact hashes (SHA-256):

| Artifact | SHA-256 |
| --- | --- |
| `bfboin_0.1.1.tar.gz` | `cea68ddca9d660d70c8d831727ca6bc3ad09d48c748c346c341853cce9683c05` |
| app `index.html` | `d17147594ddaa0c7ee21d170b467118b5deed6fccf0d7cae29108f1bd912cc38` |
| app `Guide.pdf` | `65b01bbaf620d92ac03222a3dffb1f1f755808e5eb79e606d6356da4f0257e7d` |
| app `Backfilling.pdf` | `80edfd43fe7b70fb1f61597f6938e11bde607fe5b909e7719f474c5969654b19` |

These hashes were recorded during the original audit. The temporary source
worktree was lost after a machine crash; the committed generator and CSVs
survived. The CRAN source archive was subsequently restored to ignored
`research/raw/BF-BOIN` and its SHA-256 verified against the recorded value.
The app snapshots have not yet been restored.
