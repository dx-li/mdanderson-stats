# BARD BF-BOIN six-patient modifiers and remaining full-trial OC workflow

The cached official BARD guide resolves both optional small-sample BF-BOIN
modifiers. `research/raw/BARD/Guide.txt`, Remarks 4 (pp. 4–5), says the 1/3
option is for target DLT rates 0.20–0.279 and sets the action at exactly three
patients to escalation for 0/3, stay for 1/3, and de-escalation for at least
2/3. Remarks 5 (p. 5) says the 2/6 option is for targets 0.28–0.33 and sets
the action at exactly six patients to escalation for at most 1/6 and
de-escalation for at least 2/6. These are BARD BF-BOIN rules; they do not
change the ordinary `BOINDesign` contract. `BFBOINDesign` now exposes both
flags and the reports record them. For either option, Python applies the
individual action before existing backfill-conflict pooling. The guide does
not state how the modifiers compose with backfill conflicts, so that ordering
remains an explicit Python policy rather than a native-parity claim.

The separate [BARD workflow audit](bard-remaining-simulation-audit.md) records
the now-completed BF-BOIN two-stage operating-characteristic workflow, saved
studies and reports, plus the paper's BF-BLRM stochastic route. Native quota,
calendar and source-prior limitations remain explicit.

## BF-BOIN completion review

The October 4 review cross-checked the cached BF-BOIN guide's Figure 15 against
the calendar simulator and saved protocol report. Dose truth, selection,
treatment share, total enrollment, early-stopping summaries and duration are
available, with explicit Python denominators and Monte Carlo errors. The
guide does not define a different native aggregation formula that could be
implemented from the inspected evidence. The recovered decision, boundary,
final-selection, titration and simulation methods are also covered.

The modifier convention applies to each dose's hypothetical individual
action, including lower doses with backfill, before conflict detection and
pooled-data resolution. The guide phrases the options at the current dose
and does not define this composition. This remains an explicit Python rule,
not a claim about hidden app ordering. Native report layout, random streams
and undefined aggregation choices are compatibility boundaries. No additional
source-defined calculation was identified; the recovered BF-BOIN workflow is
functionally implemented.
