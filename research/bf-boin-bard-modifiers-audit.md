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
the still-missing source-advertised complete two-stage operating-characteristic
simulation and distinguishes that method work from unresolved native quota and
calendar conventions. This modifier change does not close BARD as a whole.
