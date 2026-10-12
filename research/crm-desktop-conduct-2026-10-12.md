# CRM/BMA-CRM desktop conduct integration

This checkpoint adds populated native patient files, explicit locale/date parsing,
editing, as-of censoring, cohort-entry replay, conduct reports and visible saved
simulation parameters. It does not yet promote either catalog entry. The full
application crosswalk must distinguish the independent Python inference profile
from native completion-bound and adaptive-MCMC shortcuts.

## Recovered contracts

Both original desktop versions use the same saved CRMSimApp envelope. Synthetic
managed patient files establish US and GB short dates, OutcomeCode and patient
fields. The caller supplies date order; the reader never guesses. Original outcome
IL truncates pending follow-up at last assessment, even if a later decision date
is supplied. Outcome ordering is toxicity, completed window, decision, assessment
for exact-day ties. Future treatments are excluded.

The native conductor's incomplete-cohort helper counts trailing equal cohort
labels. Pending DA within a strict partial cohort retains the current dose and
turns its safety signal into WAIT. At capacity, final pending-outcome assessment
uses complete-outcome lookahead. The managed wrapper first keeps current safety/
wait decisions; if treatment is allowed in a partial cohort, it then recensors
preceding patients at the first current-cohort patient's treatment date and
recomputes the cohort-entry recommendation. It is not simply a last-dose hold.

Current GetDecision managed RVA is 0xe634, older 0xe620. IL346..588 controls the
replay gate; IL635..737 builds/reassesses the preceding prefix; IL801..820 replaces
the current dose/status. Native current raw-rate gating at 0x32eb3 acts after
lookahead using actual observed counts, only at a new cohort. The explicit
`raw_rate_policy="observed"` calendar option supplies this ordering to
`scheduling="native_desktop"`; the ordinary profile remains unchanged. A strong-
prior example distinguishes this from applying the cap separately to hypothetical
completions: the latter waits, while the original observed-data ordering treats.

CRM Suite's alternative escalation radio controls are hidden: MainFrm
InitializeComponent RVA0x12b5c, IL19991..20008 calls Visible=false on the group.
Visible workflows use median skeletons, strict cohorts and final minimum three.
Unsupported saved policy values still fail explicitly.

## Source evidence

[Research harness and checksums](../tools/crm_native_reference/README.md) records
both official archives and original library hashes. All 142 instruction cases
per version regenerate byte-identically: 60 outcome/censor, 34 native cohort/
safety, 48 managed cohort-replay gate. Four managed serialization files also
agree across both versions. Runtimes and skipped operations are declared.

The original GetDecision and outcome managed method bodies match across versions;
the native cohort-helper slice also matches byte-for-byte. Full native controllers
have relocation/call-address differences and are **not** asserted byte-identical.
No original binaries are redistributed, and existing license notices are retained.

Independent base-R posterior references regenerate within 1.07e-14 absolute
for 120 BMA-CRM rows and 1.00e-15 for 94 DA-CRM rows. These verify posterior
mathematics separately from the original branch probes. They do not establish
clinical validity, full Windows execution, or native random-stream identity.

## Remaining closure contracts

The older controller comparison now establishes identical bounded control and
arithmetic regions after explicit pointer relocation. The [guide crosswalk](crm-guide-scope-2026-10-12.md)
records the completed user workflows and independent inference profile.
Native shortcut behavior must either receive bounded source evidence or remain
an explicit independently validated inference alternative. Online calibration
(entry133) is separate: its Q equation, regression intercept and candidate
selection remain unresolved. ToxFinder's saved-prior execution mapping is next.

Final validation results are recorded separately in the checkpoint manifest.
