# CRM saved patient histories

`read_crm_desktop_history` opens populated CRM Suite and BMA-CRM desktop HTML
files. Supply the actual date order explicitly: `mdy`, `dmy`, or `ymd`. Dates
must have four-digit years. Python dose indices are zero-based.

```python
from datetime import date
from pathlib import Path
from mdanderson_stats import read_crm_desktop_history, recommend_crm_desktop

history = read_crm_desktop_history(Path("my-crm-study.htm").read_text(), date_order="mdy")
result = recommend_crm_desktop(history, at=date(2026, 10, 12), seed=2026)
Path("crm-decision.html").write_text(result.to_html())
if result.can_add_patient:
    history = result.add_patient(patient_id=101, fingerprint=101)
    Path("crm-updated.html").write_text(history.to_html())
```

Patient IDs and fingerprints must be unique. `CRMDesktopPatient` records are
immutable. To record an assessment, use `dataclasses.replace` on a patient and
`history.with_patients(tuple_of_updated_records)`. Outcomes are -1 pending,
0 no DLT, and 1 DLT. Pending assessments precede the evaluation-window endpoint;
a no-DLT assessment reaches it; a DLT occurs within it. Treatment dates must be
chronological. Invalid or ambiguous histories raise before inference.

The as-of snapshot excludes future treatments. For retained patients the earliest
of toxicity, window end, decision date and last assessment determines the outcome
and follow-up. Ties prefer toxicity, then window completion, then decision time,
then assessment time. **A later decision date never supplies an unrecorded
assessment.** Correct the history when a new outcome is actually known.

The conduct controller checks current safety and waiting first. If treatment is
allowed in a partial strict cohort, it recalculates the cohort's entry decision
at its first treatment date, using corrected preceding patient records. The
report exposes both analyses. At a new cohort, escalation is capped using actual
observed raw toxicity rates after pending-outcome consensus. At maximum enrollment,
pending DA studies use complete-outcome lookahead for final selection too.
Historical analyses cannot add patients before retained assessments or after
capacity. Patient addition returns a new history with a same-day pending record;
it does not save or transmit anything automatically.

## Inference and evidence boundaries

The controller uses independent Python integration and DA sampling. BMA lookahead
enumerates bounded complete-outcome counts; exceeding `max_completions` yields
an explicit wait. Evaluation-budget exhaustion raises. DA `draws`, `warmup` and
`chains` are explicit Python settings, with MCSE and split R-hat in the report.
Native completion-bound shortcuts, adaptive MCMC and random streams are not
claimed equivalent. Very small sampler settings test plumbing only.

The supported saved policies are median skeletons (`PriorsAre_P=1`), strict
cohorts and the three-patient final MTD rule. Original CRM Suite GUI alternative
escalation controls are hidden (`MainFrm.InitializeComponent`, RVA 0x12b5c,
IL19991..20008); these are not advertised working design options. Unsupported
saved policy values fail explicitly.

Both desktop versions produce the same 142 bounded original-instruction fixtures
and four synthetic managed serialization files. See the
[reference harness](../tools/crm_native_reference/README.md). These establish
specific source contracts, not complete native stochastic equivalence or clinical
validation. The [simulation guide](crm-desktop.md) covers prospective scenarios;
simulation does not resume an imported partially accrued trial.

For the guide's delete-last or delete-all operations, use
`history.with_patients(history.patients[:-1])` or `history.with_patients(())`.
These return new objects. For DA diagnostic exploration, repeat
`recommend_crm_desktop` on the same history/date with explicit seeds and sampler
settings, inspect `result.analysis.posterior.parameter_summary`, and compare the
returned actions. Increasing draws is not a guarantee of convergence. Native
hidden MCMC controls remain preserved metadata; the Python settings and diagnostic
results are recorded in each report.

A final pending-outcome consensus retains `high_uncertainty` whenever any
completion selects the dose without the three-subject qualification. Agreement
on a dose does not remove that qualification. The recommendation and printable
report expose it, in addition to the full completion witnesses.
