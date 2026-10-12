# CRM Suite saved studies and desktop scheduling

The Python workflow can now read an original CRM Suite 1.0.0 saved HTML study,
simulate its CRM/BMA-CRM or DA-CRM scenarios, and write a readable HTML report
with the native XML settings preserved. This is independent Python inference
with recovered desktop scheduling. It does not claim the same random sequence,
adaptive MCMC stopping, or pixel-identical report as the Windows application.

```python
from pathlib import Path
from mdanderson_stats import read_crm_desktop_html, simulate_crm_desktop_study

study = read_crm_desktop_html(Path("my-crm-study.htm").read_text())
print(study.skeletons, study.scenario_names)
# Workload and seed are explicit; saved studies may request hundreds of trials.
run = simulate_crm_desktop_study(study, trials=10, seed=2026)
Path("python-crm-report.html").write_text(run.to_html())
restored = read_crm_desktop_html(run.to_html())
assert restored.skeletons == study.skeletons
```

For DA studies, the Python sampler's `draws`, `warmup` and `chains` are explicit
options. The native file's adaptive-MCMC/thinning settings are retained as
metadata; they are not silently mapped to another sampler. Low draw counts are
smoke tests, not evidence of adequate Monte Carlo precision. The underlying
simulation returns per-trial MCSE and R-hat diagnostics.

## Saved settings and supported execution

The original HTML envelope contains XML between
`beginA18959FB-1E57-4CE0-AAB1-B038A7B78423` and the corresponding `end` marker.
The root is `CRMSimApp`, with matching `ProgID` and `FileVersion="1.0.0"`.
The reader discards original HTML/scripts and preserves XML settings semantically,
including unused probability slots, disabled skeletons, patient records, literal
patient date text, and unknown fields. It does not execute input HTML or XML
entities. Documents are limited to two million characters.

Only the first `NumSkeletons * NumDoses` probability slots define skeleton rows,
in skeleton-major order. The relevant DA/non-DA Boolean array selects active
rows. DA requires exactly one active row, which need not be the first stored row.
Scenario vectors and dose labels use their first `NumDoses` entries. Extra slots
remain in the saved XML. `study.to_html()` round-trips settings without simulation.

Execution currently accepts the recovered sample conventions: probability/median
skeleton input (`PriorsAre_P=1`), strict escalation, and the three-patient final
MTD rule. Other policy values raise instead of being ignored. `EvalTime` must be a positive whole number of days, matching the native
TimeSpan.Days conversion. All saved scenarios are simulated serially. The explicit trial count and seed replace their XML
values in the resulting report. The output exposes selection probabilities and
MCSEs, mean dose-specific enrollment/toxicities, no-selection/early-stop rates,
mean total toxicities, and trial duration. Native report months use exactly
30.4375 days; days are also shown.

Imported patient records are preserved, but this simulation evaluates a prospective
design, not the unfinished imported trial. Native dates use locale-dependent
short-date strings. They are not guessed or silently converted to a calendar.
For patient-level replay, use the explicit numeric-time [calendar API](crm-conduct.md).

Default total work across all scenarios is 20 million evaluations, configurable
up to 200 million. Exhaustion raises rather than returning a partial report.
The existing per-trial and per-decision bounds also apply.

## Native desktop scheduling profile

Both `run_crm_trial` and `simulate_crm` accept `scheduling="native_desktop"`.
Their default `"ascertainment"` preserves existing Python behavior.

For native replay, `dlt_delays` determines planned patient count;
`interarrival` is a **candidate-gap tape**, rather than one gap per patient:

1. The first patient arrives at time zero.
2. A cohort retains one dose. Consume one positive gap after every enrolled
   patient, including the last patient of the trial.
3. A wait consumes another candidate gap, and reassesses at that next arrival;
   it does not jump to the earliest outcome ascertainment.
4. Final assessment observes at the post-enrollment clock plus the window.
   Reported duration is separately the latest actual ascertainment among enrolled
   patients. It can precede the final assessment clock.
5. An early safety stop still follows enrolled patients to ascertainment, and
   does not reopen or select an MTD. Full-sample final safety nonselection does
   not count as early termination.

```python
import numpy as np
from mdanderson_stats import run_crm_trial

trial = run_crm_trial(
    [0.08, 0.2, 0.4],
    [0.25, 0.25, 0.75, 0.25, 10],
    np.full((4, 3), np.inf),
    1,
    target=0.25,
    cohort_size=2,
    safety_cutoff=1,
    max_completions=1,
    scheduling="native_desktop",
)
assert trial.enrollment_times.tolist() == [0, 0.25, 1.25, 1.5]
assert trial.final_time == 2.5
assert trial.decision_time == 12.5
```

Exhausting the supplied tape raises. Simulation creates a separately seeded
per-trial arrival tape; `max_arrival_gaps` defaults to 2,000 and permits
200–10,000. Changing that bound does not change potential outcomes or later
trial scenarios. This reproduces scheduling rules, not native RNG identity.

The recovered DA six-interval hazard prior uses Gamma rate .5 (dispersion2).
`dacrm_trimester_prior(..., dispersion=2, endpoint_policy="native_desktop")`
uses the native per-interval guard whenever the next cumulative probability
is within 1e-6 of one: .99 when the previous cumulative value is below .99,
otherwise `.9 + .1*q_previous`. Intervals are calibrated independently; a
modified endpoint does not replace the next interval's original lower endpoint. The existing
fixed-.99 API remains the default.

## Evidence and remaining limits

Three original installer studies (CRM, BMA-CRM, DA-CRM) establish the saved schema.
Hash-pinned original x86 probes establish wait disposition, raw-rate handling,
prior parameters and final duration; independent R references establish posterior
and complete-data trial calculations. See
[recovery evidence](../research/native-workflow-recovery-2026-10-12.md).

Exact native fast-MCMC stopping/completion-bound shortcuts, populated native
patient-file reopening, and native generated-report comparison remain unverified.
The installer examples contain placeholder reports. Online BMA-CRM is a separate
contract; these desktop settings are not assumed to be its defaults.
