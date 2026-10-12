# Dose Schedule Finder desktop workflow

The Python workflow imports Dose Schedule Finder2.2.0 saved XML/HTML, preserves
model inputs and actual treatment histories, fits as-of studies, simulates every
saved scenario, and produces standalone reports with reopenable execution inputs.
The original Windows application and its integration/random streams are not
reproduced. Independent posterior accuracy and source contract compatibility are
separate evidence. This is not clinical validation.

## Study and conduct

`DoseScheduleDesktopStudy` captures the prior probabilities, peak/tail means,
nu constants, standard cycles, target, overdose cutoff, scenarios, low-grade
settings and actual patient ledger. Dose and schedule indices in Python/XML are
zero-based. Report tables display one-based indices. A native scenario vector
has35 entries with a fixed schedule stride of five; inactive entries do not become
active scenarios. Native integration speed and extra-sample factor are retained
as metadata; Python sampler settings are explicit and are not inferred from them.

`read_dose_schedule_desktop_study` accepts original2.2.0 XML, the original GUID
HTML envelope, or versioned Python JSON. `study.to_json()` saves all scientific
inputs. Numeric fields, dimensions, duplicated fields and bounded input size are
validated. XML entity/document-type declarations are rejected. No code or macro
in an input is executed. The importer does not import old computed results as
newly fitted evidence.

Each `DoseScheduleDesktopPatient` records entry time, assigned pair, outcome,
outcome/censor time, actual administration times and the actual dose at each
administration. Replace an immutable record/study to add, edit or remove a patient.
Actual doses may differ from the assigned dose. Future entries and administrations
are masked at analysis time. An event includes only administrations before its
onset. A censored record includes administrations through its observation time.

`analyze_dose_schedule_desktop(study, seed=..., sampler=...)` returns the fit,
allocation, safety/eligibility masks, pending outcomes, final readiness and
posterior diagnostics. `final=True` in the default safe profile requires observed
patients with completed follow-up and relevant qualification. Early withdrawal
before the horizon is not silently labeled completed follow-up. Calendar-time
comparisons avoid losing an exactly completed horizon to floating subtraction.

## Safe and explicit historical rules

The default `safe` profile uses the paper's strict probability cutoff, evaluates
each regimen's risk independently, and removes no-skip at final selection. Both
profiles begin at the lowest dose/shortest schedule. Exact distance ties choose
lowest dose then shortest schedule.

`desktop_legacy` reproduces the recovered controller contracts:

- Equality at the overdose-probability cutoff is accepted.
- Exclusion of a lower dose propagates to higher doses within that schedule,
  even when different hazard timing would make the higher-dose risk lower.
- Coordinate-wise no-skip remains active at final selection.
- The first assignment bypasses the posterior screen.
- Low-grade qualification uses the model's fixed14-day delay. The GUI's separate
  simulator delay setter does not change that model field.
- Pending low-grade observations may accrue censored follow-up beyond the risk
  horizon. Qualifying events are backdated to onset.
- Simulation retains the entire planned administration ledger, and likelihood
  construction masks it at the observation time. Its final reported duration is
  the maximum entry-plus-outcome time; final analysis is one day later. This may
  leave low-grade events pending. An early safety stop is not reversed by a new
  final selection.

Those historical choices are explicit compatibility behavior. Safe simulation
stops planned administrations at onset, honors the requested low-grade delay and
waits for completed horizon/qualification. Existing paper APIs and explicit
clinical grade-2 adjudication helpers retain their distinct contracts.

## Simulation and event laws

`replay_dose_schedule_desktop_trial` accepts separate arrival, event, grade and
posterior-seed tapes. `simulate_dose_schedule_desktop` generates those tapes and
runs all saved scenarios serially. The first patient arrives on day zero;
subsequent gaps are exponential. True regimen risks determine exponential,
Weibull or lognormal event times. The lognormal calibration fixes the horizon
quantile and matches the variance of the corresponding exponential law. Accurate
normal inversion differs slightly from the original approximation; that numerical
difference is documented rather than hidden in a parity claim.

Trial simulation retains the original explicit0.01-day minimum event time.
Events at the horizon are censored. The low-grade fraction classifies generated
toxicities; every such simulated event eventually qualifies after the relevant
delay. This is the application's simulation law, not a model of spontaneous
resolution, attributed dose reductions, or the distinct2013 within-patient
adaptation policy. Clinical histories remain caller-entered actual histories.

Selection bins are row-major dose/schedule combinations followed by no selection.
Outputs include selection MCSE, mean allocation and its MCSE, pooled toxicity,
duration and its MCSE, early stopping, pending final outcomes, and separate
posterior risk/tail MCSE and split R-hat. A one-repetition sample mean MCSE is
undefined. Zero allocation has an explicitly reported pooled toxicity of zero.
A completed short-chain simulation is not evidence of adequate posterior precision.

## Reports and saved distributions

`dose_schedule_desktop_analysis_report` includes actual histories, every regimen's
mean/SD/overdose probability, eligibility and risk-density plots. An empty-history
analysis produces prior densities; an observed-history analysis produces posterior
densities. `dose_schedule_desktop_distribution_json` exports chain/draw arrays,
parameter names and exact analysis context. `dose_schedule_desktop_simulation_report`
contains every scenario and its operating characteristics, visible study inputs,
per-regimen mean toxicity counts and total mean enrollment/toxicity.

The optional `trial_callback(scenario_index, trial_index, result)` streams detailed
trial records without retaining a batch of posterior arrays.
`dose_schedule_desktop_trial_json(result)` exports the exact study, truth,
sampler/budgets and all four input tapes, actual histories, every allocation and
final decision. Undefined precision diagnostics use an explicit JSON null marker.
The final posterior draws can be exported separately from `result.analysis`.

`read_dose_schedule_desktop_report` restores the study, profile, seed, sampler,
report kind and final/interim setting. Saved inputs can reproduce an analysis
without treating the report's displayed values as a new computation.

## Numerical bounds and evidence

The desktop study permits1–7 doses,1–5 schedules,200 patients and100 scenarios.
Candidate schedules allow at most50 administrations each and200 across all
schedules. This extends the prior20-administration core bound to cover the native
50-entry treatment array; likelihood and total-work limits remain enforced.
Simulation enforces aggregate posterior-evaluation/work budgets before and during
serial execution. Request fewer trials or an independently justified sampler
configuration when a requested run exceeds the bound.

See [source recovery](../research/dose-schedule-desktop-recovery-2026-10-12.md)
for original hashes, portable reference generators, numerical qualifications and
remaining acceptance. Original assets are not redistributed. The repository's
mixed-license notices remain in force.
