# Current EffTox saved-study workflows

The current six-coefficient, binary-model EffTox 5.2.3 XML format now supports
Python patient conduct, pending-outcome decisions, prior calibration,
complete-outcome simulation and self-contained HTML reports. These APIs are
available from the modules below. Historical four-coefficient files are not
silently migrated; historical trinary and inverse-quadratic utilities retain
separate APIs in [the mathematical guide](efftox.md).

This is independently maintained Python inference with recovered desktop control
flow. Source-compatible behavior, numerical accuracy and clinical validity are
separate claims. Native posterior integration, optimizer and random-number
identity are not promised. The catalog remains partial pending final acceptance
of the complete declared workflow scope.

## Read and resolve a prior

```python
from pathlib import Path
from mdanderson_stats.efftox_desktop import read_efftox_desktop_xml
from mdanderson_stats.efftox_desktop_calibration import calibrate_efftox_desktop
from mdanderson_stats.efftox_desktop_inference import EffToxDesktopInference

study = read_efftox_desktop_xml(Path("study.sim").read_text(encoding="utf-8"))
if study.newest_hyperparameters == "input":
    resolved = calibrate_efftox_desktop(study, monotone_toxicity=False)
    if resolved.study is None:
        raise RuntimeError("Review calibration optimizer/quadrature diagnostics")
    # Inspect achieved means/ESS, objectives and boundary hits, even on success.
    print(resolved.calibration)
    study = resolved.study

inference = EffToxDesktopInference(
    study,
    monotone_toxicity=False,
    seed=901,
    draws=1000,
    warmup=500,
    chains=4,
    toxicity_summary_profile="model",
    max_work=20_000_000,
)
```

Use the file's actual encoding; the original distributed default uses ISO-8859-1.
`NewestHyperParams=input` cannot silently use stale output. Calibration requires
an empty patient history, preserves the saved curvature/association prior inputs,
and emits executable output only when both optimizers and quadrature checks
succeed. Nonzero residuals and active bounds remain scientific qualifications.
Saved Python calibration metadata records its slope profile and quality evidence;
reopening that output with a different slope profile fails explicitly.

The required `monotone_toxicity` choice controls the actual statistical prior:
`False` uses independent untruncated Gaussian coefficients; `True` conditions the
toxicity slope to be positive. Executed original posterior-integrand references
confirm the untruncated Gaussian density, including negative slopes. They do not
establish identity of native adaptive integration bounds or numerical errors.

The separate `toxicity_summary_profile="desktop_legacy"` reproduces a recovered
native inconsistency: when a draw's toxicity slope is negative, reported toxicity
means and toxicity-tail indicators use the **first-dose** toxicity for every dose,
while the likelihood still uses each dose's actual modeled toxicity. The default
`"model"` preserves the modeled probabilities consistently. Neither choice changes
the posterior likelihood; generic `efftox_decision` defaults remain unchanged.
Accurate Python Lp contour solving also differs from the native coarse bisection.

## Conduct and edit a patient history

```python
from mdanderson_stats.efftox_desktop_controller import recommend_efftox_desktop
from mdanderson_stats.efftox_desktop_patients import (
    add_efftox_desktop_patient,
    update_efftox_desktop_patient,
    delete_last_efftox_desktop_patient,
    clear_efftox_desktop_patients,
)
from mdanderson_stats.efftox_desktop_report import efftox_desktop_conduct_html

recommendation = recommend_efftox_desktop(study, inference, max_analyses=1000)
Path("conduct.html").write_text(
    efftox_desktop_conduct_html(study, recommendation, inference=inference),
    encoding="utf-8",
)
if recommendation.status == "go":
    study = add_efftox_desktop_patient(
        study,
        inference,
        patient_id=101,
        begin_date="2026-10-12",
    )
    study = update_efftox_desktop_patient(study, 101, outcome=3)
    Path("updated.xml").write_text(study.to_xml(), encoding="utf-8")
```

All dose indices are zero based. Native outcomes are `-1` pending, `0` toxicity
only, `1` neither, `2` both, `3` efficacy only. Dates and IDs are identifiers,
not time-to-event observations. Counts use administered dose; it may differ
from the calculated dose. Edits retain the original calculated dose and cohort,
and can correct IDs, dates, administered doses and outcomes. Duplicate IDs fail.
Last-patient deletion and explicit history clearing preserve other saved settings.
Unknown study and patient metadata survive semantic XML round trips.

The current GUI analyzes the completed-cohort prefix while the trailing cohort
is incomplete. Its two latent within-cohort escalation flags are both false.
A pending lookahead compares every possible completed count matrix with the
**observed-only baseline**, including the no-selection state. Agreement only
among completed cases is insufficient. Exchangeable pending patients are grouped
by administered dose; independent patient-level enumeration verifies the same
possible sufficient statistics. A count-keyed sampler seed makes results
independent of traversal order.

`wait` and `work_limit` release no dose. Work limits are checked before any fit;
no truncated search is reported as consensus. `stop_early` means no acceptable
dose; `complete` can retain the final selected dose. A manually shut-down saved
trial cannot enroll. Actual patient capacity is always enforced by the add API.

At capacity with a partial final cohort, the default safe controller includes the
whole history. The explicit `capacity_profile="desktop_legacy"` retains the native
prefix bug, which can return GO at capacity; this profile cannot bypass the add
API's independent capacity check. Overfull cohorts cannot accept new patients.

## Simulate and report

```python
from mdanderson_stats.efftox_desktop_simulation import simulate_efftox_desktop
from mdanderson_stats.efftox_desktop_report import efftox_desktop_simulation_html

# A fresh service keeps the budget and diagnostics scoped to this simulation.
inference = EffToxDesktopInference(
    study,
    monotone_toxicity=False,
    seed=901,
    max_work=20_000_000,
)
result = simulate_efftox_desktop(
    study,
    inference,
    scenario_index=0,
    trials=2,
    seed=301,
)
Path("simulation.html").write_text(
    efftox_desktop_simulation_html(result),
    encoding="utf-8",
)
```

Simulation starts from an empty history. Saved patient records are not simulated
observations. The saved independent/dependent truth determines joint outcomes;
inactive conditional probabilities are retained, and impossible active joint
probabilities fail validation. A final partial cohort never exceeds capacity.
Every trial receives a final analysis, including early stops, using the same
current reachability rule as interim selection.

Results retain per-trial cell counts, selected indices (`-1` for no selection),
analysis counts, selection probabilities/Monte Carlo standard errors, mean
patients by dose, and every supplied posterior diagnostic. Rselect is conditional
on selection, matching executed native arithmetic. With no selections it is zero;
with selections and constant true utility it is explicitly undefined, rather
than silently returning a native division-by-zero artifact.

Reports include true efficacy/toxicity/utility, allocation, selection, all saved
inputs, actual sampler/compatibility settings, prior-only marginal means/tails/
utility, and all fit diagnostics. Missing diagnostics from a custom analysis
service are labeled as missing. HTML escapes identifiers and saved XML.

`max_work` bounds nominal dose/coordinate/sweep work, not actual likelihood calls
or elapsed time; individual fits also retain their existing bounds. Its supported
range is 1..200 million. Large replication requests need explicit budget increases
or distinct-seed batches. Tiny-draw runs are workflow smoke tests only. Posterior
R-hat/MCSE and simulation Monte Carlo error address different uncertainty sources;
neither certifies clinical validity.

## Evidence and remaining acceptance

The current source references contain 738 executed cases: 500 selection/
admissibility, 86 cohort/pending/controller, 8 Rselect, 48 negative-slope summary,
15 normal-log-density and 81 complete posterior-integrand cases. The complete
integrand matches the independently implemented Python likelihood and Gaussian
prior to maximum relative error below `7e-15` in those cases. Original DLLs are
hash-pinned, ignored reference inputs; they are not distributed or needed at runtime.

See [source contracts and checkpoint](../research/efftox-current-workflow-contracts-2026-10-12.md)
for exact execution boundaries. Integration limits, native posterior numerical
identity and native calibration optimizer identity remain distinct from these
executed integrand/control-flow results. Full workflow acceptance, including
all-scenario/batched operating-characteristic reporting and precision sensitivity,
remains open before catalog promotion.
