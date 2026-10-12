# Desktop CRM guide-to-Python scope crosswalk

Scope: CRM Suite 1.0.0 (entry132) and BMA-CRM 2.2.4 (entry81), using independent
Python posterior integration/sampling with recovered desktop control rules.
This is not online entry133's automatic skeleton calibration, Windows executable
identity, or a clinical validation claim. Final catalog promotion requires the
completed regression/artifact checkpoint.

| Guide contract | Python workflow and independent evidence |
| --- | --- |
| CRM and multiple-skeleton BMA-CRM design, prior medians, target/safety, starting dose, cohorts, sample/window bounds | `CRMDesktopStudy`, `fit_bmacrm`; original three studies from each installer; 120 independent R posterior rows; visible controls and original saved schema. Unsupported latent policy values fail. |
| DA timing prior and incomplete outcomes | `dacrm_trimester_prior` native six-interval profile and `fit_dacrm`; original hazard-prior probes plus 94 independent exact R marginal rows, including interval boundaries. Sampler controls are explicit Python settings, with retained diagnostics. |
| No skipping, observed raw-rate cap, safety wait/stop, final minimum-three fallback | Decision/controller modules; guide AppendixII, original x86 branch probes, complete-data R trial references, complete-outcome enumeration and explicit work-limit waits. Desktop raw cap follows lookahead on observed counts. |
| Cohort simulation, accrual, DLT timing, early stop, final duration and scenarios | `simulate_crm_desktop_study`, native scheduling; original wait/clock/duration probes, count/timing checks and all five scenarios in each of six original installer studies. |
| Input parameters and per-scenario selection, patient/toxicity counts, termination, DLT total and duration reports | `CRMDesktopRun.to_html` captures visible inputs and all guide-required outputs, plus explicitly labeled MCSE; native XML reopens. Original report layout/bytes and RNG identity are not claimed. |
| Add/update/delete last/all patients, retrospective decision date, final MTD | `CRMDesktopHistory`, immutable patient edits, `with_patients(history.patients[:-1])` or empty tuple; `recommend_crm_desktop` and `add_patient`. 60 original censor cases, 82 cohort/controller cases, managed US/GB files and complete synthetic history/report round trips. Earlier dates disable addition exactly as described in the guide. |
| Save/reopen and printable output | Study/history HTML envelopes preserve native settings and unknown patient fields; generated HTML is printable in a browser. Caller chooses a path and writes the returned string. Original embedded scripts are not executed. |
| DA estimates and power-user diagnostic exploration | Recommendation objects retain posterior estimates, parameter MCSE and split R-hat. Explicit seeds, draws, warmup and chains permit repeated analyses and settings comparison. Native obscure/adaptive/thinning controls remain saved metadata, not silently translated sampler parameters. |

The older guide labels trial conduct “Currently Under Construction”; the newer
guide advertises it fully. Both recovered managed conduct methods are identical.
The native conductor, lookahead and dose-finder regions are identical after the
explicit pointer relocation map in `tools/crm_native_reference/version_compare.py`.
The complete cohort-helper slice is byte-identical. This supports shared bounded
control contracts, including older within-cohort DA safety disposition, without
claiming the called numerical runtimes are identical.

The native lookahead bound shortcut is replaced by exhaustive bounded completion
checks; the DA sampler uses independent inference and explicit diagnostics. These
are declared computational profiles with independent numerical evidence. Exact
adaptive/native RNG or debug-dialog appearance are compatibility extensions,
not unimplemented statistical endpoints. The full distribution's retained license
restrictions are separate and unchanged.
