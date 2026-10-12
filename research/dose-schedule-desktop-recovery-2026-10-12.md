# Dose Schedule Finder2.2.0 source recovery

Work follows preserved mTPI checkpoint8a53a57b1b4ee0c62e60be720db37458ff59105b.
Catalog75 acceptance is recorded in dose-schedule-desktop-complete-checkpoint-2026-10-12.md.
The historical audit's unavailable-executable and broad calibration gaps are
superseded by this evidence; no automatic2013 adaptation policy is substituted.

## Original assets and provenance

Official archive:
https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/DoseScheduleFinder/DoseScheduleFinder_V2.2.0_NoFX2.0.zip

| Original asset | SHA256 |
| --- | --- |
| Archive | 48b5a0618dc79e14a8ac7e3b966506d03dbee1d95ff21c7bea1b83bca4545684 |
| DSF_KernelDLL.dll | f203c2f774ecabb390e58213076fabe02df15aa74c62df9861a7e2e0ac675a1e |
| DSF_Simulator.exe | 9eddb6aa65535fb6a88959f63ebab905b1af4c1f9f80beeb08850dbd029d4062 |
| Utilities.dll | 77e3997f2375ba3d54e3185d7659dd5ef78a378f8c2f04ff9d506a0e7cfc3b64 |
| DoseScheduleFinder_UsersGuide.pdf | 812894e460ea11600db492fe731f1eda403767a61c188cddb2cc3f2232a0c969 |
| License.rtf | d7638706bb0d68d62b9440afe88791884f99c1a0b83ea9dd0f4c093c5ad66c99 |

Original assets remain ignored under research/raw. The license prohibits
redistribution of the original software. No original executable/library/source
or guide text is bundled. New probes are independently authored and fixtures
contain synthetic numerical outputs or a synthetic managed saved study.
Existing repository notices are not removed or relaxed.

## Executed native contracts

`tools/reference_dose_schedule_desktop.py` hash-checks a user-supplied kernel and
runs bounded original x86 instruction probes with pefile/Unicorn. The probes use
host vector/matrix services and CRT mathematics. They are not a full Windows
execution or a validation of the native integration engine.

- 45 prior-calibration cases, native RVA1a630–1aac2, agree with the existing moment
  implementation for1/3/7 doses,1/3/5 administrations and five nu pairs. Both means
  and standard deviations are checked. No alternative variance formula is needed.
- 5133 controller cases:2565 interim,2565 final and3 initial assignments. Native
  admissibility16930 and ranking16b50 consume supplied posterior summaries;
  first assignment16e20 and final16f00 are executed. Inference is not emulated by
  these probes. Cutoff equality, per-schedule exclusion propagation, retained final
  no-skip and lexicographic ties are observed directly.
- 486 observation branch cases execute1b110 with synthetic patient services and
  analytic hazard/survival callbacks. Calls record censor/event times independently
  of a particular hazard model. The model getter156c0 reads field44; constructor
  instruction15868 sets it to14. Simulator setter108e0 stores the GUI delay in a
  different field68, with no read found in the simulator code. The unchanged
  managed Populate routine calls that simulator setter. Production legacy uses14;
  other memory-injected delays in the fixture test the branch itself.
- 30 lognormal calibrations execute10db0, including its original scalar root solve
  and approximate inverse-normal routine. Host log/exp/sqrt replace CRT services.
  The law matches the horizon quantile and the corresponding exponential variance.
  Native CDF error against accurate normal evaluation is at most5.461e-8; at p=.99
  it amplifies to1.228e-5 in mu and4.609e-6 in sigma. Python uses accurate inversion.
- 40 native schedule assignments execute f350, covering1/3/5/10 daily
  administrations,1–5 cycles and doses0/6. The original patient structure contains
  a50-entry timing array. The full plan is installed before event generation.
- 9 final-clock cases execute the14aaa reduction through14b9e. Reported duration
  is max(entry+recorded outcome time); the analysis call receives duration+1.
  The early-stop branch14b7d bypasses final selection and retains no selection.

Native event generation12570 uses a strict time<horizon event comparison and
strict uniform<low-grade-fraction classification. Minimum event time is0.01,
set at15409 from constant29b28. Exponential/Weibull/lognormal virtual tables are
28e68/28e78/28ef0. Python uniforms have an explicitly CDF-oriented convention;
native random-stream equivalence is not claimed.

## Managed saved-file contract

`tools/reference_dose_schedule_study.py` hash-checks the kernel and Utilities DLL,
copies them into a temporary directory, and disables only the native CRT module
initializer in that copy (tiny IL return at RVA1f990). Original managed DTO
constructors/properties and BSSerializer.Serialize then run under Mono. This is
not native initialization equivalence. Original files remain unchanged and the
modified copy is never distributed.

The generated fixture fixes all timestamps, contains a3×2 scenario and a patient
with actual dose changes. The original root is DSFSimApp, FileVersion2.2.0,
ProgID4DE4157E-8DBC-4c28-B3DE-807264EA0D80. Scenario arrays use dose*5+schedule.
Managed saved HTML surrounds XML with begin/end plus that GUID. Python JSON and
reports have their own versioned captured-input schema.

## Scope and acceptance

Current guide-defined prior calibration is the recovered nu moment calculation;
there is no automatic simulation optimizer to invent. Simulation low-grade events
use the stated fraction/delay law. Within-patient dose changes are actual-history
inputs. The distinct2013 automatic policy remains outside this application's
current declared scope. Existing explicit clinical adjudication APIs are retained.

The new integration supplies studies, conduct, all-scenario simulation, saved
reports, prior/posterior risk densities and portable draw export. Safe defaults
and explicit historical profiles remain separate. Final-code regression, fresh-wheel checks, independent posterior integration,
fixture regeneration and artifact/notices checks are recorded in the separate
completion checkpoint and validation manifest.
