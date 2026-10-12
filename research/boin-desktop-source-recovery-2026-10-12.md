# BOIN desktop1.1.0: newly recovered executable contracts

This is a source-reference checkpoint for application99, not a status promotion
or evidence that the current web application128 uses the same wrappers.

The [official version1.1.0 page](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/99)
was checked again. Its asset directory is `BOINDeskTop`, not `BOIN`. The official
[installer archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BOINDeskTop/BOIN_V1.1.0.zip)
is127651084bytes, SHA256
`f9f2bd574d4e72192ffe9b76e7c3486098d7fc36748e2f5d6cebfdf077754af2`.
The MSI was extracted for inspection; the Windows installer was not run.
Hash inventory: [source-recovery JSON](boin-desktop-source-recovery-2026-10-12.json).
Ignored inputs are under `research/raw/boin-desktop-2026-10-12/extracted`.

## Active source paths, not incidental files

`BOIN.exe` MainForm.SimulateTrialOnLocalR RVA0x1df58, IL596..625, explicitly
loads `R/Functions for BOIN_Comb.R` and selects `get.oc.comb_MTDcontour` or
`get.oc.comb_singleMTD`. Its IL1393..1409 supplies `titration.cap` from the saved
`AcceleratedTitrationCap_DrugA` property. The readable R file's SHA256 is
`32a967c9ae89d10445174252a8401c534aee2c41a18b0690d63694f98e62d192`.

Interactive conduct instead uses `BOIN_Calculations.dll`,
BOINtrialConductor.ProcessListOfLookaheadOutcomes RVA0x2df8. It loads `get.boundary.R` for both modes. With FindContours=true it loads
`next.comb2_for_winApp.R`, invokes `next.comb2_for_winAPP`, and uses
`next.subtrial2_for_winApp.R`; otherwise it loads/invokes `next.comb.R`/`next.comb`.
Both paths forward `boDoThreePlusThreeRunIn`.
Do not silently use the simulation's rules for this separate conduct path.

TrialDesign.get_ThreePlusThreeRunIn_Allowed RVA0x305d allows the GUI option only
when target=.25 **and cohort size=3**. The simulation wrappers test target=.25
and their supplied RunIn flag; the interactive boundary function changes n=3
for any target if directly called with the flag, outside GUI validation. GUI cap choices in
MainForm.Refresh_cmbTitrationCap_DrugA RVA0xd598 range from
min(starting DrugA dose+1, number of DrugA doses) to the number of DrugA doses.
These source-inspected UI constraints are separate from the lower-level R probes.

## Thirty executed startup references

`tools/reference_boin_desktop_startup.R` loads the hash-pinned original R as an
external input, selects its original parsed expressions, and executes them with
supplied outcome/direction draws. It does not translate those expressions or
bundle original R. There are24 titration/first-cohort records in
`fixtures/boin-desktop/startup.csv` and6 run-in boundary records in
`fixtures/boin-desktop/run-in-boundaries.csv`.

The run-in flag changes the single-MTD simulation's de-escalation boundary at
n=3 to2. The waterfall subtrial changes boundaries at n=3 **and n=6** to2.
The distinction executes directly in the original R. This is not a complete
standard3+3 algorithm inferred from its label. Non-.25 targets remain unchanged.

The startup probes expose legacy defects that must not become silent scientific
defaults. On a2x3 grid with no DLTs, a cap can end titration before the upper-right
cell, yet the source unconditionally sets the next dose to upper right when no
DLT occurred. Its first top-up generates cohort_size-1 outcomes but increments
reported enrollment by cohort_size, which can create a phantom patient. The
`ft` first-top-up flag is also initialized outside the trial replication loop;
its cross-trial consequences still need a full-wrapper reference. Some probes
use cohort/cap values outside the GUI's allowed combinations intentionally to
isolate lower-level branches; they are not valid-GUI-workflow claims.

## Provenance boundary and next work

The installer contains `License.rtf`, whose text says “redistribution of this
program is not permitted.” Original installers, R files, managed assemblies and
resource templates remain ignored reference inputs and are not redistributed.
This discovery does not authorize removing existing notices, treating this
archive as a GPL grant, or bundling its protocol templates. Independently authored
mathematical workflows and reference-only execution must remain distinguished
from source adaptations in the file-level provenance review. No license split,
permissions change, release or outreach has occurred.

Next recover full simulation traces under the active wrapper, interactive pending/
cohort/titration rules, current saved-project fields and advertised protocol/report
inputs. Check source defects under explicit compatibility profiles and preserve
actual patient counts in scientific defaults. The shipped TITE/3+3/Rolling6 R
scripts and embedded help are additional actionable leads, not completed ports.

## Full-wrapper deterministic references

`tools/reference_boin_desktop_simulation.R` now executes48 complete original
simulation-wrapper requests: single MTD/contour, caps2/3, titration off/on,
run-in off/on, and three truth matrices on a2x3 grid. Each request runs two
replications. `fixtures/boin-desktop/simulation.csv` retains numeric selection,
allocation, toxicity and enrollment outputs; all48 complete without errors.
The two wrappers expose different return-field names, mapped explicitly.

The original external R is hash-pinned. Host services supply constant outcome
uniforms, axis-direction draws, and the independently authored exhaustive
six-cell isotonic oracle already used by the waterfall references. GUI
notification commands are disabled. This validates deterministic wrapper
contracts, not native random-stream identity, general-grid isotonic identity,
clinical validity or Python integration. Original program report text and
source are not redistributed. Together with the startup/boundary probes this
is78 original-reference cases; BOIN desktop remains partial.


## Interactive boundary contract

`tools/reference_boin_desktop_boundaries.R` executes the hash-pinned active
`get.boundary.R`, producing216 rows in `fixtures/boin-desktop/interactive-boundaries.csv`.
Targets .2/.25/.3, run-in off/on and extra-safe off/on cover n=1..18. The144
GUI-valid rows (432 boundary values) agree exactly with the installed Python
`BOINDesign` using its explicit one-of-three modification. The72 direct-R
out-of-GUI rows remain reference probes, not supported GUI-input claims.
`tools/check_boin_desktop_boundaries.py` independently checks that correspondence.

Crucially, interactive contour conduct changes n=3 only, while the simulation
waterfall subtrial changes n=3 and n=6. The simulation behavior must not silently
replace interactive conduct. Managed ProcessListOfLookaheadOutcomes IL105..357
confirms shared boundary loading and the mode-specific next.comb/next.comb2
entry points. This closes the boundary-source question, not pending-patient
controller integration, saved files or protocol generation.
