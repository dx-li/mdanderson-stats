# BOIN desktop complete-workflow acceptance

This extends b8dae6c. Catalog99 is implemented for the declared current1.1.0
Python workflow; counts99 implemented/33 partial/6 pending.
All original source, binaries, templates, defaults and help files remain external
and ignored. Reference fixtures are synthetic numerical/serializer outputs.

## Advertised workflow crosswalk

| Advertised operation | Integrated Python contract and evidence |
| --- | --- |
| Single-agent design/simulation and 3+3 comparison | Saved active bounds, titration/run-in, safety, all scenarios/batches, fixed-selection comparator expansion; previous192 native wrapper cases and12 comparator cases. |
| TITE design/simulation and Rolling Six comparison | Saved units, accrual, three event-time distributions, uniform/informative analysis weights, pending gates;2304 original calendar traces and238 complete Rolling traces. Two native unbounded loops remain explicit guard errors. |
| Combination single MTD and waterfall contour | Saved simulation/controller, capacity/titration/run-in, edited histories, final selection, reported/actual observations and legacy qualifications preserved from prior checkpoints. |
| Pending decision tables and flowchart | Safe unrounded thresholds and observed-data precedence; explicit native-table rounding/strict boundaries/ignored run-in.7124 compressed native rows from64 requests, independently checked threshold neighborhoods. Two native high-target edge cases distinguish false elimination and undefined stop flag. |
| Combination desirability scores |16 original requests cover cohort1/3, maximum12/30, target.25/.3 and two upper bounds. Global ranking precedes EarlyStop display filtering. Safe controller score is interval mass+.0005*n; native table ranks raw mass and eliminates at equality. Independent beta calculation and cutoff-equality tests validate the distinction. |
| Final single/TITE analysis | Saved TrialData, completeness gate,960 original report-numeric cases, standalone reopenable report; estimates, marginal transformed limits and separate safety/report tails. |
| Final combination/contour analysis | Dedicated jagged int arrays (drug-A rows, drug-B columns), edit/reset/read, explicit completeness, source-validated selector and standalone report. Original serializer fixture uses the two relevant original properties, ignoring unrelated optional dependencies; it does not establish full native XML round-trip identity. |
| STFT/WSTFT calculator | Saved days, edit/reset, per-patient weights and standalone report.84 original managed cases from prior checkpoint. Zero follow-up supported mathematically, unlike native GUI's positive entry gate. |
| Combination patient conduct | Add/edit/delete/reset, go/wait/stop and final analysis, safe capacity/exclusion checks. Selected-row review recomputes from preceding rows with current outcomes, not historical outcomes. |
| Scenario operating characteristics |256 native wrapper cases distinguish exact-target console from nearest-target returned metrics, mutable native denominator, geometric contour residual versus exact reference points, actual/reported enrollment and no-selection versus recorded safety stop. TITE conventional duration is explicitly an approximation. |
| Protocol output | Captured-input methods, decision appendices and simulation/comparators in one reopenable HTML. English/Chinese methods and tables; detailed diagnostics remain English. Safe methods reject legacy simulations. Native prose/layout is not copied. |

## Additional source inspection

`MainForm.DisplayStatDetails` RVA0x21354 calls
`FormGetDecision.DisplayRecalculatedDecisionForEnrolledPatient` RVA0x26d8.
The latter copies rows with index strictly below the selected row and uses their
current outcomes. Python's review function returns that prefix plus a freshly
seeded safe recommendation; original Windows dates and RNG identity are not
claimed. The source `GetScoreTable` applies EarlyStop filtering only after the
full score table returns. Hashes are recorded in docs/boin-desktop-sources.json.

The native table's false elimination at target.6/n3/y3 is isolated: posterior
.8704 does not exceed.95, so safe action de-escalates; native table eliminates.
With extra-safe enabled the original fails with an undefined `stopneed` flag;
compatibility raises explicitly. No defect is a silent scientific default.

## Acceptance status

Final unchanged-production full suite passed **52,790 tests in 865.40s**, all extras,
warnings as errors. Catalog status metadata was updated after the full run and
independently checked; no production Python changed. Fresh wheel: 17,396 desktop tests passed in 26.09s;
seven retained independent-review regressions and 100 optional plot/image tests
passed. All four original defaults plus a derived contour workflow passed, with
standalone reports, pending/score tables and captured-input bilingual protocols.
The installed 40,000-trial quality run and all 15 R fixture regenerations passed.
All 746 package Python/JSON files matched the tested installed wheel before
status-metadata update. Wheel/sdist preserve all 29 notice files, exclude raw
original assets, and pass final build/provenance checks. Exact final hashes are
in the preservation commit message. See the validation manifest for details.

The first full-suite attempt was restarted because a new test left a CSV file
handle open under warnings-as-errors. The fixture loader was corrected; no
production algorithm changed. Installed tests subsequently passed without warnings.

The final rebuilt wheel, including updated catalog metadata, passed all 17,396
desktop tests again in 24.39s. All 746 source/installed package files match;
catalog and coverage-index membership/counts agree. No Python changed.
