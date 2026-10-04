# TITE-BOIN protocol report source crosswalk

This report is a Python summary over a newly executed `simulate_tite_boin`
result. It is not an implementation of the application's HTML/Word/PDF
templates, and it does not claim the native application's scheduling or RNG
sequence.

| Source-visible operation | Existing computation | Report treatment |
| --- | --- | --- |
| Simulation accepts typed-in toxicity scenarios, trial count and protocol settings | `simulate_tite_boin`, `run_tite_boin_trial` | Request captures truth vectors and all settings, then runs each scenario serially |
| Simulation reports selection/operating characteristics | `TITEBOINSimulation` | Selection probabilities and MCSE; mean per-dose patient/DLT counts, duration, suspension and stop-reason summaries |
| STFT calculator accepts assessment window, timing prior and pending follow-up | `toxicity_followup_weights`, `tite_boin_estimate` | No duplicate calculator is introduced; these existing APIs remain directly usable |
| Select MTD uses completed/evaluable per-dose patient and DLT counts | `BOINDesign.select_mtd` | Kept separate from scenario reports; no synthetic patient-count inputs are accepted |
| Protocol tab offers HTML/Word templates, flowchart and Table 1 | Cached `research/raw/TITE-BOIN/app.html`, Trial Protocol tab | No document-template or native flowchart rendering is claimed |

The cached app's simulation UI and published supplement identify correct/overdose
selection, correct/overdose allocation, regretful trials and average duration as
operating-characteristic labels. They do not, in the inspected app HTML or guide,
fully define a native report schema or all evaluation conventions for those
labels. This report therefore presents the simulator's exact selection
distribution and per-dose mean allocation counts without assigning an
unverified “correct MTD” or regret definition. Its scenario truth is included so
users can conduct an explicitly chosen downstream evaluation.

Cached source anchors: `research/raw/TITE-BOIN/app.html` tab labels and controls
for Trial Setting, Simulation, Trial Protocol, STFT Calculator and Select MTD;
`research/raw/TITE-BOIN/Guide.pdf` pages 3–8 for suspension/safety and simulation
workflow; `research/raw/TITE-BOIN/supplement.pdf` Table S3 and Figures S1–S9
captions for published operating-characteristic labels. The exact computational
rules for interim conduct are in `docs/tite-boin.md` and
`src/mdanderson_stats/tite_boin.py`.
