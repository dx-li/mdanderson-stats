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

The [primary article's Numerical Study section](https://pmc.ncbi.nlm.nih.gov/articles/PMC6191365/)
defines selection and allocation relative to the true MTD, including allocation
above and below it. It also defines regret through a failure to de-escalate
when two of a dose's first three patients have DLTs. The surrounding explanation
explicitly includes decisions that become regrettable after pending outcomes
resolve. Thus regret is a trial-level history statistic, not a transformation
of mean dose allocations.

The article does not specify the operational comparison time: the next
assignment, the second DLT's ascertainment, or an earlier decision evaluated
retrospectively. This matters for continuous accrual and revisited doses. The
current report consequently preserves the simulator's selection distribution
and per-dose mean allocation counts without assigning a purported native
regret indicator. The article resolves the named metric's high-level meaning;
its precise history rule and native report schema remain unverified.

The main-article evidence above was available in an indexed primary full-text
result on October 4, 2026. Direct publisher retrieval returned HTTP 403; it was
not retried. No main-article PDF was downloaded or redistributed.

Cached source anchors: `research/raw/TITE-BOIN/app.html` tab labels and controls
for Trial Setting, Simulation, Trial Protocol, STFT Calculator and Select MTD;
`research/raw/TITE-BOIN/Guide.pdf` pages 3–8 for suspension/safety and simulation
workflow; `research/raw/TITE-BOIN/supplement.pdf` Table S3 and Figures S1–S9
captions for published operating-characteristic labels. The exact computational
rules for interim conduct are in `docs/tite-boin.md` and
`src/mdanderson_stats/tite_boin.py`.
