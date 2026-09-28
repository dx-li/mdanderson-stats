# Dose Schedule Finder aggregate operating characteristics

The aggregate driver reuses `run_dose_schedule_trial` once per replicate. Each replicate receives independent event and sampler seeds and retains the same explicit arrival schedule, physical truth, prior, candidate schedules, safety rule and posterior settings. The two returned seeds recreate those streams for a direct single-trial replay. No alternate event-time law or allocation rule is introduced.

The aggregate retains final selected-pair counts, no-selection and stop-reason counts, mean cell allocation, enrollment and duration summaries, and pooled observed toxicity by dose/schedule. Selection and stopping probabilities use binomial Monte Carlo standard errors; allocation, enrollment and duration use sample-mean Monte Carlo errors; pooled toxicity uses a trial-clustered ratio error. Full trial histories and posterior draws are released after each replicate. Diagnostic maxima retain infinity; the parameter maximum is from final fits, and risk maxima include retained interim and final fit summaries.

The driver prevalidates the common configuration, output/history/final-fit storage, and a conservative minimum evaluation/work budget before advancing its supplied generator. Remaining aggregate work is passed into each calendar replay and accumulated across replicates. Errors from replay or fitting are not silently discarded or redrawn. These are Python operating-characteristic simulations using the explicit calendar replay; they do not claim native RNG, delayed low-grade classification, within-patient adaptation, or automatic calibration parity.

## Focused validation

Three worker checks passed in 1.57 seconds with numerical libraries restricted
to one thread. Recorded event/sampler seeds reconstruct single trials and
aggregate counts, pooled-rate errors and work totals; a deficient aggregate
budget preserves the input generator; time-unit scaling by `1e200` preserves
finite duration mean and error. Targeted Ruff, formatting, mypy and diff
checks pass. Read-only review confirmed the clustered-ratio error formula,
remaining-work forwarding and scale-normalized online duration moments.
No new CI workflow or broad numerical suite was added.
