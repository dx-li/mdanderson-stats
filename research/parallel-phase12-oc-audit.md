# Four-arm Phase I/II operating-characteristic reporting audit

The archived `main_4arms_nobugs.c` writes one row per trial with a 1-based
selected arm (zero for no selection), a posterior probability, total enrollment,
then each arm's enrollment, toxicity count, response count and admissibility
flag. At final selection it changes the printed posterior probability to
`Pr(response > .20)` after deciding with `Pr(response > .05) > .95`; the global
value can also retain an earlier check result on nonstandard exit paths. The
aggregate API reports the unambiguous selection, enrollment, allocation and
admissibility quantities, and does not summarize that stateful output column.

`simulate_parallel_phase12_oc` executes the existing four-arm simulator in
series. It summarizes per-arm selection and admissibility, no selection,
stopping reasons, allocation/enrollment, phase-I enrollment, toxicity and
response event counts, and pooled event rates. Means and probabilities use
trial-level Monte Carlo standard errors. Pooled event-rate MCSE uses the trial
as the independent unit (delta method); a zero-enrollment arm has an undefined
rate and MCSE. One-trial MCSEs are undefined. The optional `optimal_arms`
statistic is explicitly only the probability of selecting within the supplied
set; the C source defines no optimal-arm success event.

The caller supplies a nonnegative root seed. Each trial receives a recorded
uint64 seed derived by index, and can be replayed with the existing single-trial
simulator. No individual patient histories are retained. Trial count and
worst-case patient work have explicit bounds. This reporting workflow does not
reproduce the archive's external R posterior-comparison callbacks, operating
characteristic scenarios, or random stream.
