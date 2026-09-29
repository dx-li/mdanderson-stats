# iBOIN final selection and complete-outcome simulation

`select_iboin_mtd` accepts aggregate final patient and DLT counts. It computes
the raw estimate as `y/n`, or, when borrowing is requested, as
`(y + m*q)/(n + m)`, where `q` is the design skeleton and `m` is explicitly
chosen as no prior ESS, original prior ESS, or robust-effective prior ESS.
This follows the recovered iBOIN final-estimation help. It fits an increasing
weighted isotonic regression over all treated doses, then applies safety and
candidate-dose eligibility. Untreated doses are never selected. The native
program's isotonic weights, tie handling, and several prior-linkage details
were not recovered; these are explicit Python policies, not native parity.

The caller must choose `isotonic_weights`: `"patients"`, `"effective"`,
`"equal"`, or a custom vector. For custom weights, aggregate selection ignores
zero weights only at untreated doses and requires positive weights at treated
doses. Simulations require custom weights positive at every dose because any
dose may be treated. `eligible_doses` is an optional boolean candidate mask;
treated doses outside it still contribute to the isotonic fit. Ties in
distance from the target within an absolute tolerance of `1e-14` use the
requested lowest/highest dose policy. The optional de-escalation-bound filter
uses each dose's design boundary at its observed final sample size. If no dose
survives, the returned selection is empty rather than falling back.

```python
from mdanderson_stats import IBOINDesign, select_iboin_mtd

design = IBOINDesign([0.10, 0.25, 0.40], [0, 0, 0], target=0.25)
selection = select_iboin_mtd(
    design,
    [6, 6, 6],
    [0, 1, 3],
    prior_mode="none",
    isotonic_weights="patients",
)
```

`simulate_iboin_trial` generates independent patient outcomes from caller
supplied mutually exclusive maximum-severity probabilities for grade 2 and
DLT (the remaining probability is no toxicity), applies the existing iBOIN
conduct rules, and returns the replay and final selection. It has one explicit
seed for direct reproduction. `simulate_iboin` runs bounded trials serially;
give either a root `seed` or an exact `trial_seeds` vector to reproduce each
trial independently. The summary reports per-dose mean enrollment and event
counts, their trial-level Monte Carlo standard errors, mean total enrollment,
selection and stop probabilities with Monte Carlo errors, quantiles, and
replay seeds. With one repetition, Monte Carlo standard errors are undefined
and reported as NaN.

```python
from mdanderson_stats import simulate_iboin

oc = simulate_iboin(
    design,
    grade2_probability=[0.05, 0.10, 0.10],
    dlt_probability=[0.02, 0.12, 0.30],
    cohort_size=3,
    max_patients=30,
    repetitions=100,
    prior_mode="none",
    isotonic_weights="patients",
    seed=20260929,
)
```

Safety stops, including the extra-safe terminal rule, force no final MTD.
Simulation is a complete-outcome Python workflow using the package's
accelerated-titration and cohort conduct implementation. Its RNG stream,
outcome association model, final-selection policy, and aggregate summaries
are explicit Python behavior; this does not claim application RNG or report
parity. The serial run preflights worst-case conduct work and memory and
retains no per-cohort histories in aggregate mode. Work, history-cell,
repetition, and storage ceilings are bounded; reduce `max_patients` or
`repetitions` when a request exceeds them.

Mean-count Monte Carlo errors use bounded integer sums and sums of squares;
under the configured 100,000-repetition and 100,000-patient ceilings, the
largest accumulated square sum remains below `2**53` and is exactly represented
in float64.
