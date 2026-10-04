# Dose-specific Beta priors for TPI

`TPIDesign(prior=(a, b))` retains the existing common Beta prior exactly.
For a separately elicited prior at each dose, pass a matrix with one `(a, b)`
row per dose:

```python
from mdanderson_stats import TPIDesign, simulate_tpi

design = TPIDesign(
    target=0.30,
    prior=[(1.0, 19.0), (3.0, 17.0), (6.0, 14.0)],
)
one_dose = design.posterior(3, 1, dose=2)
table_at_dose_2 = design.decision_table(max_patients=12, dose=2)
simulation = simulate_tpi(
    design,
    [0.10, 0.25, 0.40],
    cohorts=8,
    cohort_size=1,
    trials=100,
    rng=72,
)
```

Each pair defines an independent Beta prior for that dose. Shapes must be at
least `1e-6`, have sum at most `1e6`, and each prior must pass the design's
existing untried-dose safety check. This dose-varying specification is a
conjugate Python generalization; the cached original-TPI slides specify
independent identically distributed Beta priors, not dose-specific shapes.
See [the source audit](../research/tpi-informative-priors-audit.md).

For dose-specific priors, `posterior` uses the last axis as the dose axis when
the count arrays have that configured length. Thus arrays shaped `(doses,)`
or `(replicates, doses)` apply the matching row-specific prior. For scalar or
other count summaries, pass the one-based `dose=` explicitly. A per-dose
`decision_table` also requires `dose=` because it describes one dose's
posterior decision map. `next_dose` and `select_mtd` infer the dose axis from
their required dose-length count vectors.

The TPI simulation requires the toxicity-probability vector to match the
number of per-dose priors. It precomputes compact ordinary-move,
escalation-barred-move, and unsafe-state tables per distinct prior pair; equal
rows share one table. The storage/work preflight occurs before random draws,
and the lookup table is limited to five million prior/state cells and 64 MiB.
Common-prior simulations retain their existing lookup and RNG path.

These settings change only the prior used in the existing Beta posterior,
posterior interval masses, safety checks, dose decisions, and isotonic point
selection. The separate [posterior interval function](tpi-isotonic-posterior.md)
also accepts these priors. Neither feature supplies scenario tuning or prior
calibration, and neither implies native application parity.
