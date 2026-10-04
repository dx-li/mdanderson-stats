# Random-forest censoring reference

This small base-R harness checks the cached `get.brier.survival` helper's
`cens.model = "rfsrc"` data flow and Brier arithmetic against explicit
fixtures. It does not fit a censor forest or claim randomForestSRC RNG or
forest-fit parity: `rfsrc` and `predict` are deterministic boundary stubs.

The pinned source is randomForestSRC 3.2.2, tag
`b4d099e262423362a8872c13c468e6dbe2f9e9da`, cached at
`research/raw/randomForestSRC/R/utilities.survival.R` (blob
`9ed62c12c118edd11d78fb85f2ec230448646406`). The harness calls the unchanged
`get.brier.survival`, `set.nodesize`, `sIndex`, and `trapz` definitions.

The cases exercise full-training-cohort censor-model fitting, prediction-row
alignment, censor-survival projection to the event-interest grid, ties at the
grid boundary, reduced event grids, row-specific censor curves, missing OOB
survival rows, zero censor-survival denominators, and integrated score
calculation. The independent calculation preserves the source's two weighted
terms separately and reports finite and nonmissing contribution counts.

With no censoring, the reference calculation uses the mathematical convention
`G(t) = 1` and reports the corresponding Brier results. The unchanged source
helper instead assigns a vector in its no-censor branch, then indexes it as a
matrix in the `rfsrc` Brier branch; the harness records this dimension error as
a source limitation rather than concealing it. This is distinct from a failure
of the mathematical G=1 reference.

Run the script with the cached helper path and a fixture output directory:

```sh
Rscript tools/reference_random_survival_censoring_brier.R \
  research/raw/randomForestSRC/R/utilities.survival.R /tmp/random-survival-censoring-reference
```

The generated CSVs preserve the exact inputs, censor curves, projected
denominators, per-row weighted terms, scores, and stub call arguments for
comparison with the Python implementation. They are deterministic and require
base R only.
