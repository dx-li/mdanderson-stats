# Keyboard design for drug combinations

The Python port supplies dose decisions, safety monitoring, final matrix-isotonic
MTD selection and complete-outcome simulation for
[KeyboardComb](https://biostatistics.mdanderson.org/shinyapps/KeyboardComb/),
catalog entry 121. The [source audit](keyboard-combination-source.md) records the
original R package version, source hashes, reference cases and discrepancies.

```python
import numpy as np
from mdanderson_stats import KeyboardCombDesign, simulate_keyboard_combination

design = KeyboardCombDesign(target=0.3)
patients = np.array([[3, 0, 0], [0, 0, 0]])
toxicities = np.zeros_like(patients)
step = design.next_dose(patients, toxicities, current_dose=(1, 1), rng=121)
assert step.next_dose in ((1, 2), (2, 1))

simulation = simulate_keyboard_combination(
    design,
    [[0.05, 0.10, 0.20], [0.15, 0.25, 0.40]],
    cohorts=8,
    cohort_size=3,
    trials=1000,
    rng=121,
)
print(simulation.selection_probability)
print(simulation.selection_mcse)

# Select a separately published movement rule; default key1 remains unchanged.
paper_variant = KeyboardCombDesign(target=0.3, movement_algorithm="key3")
```

## Decisions and safety

Matrices have rows for drug A and columns for drug B, each with at least two
levels and rows no greater than columns. Dose pairs are **one-based**. Counts
represent fully evaluated binary DLT outcomes, with at most 200 total patients. The model assumes toxicity does
not decrease as either agent's dose increases.

The target key extends from `target - margin_left` to `target + margin_right`;
both margins default to `.05`. Endpoint keys are rescaled by their relative
width, following the R package. The strongest key determines escalation,
retention or de-escalation. The default `movement_algorithm="key1"` preserves
the audited R package's non-diagonal neighbor set and ranking: candidates change
exactly one drug by one level and are scored by target-key mass under a
Beta(.5,.5) posterior plus the package adjustment `.0005 * patients`.
Remaining ties use a fresh uniform choice from the supplied NumPy RNG.

The published paper also defines source-specified alternatives `key2`, `key3`
and `key4`. Set `movement_algorithm` to select one. These options rank candidate
neighbors using the paper's unadjusted target-key posterior probability under
Beta(1,1): key2 uses non-diagonal escalation and diagonal-capable
de-escalation; key3 permits diagonal moves in either direction; key4 randomizes
over non-diagonal candidates with probabilities proportional to their target-key
posterior masses. Key2 and key3 choose uniformly among candidates tied at the
largest mass. The returned decision records valid, noneliminated `candidate_doses`
and the corresponding `candidate_probabilities` for these paper variants. For
key2/key3 those probabilities describe uniform selection among tied maxima; for
key4 they are the normalized posterior masses. Key1 keeps its prior result
fields and scoring behavior. This option changes conduct movement only; safety,
early stopping and final MTD selection retain the documented Python/package
conventions. The simulator uses the selected movement algorithm through its
design object. These paper variants do not imply native KeyboardComb package or
app parity.

Safety uses a separate Beta(1,1) prior. After at least three patients, a posterior
probability above `cutoff_eli` (default `.95`) eliminates a combination and the
rectangle of jointly higher doses during conduct. `extra_safe=True` lowers the
lowest-dose stopping cutoff by `safety_offset` (default `.05`). Pass the returned
`eliminated` mask into later decisions and final selection to retain exclusions.
`early_stop_patients` supplies a current-combination enrollment stop (default
100); `None` disables it. `boundary_table` returns integer DLT thresholds,
including the lowest-dose safety threshold. The minimum key width is `.001`.
An eliminated current dose is never retained when no lower neighbor is available;
the trial stops instead.

## Final selection

`design.select_mtd(patients, toxicities, eliminated=...)` fits
`(toxicities+.05)/(patients+.1)` with weights `patients+.1`, including untreated
cells in the weighted fit. The fit is monotone in both axes. Untreated estimates
are subsequently reported as NaN and are not eligible for selection. If no
treated admissible cell exists, Python returns `dose=None`; the original
all-untreated `(1,1)` fallback is deliberately not reproduced.

The selector follows the source's cross-shaped final safety closure, separately
from the rectangular conduct closure, and retains any supplied prior exclusions.
Among treated admissible cells, it adds the original `1e-5 * (row+column)`
selection adjustment **before** minimizing distance to the target. Remaining
ties follow column-major ordering. The returned isotonic estimates retain their
full precision and do not include that selection-only adjustment.

## Simulation and compatibility

Simulation draws independent binomial DLT counts for each fully assessed cohort.
It retains per-trial patient/toxicity matrices, eliminated-dose masks, selected
pairs and stopping reasons, along with mean enrollment and toxicity matrices.
The `(0,0)` selection bin means no MTD. Other bins use the one-based dose labels;
the unused remainder of the zero row/column remains zero. Selection probabilities
and their binomial Monte Carlo standard errors come from the trial replications.

The R conduct function resets its RNG seed on every call; the Python API uses
an explicit stream that advances across decisions. R simulation uses another
fixed seed and does not consistently forward custom margins or safety cutoffs.
Python honors the supplied design. No bitwise RNG or native simulator parity is
claimed. Native errors in low-count boundary construction are documented in the
reference fixtures rather than reproduced as failures in valid Python trials.

The original app also offers generated trial protocols and reports. Those
interfaces and paper key5 remain unimplemented; key5's paper paragraph says to
randomize among two candidates while its defined diagonal candidate set contains
three. The catalog entry remains partial. The default key1 behavior implements
the non-diagonal algorithm used by the audited R package; key2-key4 implement
the separately documented paper movement variants.


## Numerical evidence

All 16 unrounded independent `Iso::biviso` matrices agree within `1.39e-8`
maximum absolute difference, consistent with the original solver's convergence
tolerance. The Python solver retains Dykstra correction terms and checks both
convergence and row/column monotonicity. It raises on nonconvergence.

Low-target reference cases verify that elimination depends on **at least three
patients**, not at least three toxicities. The R printed boundary can disagree
with that stated rule; Python applies the posterior safety criterion directly.
The single-agent Keyboard and BOIN regression checks also passed during
integration. Real-design simulations check all-safe completion, all-toxic
stopping, reproducible fresh tie draws and nonzero Monte Carlo errors.

Deterministic integer cutoffs are memoized per design, with the patient bound
limiting the cache. Randomly chosen transitions are never memoized.
