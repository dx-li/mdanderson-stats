# Toxicity Probability Intervals: mTPI

Independent Python implementation of the mTPI method of Ji, Liu, Li and Bekele
(2010), *Clinical Trials* 7:653–663,
[doi:10.1177/1740774510382799](https://doi.org/10.1177/1740774510382799).
MD Anderson's [Toxicity Probability Intervals entry](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/72)
links the modified method and labels its distribution version 2.1 (October 5,
2012). Its [public Excel screenshot](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/TPI/TPI.gif)
is a reference for decision-table checks. [Source provenance](mtpi-sources.json)
records the screenshot and the paper's NCBI BioC text. Original programs and
spreadsheet macros are not redistributed.

## Method and conventions

At a dose with `y` toxicities among `n` evaluable patients, the posterior is
Beta(`y+1`, `n-y+1`). Three intervals partition the unit interval at `lower` and
`upper`. The posterior mass divided by interval width is the unit probability
mass (UPM). The largest UPM selects escalation, staying, or de-escalation.
This is mTPI, rather than the original 2007 TPI or the later mTPI-2 method.

`MTPIDesign` defaults to target .30, interval [.25,.35], and overdose cutoff .95.
Custom targets require a matching explicitly supplied interval. Endpoints satisfy
`0 < lower < target < upper < 1`, with each interval at least `1e-6` wide.
Numerically tied UPM scores prefer de-escalation, then staying, then escalation;
the paper does not prescribe this tie convention.

Safety follows the paper's two rules: stop when the lowest tried dose has
posterior probability above the target exceeding the cutoff; if escalation would
enter a previously tried unsafe dose, stay and permanently exclude that dose and
higher doses. There is no three-patient minimum for safety evaluation. Carry the
returned `eliminated` mask into subsequent decisions. This exclusion timing is
explicit; an unsafe higher dose does not automatically eliminate every higher
dose before an attempted return.

Final selection fits monotone posterior means across all tried doses, then finds
the closest admissible dose to the target. Equal-distance ties prefer the highest
at/below target, otherwise the lowest. Equal isotonic weights are the Python
default; `select_mtd(..., weights=...)` accepts a specified alternative. The
paper does not fully specify these weights, and the native archive has not been
audited. Untried dose estimates are NaN, and no admissible dose returns `None`.

## Usage

```python
import numpy as np
from mdanderson_stats import MTPIDesign, simulate_mtpi

design = MTPIDesign(target=0.30, lower=0.25, upper=0.35)
posterior = design.posterior([3, 3, 3], [0, 1, 2])
np.testing.assert_array_equal(posterior.move, [1, 0, -1])
np.testing.assert_allclose(posterior.probability.sum(axis=-1), 1)

table = design.decision_table(30)
assert list(table.action[:4, 2]) == ["E", "S", "D", "DU"]
# Rows enumerate DLTs 0..30; columns enumerate patients 1..30.
# E/S/D are UPM decisions; U additionally flags an unsafe dose.
# Empty cells represent impossible counts. U is interpreted by trial safety rules.

step = design.next_dose([3, 3, 0], [0, 3, 0], current_dose=2)
assert step.next_dose == 1
step = design.next_dose([6, 3, 0], [0, 3, 0], current_dose=1, eliminated=step.eliminated)
assert step.next_dose == 1
np.testing.assert_array_equal(step.eliminated, [False, True, True])

result = simulate_mtpi(
    design, [0.1, 0.2, 0.3, 0.4, 0.5], cohorts=10, cohort_size=3, trials=1000, rng=72
)
assert result.patients.shape == (1000, 5)
np.testing.assert_allclose(result.selection_probability.sum(), 1)
# Selection bins: no MTD, dose 1, ..., dose 5. Patient-level arrays are retained.
# selection_mcse reports binomial Monte Carlo standard errors for these bins.
```

Counts and maximum enrollment are capped at 200; outcomes must be complete.
Simulation precomputes decisions, draws binomial cohorts in batches, and returns
allocation/toxicity counts, exclusions, final selections, safety stops, and
operating-characteristic summaries. A fixed seed is reproducible for fixed
settings. Native Excel/R random streams and exact native simulation parity are
not claimed. Simulation currently uses equal isotonic weights.

## Validation and remaining coverage

Five focused tests check beta probabilities against exact rational-polynomial
identities, including tails at 200 patients; published screenshot columns for
3, 6, 12 and 30 patients; both safety rules and persistent exclusions; isotonic
pooling and final-selection ties; and batched simulation against exhaustive
small-trial response paths. Deterministic all-safe/all-toxic cases and seeded
replay are also checked.

On the development machine, the full 200-patient decision table took about
0.015 seconds, and 5,000 five-dose trials with ten cohorts of three took about
0.29 seconds including final MTD selection. These are local measurements.

**Catalog status remains partial.** The original 2007 TPI design and its
calibration, native archive/source and isotonic-weight audit, prior-sensitivity
options, posterior isotonic interval simulation, and native spreadsheet/report
workflows remain pending. The published screenshot validates the checked mTPI
decisions; it does not establish parity for all software features.
