# CATBUB: comparative trials with categorical outcomes

Catalog entry **97** is implemented. The Python API covers the archived R program's
scaled-beta and Monte Carlo posterior comparisons, binary comparator, fixed and
sequential count generation, group-sequential design calibration, alternative
scenario evaluation and analysis of observed data. Trinary, bivariate binary,
bivariate ordinal and CLL-type outcomes use the same categorical interface;
flatten joint categories in the same order for probabilities, counts and utilities.

The reference is Murray, Thall and Yuan's *Utility-Based Designs for Randomized
Comparative Trials with Categorical Outcomes* and the [official CATBUB archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BUBDesign/CATBUBDesign.zip),
linked from [MD Anderson's entry](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/97).
The archive contains `CATBUB-Design.R`, a manuscript, supplement and example
scripts. The [validation record](catbub-reference.json) includes its SHA-256.
Original source, manuscripts and example data are not bundled. No explicit
general redistribution license was found in the inspected R program or README.
This is an independently expressed Python implementation of the statistical
methods, with the compatibility differences below.

## Posterior comparisons

```python
from mdanderson_stats import catbub_compare

result = catbub_compare([[25, 15, 10], [30, 15, 5]], [100, 50, 0])
print(result.probability)  # [P(B>A), P(A>B)] ≈ [.913734, .086266]
```

Counts end in `(2, K)`; leading batch dimensions are supported. Utilities are a
finite vector of length K. The independent Dirichlet posterior has parameters
`counts + prior_ess * prior_probability`, with defaults ESS 1 and uniform category
probabilities. Prior probabilities must be strictly positive. Probability inputs
are normalized by row, including supplied prior probabilities.

`method="beta"` rescales utility to [0,1], computes exact Dirichlet utility mean
and variance, and matches a beta distribution to those two moments in each arm.
It reuses the package's logit-coordinate beta comparison quadrature with an
absolute error estimate. This is a **distributional approximation** for general
utilities, not exact integration of a weighted Dirichlet. With two utility levels
it reduces to a beta comparison. Moments use centered sums rather than subtracting
large covariance terms. Utility rescaling avoids overflow even at ±1e308.

`method="mc"`, `draws=100_000`, `rng=np.random.default_rng(seed)` samples actual
Dirichlet utilities in bounded batches. Its `error` contains binomial MCSEs;
beta mode's `error` contains quadrature error estimates. These are different
uncertainty measures. Constant utilities give zero probability of either strict
ordering and undefined beta shapes. Nonconstant unresolved beta moments raise
an error. `catbub_binary_compare(counts, prior=...)` implements the source's
separate success-only comparator: category zero is success and all others are
pooled as failure, with independent Beta(.5,.5) default priors.

## Simulation, calibration and analysis

`catbub_simulate_counts(sample_sizes, probabilities, trials=..., rng=...)` returns
`(trial, look, arm, category)` cumulative multinomial counts. Sample sizes are a
look vector for equal arms or a `(2, looks)` matrix for unequal arms. Counts are
patients **per arm**, not total enrollment. Generation draws multinomial count
increments directly, then accumulates them; it does not allocate patient-level
categorical arrays.

```python
import numpy as np
from mdanderson_stats import catbub_design, catbub_analysis

# Small workflow demonstration; use more replicates for accurate calibration.
design = catbub_design(
    [[[0.4, 0.6], [0.7, 0.3]], [[0.4, 0.6], [0.4, 0.6]]],
    [0.5, 1],
    [100, 0],
    null_trials=500,
    alternative_trials=500,
    epsilon=0.1,
    rng=np.random.default_rng(9704),
)
print(design.sample_sizes, design.thresholds)

analysis = catbub_analysis(
    [[20, 5], [8, 17]],
    [100, 0],
    previous_sample_sizes=[10],
    maximum_sample_size=50,
    null_trials=500,
    rng=np.random.default_rng(9705),
)
print(analysis.decision, analysis.probability)
```

The first scenario is the target alternative. Initial per-arm N uses the normal
approximation to the difference in mean utility. Calibration generates a null
with the target arm A probabilities in **both** arms. It uses cumulative spending
`alpha * fractions**rho`, with defaults alpha .05, rho 3, beta .20. At each look,
cutoffs use the conditional survivor quantile of the larger ordering probability,
R's default type-7 interpolation, and upward rounding to four decimals. All
superiority decisions use strict `>` comparisons. Planned fractions increase to
one; actual per-arm sizes are `ceil(N * fractions)`.

The sample-size search repeats until target-direction power is within `epsilon`
of `1-beta`, recording N, power and final cutoff at each iteration. Default
simulation sizes match the archive: 50,000 null and 25,000 alternative trials.
`max_iterations` (default 30), `max_sample_size`, and explicit array-size limits
bound the computation. An undefined update, an exceeded limit or failure to
converge raises an error; it does not return a purported calibrated design.
Remaining scenarios are evaluated at the resulting boundaries. Utility-sensitivity
and alternative-selection analyses can repeat this API with other utilities or
with another scenario placed first.

Returned OCs include directional first-stop probabilities at each look, overall
superiority and no-selection probabilities, mean enrollment in each arm and MCSEs.
The null and target OCs reuse the calibration simulations; assess a selected
design on fresh replicates for independent validation. Small simulation sizes or
a noisy power estimate can prevent convergence. Inner Dirichlet Monte Carlo adds
additional noise when `method="mc"` is used.

For external simulated posterior arrays, `catbub_thresholds` accepts
`(trial, look, direction)` probabilities and `catbub_operating_characteristics`
applies thresholds to actual first stopping. In beta simulations, repeated count
tables are deduplicated before posterior integration. There is no R runtime,
additional dependency, or new CI workflow.

Observed-data analysis recalibrates the null using **pooled observed category
frequencies**, retains the actual prior look sizes, and computes spending from
total observed enrollment divided by twice the planned per-arm maximum. Supply
`previous_sample_sizes=[]` at the first look. Prior settings are explicit
arguments. `incremental_alpha` is this look's additional spending;
`cumulative_alpha` is total spending through this look. Decisions are `B>A`,
`A>B`, `Continue Enrollment`, or `No superiority at maximum sample size`.

## Deliberate differences from the archived program

- The archive's analysis function refers to undeclared `n.star`/`theta.star`
  variables and ignores its `a` argument. Python uses explicit prior arguments.
- Its spending report labels a cumulative value as current spending and adds
  cumulative values for the total. Python reports incremental and cumulative
  spending separately.
- Its OCs separately search each direction for a first crossing, potentially
  counting a reversal after a trial has already stopped. Python stops once, at
  the first crossing in either direction; target power uses the correct direction.
- The archive rejects utility differences below one utility unit using `floor`.
  Python rejects zero differences and respects utility-unit invariance. Bounded
  sample-size and iteration checks replace the source's unbounded loop.
- Full precision results replace rounded display tables. R and NumPy random
  streams differ. Zero-utility ties, invalid inputs and numerical failures have
  explicit behavior rather than propagating undefined source calculations.

## Validation

`tools/reference_catbub.R` runs the unchanged downloaded comparison functions
(with tighter integration tolerance for five deterministic fixtures). It also
captures 250 null and 250 alternative two-look trials from an original design
call, preserving the source's ordinary integration tolerance for that run.
`tools/reference_catbub.py` compares Python on these identical count tables.

The 1,000 posterior comparisons differ by at most **1.84e-8**. Both implementations
return per-arm sizes **20, 40**, boundaries **.9906, .9704**, null stopping
probability **.052**, target power **.784**, and mean per-arm enrollment **39.84**
and **33.52** (R displays rounded means). This is a deterministic numerical and
workflow comparison on a small simulation, not precise operating-characteristic
estimation for a clinical design.

Focused tests also check R posterior fixtures, binary Dirichlet Monte Carlo,
large utility scaling, hand-calculated conditional spending, first stopping
without later reversal, unequal-arm cumulative counts, design calibration, and
interim/first-look analysis. Existing beta-comparison numerical tests remain in
place. Full replication of every manuscript simulation table is not claimed.
