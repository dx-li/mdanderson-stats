# BARD stage-two allocation and dose selection

Catalog 165 is **partial**. The port provides two-arm covariate-adaptive
allocation and final optimal biological dose (OBD) selection. Patients from
stage one who meet stage-two eligibility criteria must be included in the
history and outcome counts supplied by the caller. Arm 1 is the lower dose;
arm 2 is the higher dose. Actual dose labels remain the caller's responsibility.

## Allocation

`bard_minimization(history_arms, history_factors, new_factors, ...)` evaluates
both possible assignments. For each prognostic factor it counts only the
historical patients matching the new patient's level, then sums the absolute
between-arm count differences after the hypothetical assignment. The arm with
the smaller score receives `probability` (default 0.95, allowed range 0.5–1). With equal scores,
`tie_probability` is the probability of arm 1 (default 0.5). A local seeded
random generator produces the assignment; native R seed equivalence is not claimed.

Use one to five factors, coded as positive integer categories. Include all
previous assignments, even if outcomes are pending. An empty history is
valid. The history is bounded at 100,000 factor cells. This function does not
impose per-arm enrollment caps, enrollment eligibility, or trial stopping rules.

## OBD selection

`bard_select_obd` accepts a `(2, 4)` table in this order:

1. Toxicity without response.
2. No toxicity and no response.
3. Toxicity with response.
4. Response without toxicity.

The required `prior` supplies positive Dirichlet shapes, shared across arms
or specified separately. The posterior shapes equal prior shapes plus counts.
Mean utility is the posterior cell-probability weighted average of the supplied
utilities (default 0, 30, 50, 100; values must be in [0, 100] with endpoints
0 and 100). Marginal toxicity and response Beta
posteriors are derived from this same Dirichlet distribution.

Safety uses the upper tail `P(toxicity > toxicity_limit)` and activity uses
the lower tail `P(response < efficacy_limit)`. Each must be at most its
respective cutoff. If the lower dose has the larger overdose probability,
the two probabilities are pooled using the required positive `safety_weights`.
This is two-point weighted isotonic regression on **probabilities**, not on
observed toxicity rates. The raw and adjusted probabilities are both returned.

If neither arm is admissible, no OBD is selected. If only one is admissible,
it is selected. Otherwise, utility selection chooses the larger posterior
mean utility, with `tie_arm` resolving equality. Noninferiority selection uses
observed response rates: choose arm 1 if `rate1 - rate2 >= -margin`, otherwise
arm 2. Integer cross-products and the decimal representation of `margin`
preserve equality without a floating-point subtraction tolerance. Both arms
need observed patients for this method. Reported response rates are NaN for
unobserved arms in utility mode.

## Example

```python
from mdanderson_stats import bard_minimization, bard_select_obd

allocation = bard_minimization(
    [1, 1, 2], [[1, 1], [1, 2], [2, 1]], [1, 1], seed=165
)
selection = bard_select_obd(
    [[2, 2, 0, 4], [0, 2, 4, 1]],
    prior=[0.25, 0.25, 0.25, 0.25],
    safety_weights=[8, 7],
)
```

The selection example uses the official outcome template's counts, an
**illustrative** Dirichlet prior, and sample-size safety weights. These
settings must be chosen explicitly for the intended design.

## Explicit choices and source discrepancies

The [official OBD help](https://biostatistics.mdanderson.org/shinyapps/BARD/stage2OBD.pdf)
uses `rate1 - rate2 >= -margin`. The arXiv version displays `>= margin`,
while the published PMC text displays `<= margin`. Both conflict with the
help and its stated noninferiority interpretation. The port follows the
official help.

The available sources do not disclose Dirichlet prior shapes, isotonic safety
weights, or randomization/utility tie rules. These are exposed as configuration;
required prior and weight arguments avoid inventing native defaults. Deriving
both safety and activity marginals from the utility Dirichlet prior is an
explicit coherent model choice, not a verified native implementation detail.
Equal-probability allocation ties and lower-arm utility ties are documented
Python defaults. No numerical parity with hidden app internals is claimed.

The integrated stage-one/stage-two calendar simulator, BF-BLRM route,
accelerated-titration/expansion options, cap handling, calibration, and native
reports remain open. Existing BF-BOIN components provide separate stage-one
functionality; this addition does not claim the full BARD workflow is complete.

## Numerical validation and sources

`tools/reference_bard.R` uses independent base-R Beta tails and direct
Dirichlet mean calculations for six arm states, including unequal priors and
reversed safety ordering. It enumerates all 12 possible new patient profiles
in the official three-factor enrollment example. These are mathematical
reference results under declared parameters, not outputs captured from the
native application. Focused checks also cover equality, admissibility,
reproducibility, immutable results, and bounded input sizes.

Source versions and hashes are recorded in [bard-sources.json](bard-sources.json).
The [published paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12240483/)
is Zhao et al., *Clinical Trials* 22(4), 393–404 (2025),
DOI 10.1177/17407745251350596. Downloaded vendor artifacts remain in ignored
research storage and are not distributed with this package.
