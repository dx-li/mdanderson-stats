# U-BOIN joint posterior and allocation

Catalog entry 142 is **partial**. These functions implement the complete-outcome
Dirichlet utility model and the three allocation probability rules from
[Zhou, Lee and Yuan (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7927960/)
and the [official U-BOIN application](https://biostatistics.mdanderson.org/shinyapps/UBOIN/)
(V2.4.4.0, PID 1014). Source hashes are in [uboin-sources.json](uboin-sources.json).

```python
from mdanderson_stats import uboin_posterior, uboin_allocation

posterior = uboin_posterior(
    [[[2, 1], [5, 2]], [[3, 0], [4, 1]]],
    prior=[[.25, .25], [.25, .25]],
    utilities=[[30, 0], [100, 50]],
)
probabilities = uboin_allocation(
    posterior, eligible=[True, True], method="proportional"
)
```

Counts have shape `(dose, efficacy, toxicity)`, with both category axes in
ascending order: response and toxicity increase along their respective axes.
Each endpoint supports two or three categories; `response_level` and `dlt_level`
specify the first category counted as response or DLT. Categories at or above
the threshold are pooled for admissibility only. The utility calculation keeps
all joint cells. In the binary example, the matrix therefore lists no-response
first and no-DLT first, unlike the paper's flattened outcome order.

The prior is required, either one common matrix or a matrix per dose. The paper
specifies positive cell parameters with total prior mass one, without identifying
the individual parameters. The uniform example above is an explicit choice.
Other positive prior masses are supported. Utilities are required on the 0–100
scale. A three-category example, with efficacy rows PD, SD, PR/CR and toxicity
columns minor, moderate, severe, is `[[30,15,0],[50,30,0],[100,45,15]]`.

For posterior cell shapes `a = prior + counts`, total `A`, and utility `u`,
the mean is `sum(a*u)/A`. The variance is
`sum((a/A)*(u-mean)**2)/(A+1)`, in squared utility units. Toxicity and efficacy
tails use the exact Beta marginals of this joint Dirichlet distribution.
A dose is admissible when both tails are **at most** their cutoffs. These are
posterior means and probabilities, without Monte Carlo error.

`uboin_allocation` intersects admissibility with the caller's required `eligible`
mask. `winner` assigns probability one to the maximum-mean dose, with the lowest
array index resolving exact ties; `proportional` normalizes mean utilities;
`equal` assigns uniform probabilities. No eligible dose returns all zeros.
Proportional allocation with zero utility at every eligible dose raises an error.
Tie and zero-utility policies are explicit Python choices, not verified native
defaults. Posterior and allocation arrays are read-only.

This API does not infer a stage transition, decide whether untried doses belong
in the allocation set, or implement stage-II exploration. The trial controller
must supply eligibility. Full stage-I/II conduct, final OBD rules, 3+3 run-in,
titration, delayed-efficacy/immune-response imputation, simulation and native
reports remain unimplemented. The application's categorical controls also carry
under-development labels; native categorical output parity is not claimed.
Stage-I overdose monitoring uses a separate Beta(1,1) model and should not be
substituted with these stage-II Dirichlet marginals.

The independent base-R script `tools/reference_uboin.R` checks binary, 3×3 and
prior-only 2×3 cases, including utility variance from Dirichlet cross moments.
Its fixtures validate the declared mathematical model, not hidden native output.
The paper supplement was inaccessible behind a challenge page during the audit.
