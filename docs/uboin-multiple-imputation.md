# U-BOIN pending-efficacy multiple-imputation evaluation

This extension evaluates Stage-II conduct when patients' toxicity categories
are observed but binary efficacy is pending. It accepts externally supplied
posterior-predictive efficacy probabilities. It does **not** fit the delayed
efficacy model described in the paper: the cached paper omits the displayed
scaled-logistic equation and its standardization/likelihood details, while the
application labels the immune-response option as under development.

The caller supplies the complete-data joint counts and an `H × M` matrix whose
row `h` contains one predictive-probability draw for each of the `M` pending
patients. Each row is completed with Bernoulli draws using the required
`numpy.random.Generator`. The existing complete-data Dirichlet posterior is
then evaluated separately for each completion. The returned dose-level mean
utility and low-efficacy probability are averages of those complete-data
functionals. Counts are never pooled across imputations. Toxicity probabilities
are unchanged because pending toxicity is supplied and efficacy is marginalized
out of the toxicity posterior.

`observed_counts` includes only patients with known efficacy. `pending_dose` and
`pending_toxicity` list each pending patient once and use one-based dose and
category indices. Thus toxicity category `1` maps to `counts[..., 0]`; category
`2` maps to `counts[..., 1]`. Efficacy must be binary, while toxicity may use
the design's two- or three-category axis.

```python
import numpy as np

from mdanderson_stats import UBOINDesign

design = UBOINDesign(
    prior=[[0.25, 0.25], [0.25, 0.25]],
    utilities=[[0, 30], [50, 100]],
    candidate_scope="tried",
    s1=3,
    s2=20,
    max_patients=20,
)
observed = np.zeros((2, 2, 2), dtype=int)
observed[0, 0, 0] = 3

# Two pending patients at dose 1, each already classified as non-DLT.
# These probabilities are supplied predictions, not model fits.
predictive_probability = np.array([[0.35, 0.65]] * 20)
decision = design.decision_multiple_imputation(
    observed,
    pending_dose=[1, 1],
    pending_toxicity=[1, 1],
    efficacy_probabilities=predictive_probability,
    current_dose=1,
    rng=np.random.default_rng(2026),
)
print(decision.action, decision.allocation_probabilities)
print(decision.mean_utility, decision.low_efficacy_probability)
```

The evaluator is Stage-II only and requires binary efficacy. Pending patients
contribute to treated-patient totals, the per-dose `s2` threshold, the global
patient cap, tried-dose eligibility, and the Stage-II B1 empirical-toxicity
escalation check. Their efficacy completions contribute only to the posterior
functionals. The existing ordering remains in effect: lowest-dose safety stop,
patient and `s2` stops, B1 escalation, then B2 admissibility/allocation.

At least five predictive rows are required, following the paper's stated
minimum; its simulation used 20. Limits are 1,000 imputations, 1,000 pending
patients, 100,000 predictive probabilities, and 2,000,000 posterior category
cells across the completion loop. Returned arrays are read-only. The Bernoulli
completion tape and dose-level summaries are retained; a full completed count
table for every imputation is not.

See the [source audit](../research/uboin-imputation-evaluation-audit.md) for
the equation-level crosswalk and the unresolved predictive-model contract.
