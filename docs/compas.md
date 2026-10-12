# ComPAS explicit hierarchy

`ComPASStudy` implements the five-candidate logit-normal/binomial hierarchy
recovered from PNOC022 protocol v3.0 (January26,2023), pp83–84. This is a useful
published mathematical core with saved inputs, inference and auditable reports.
Catalog140 remains **partial**: the original app defaults and complete dynamic
platform workflow are not recovered.

```python
from mdanderson_stats import ComPASStudy, compas_pnoc022_variances

study = ComPASStudy(
    completed=(11, 11, 11),
    successes=(4, 5, 8),
    groups=("explicit_group",) * 3,
    variances=compas_pnoc022_variances("cohorts_1_2"),
    model_prior=(0.2,) * 5,  # illustrative caller choice, NOT a recovered default
    null_response=0.4,
    equality_policy="hold",
)
reopened = ComPASStudy.from_json(study.to_json())
fit = reopened.fit()
print(fit.model_probabilities)
print(fit.selected_model)  # zero-based; None if no numerically separated maximum
print(fit.decisions)
print(fit.to_json())
```

For arm k in explicitly supplied group j:

- `y_jk ~ Binomial(n_jk, logistic(theta_jk))`;
- `theta_jk | mu_j, model_l ~ Normal(mu_j, variance_l)`;
- independent group means `mu_j ~ Normal(0, variance=1_000_000)`.

Conditional independence within the displayed hierarchy is explicit. All
binomial and Gaussian normalizers are retained when comparing evidence. Model
probabilities are proportional to caller-supplied prior weights times integrated
likelihoods; decisions use the highest-posterior model, **not model averaging**.
The source alternates descriptions of compound versus backbone grouping, so
assignments are required inputs. Model prior probabilities are also required;
equal weights in this example are an illustrative choice.

`compas_variances(maximum_n, alternative_response)` computes
`(1-w)/(N*p*(1-p)*w)` for `w=(.999,.75,.5,.25,.001)`. N and p are fixed design
maximum and alternative response, never interim N or observed response. A
calculated tuple may exceed the inference implementation's supported bounds.
`compas_pnoc022_variances` returns the separately printed protocol tuples:

| Profile | Literal five variances | Unresolved source discrepancy |
|---|---|---|
| cohorts_1_2 | .00012, .042, .126, .379, 126.1 | Tracks N33/p.60 rather than sizing p.65; preserve coarse rounding |
| cohort_3 | .00007, .023, .068, .204, 67.96 | Approximately tracks N70/p.70 rather than stated maximum N42; no correction inferred |

Only completed binary outcomes enter n and y. An outcome still pending is not a
completed failure. PNOC's final unknown-status reporting rule does not change
this interim distinction. Numerical scope is1–12 arms, at most6 per group,
completed N0–100 per arm, and five variances in[1e-6,1000]. These are implementation
limits, not protocol eligibility or clinical restrictions.

The numerical method conditions on each group mean and integrates each arm's
logit posterior, followed by adaptive one-dimensional integration over the group
mean. Mode centering, stable log-density ratios and explicit tail-transition
splits handle the very diffuse mean prior and sharply concentrated candidates.
Conditional truncation at12 prior standard deviations has a concavity-based
tail bound. Requested quadrature tolerance is saved with the study; reported
errors are **numerical estimates**, not certified bounds or clinical credible
intervals. Failed integration raises `ArithmeticError` and emits no decision.

A selected-model probability below.1 drops an arm; above.95 graduates it. For
exact equality, callers must explicitly select `hold` or `continue`.
`compas_decisions` exposes these strict rules independently. The fitted workflow
returns `numerical_hold` when error estimates overlap a decision threshold or
when model probability ranges fail to identify a separated maximum. This is
not a futility decision or a trial stop. Cohort3 prose also mentions.8 for
superiority; this implementation expressly follows the equation's.95.

Independent references integrate a two-arm marginal logit prior after analytically
eliminating the group mean (15 cases), and use a different Laplace-centered
Gaussian tensor quadrature for three-arm evidence (five candidates). Tests also
check no-data identities, group factorization, arm permutation, unobserved-arm
invariance, complementary outcomes, source calibration discrepancies and saved
report integrity. These references validate the declared mathematical target;
they do not recover missing original software defaults.

Not implemented: changing-arm data-retention conventions, replacement ordering,
final selection among continuing arms, platform-wide sample-size accounting,
PNOC minimization randomization, or its optional Shotgun-style cross-cohort
borrowing. The protocol's rounded500-run tables are not exact numerical fixtures.
No original protocol PDF is distributed and existing licensing restrictions are
unchanged. See the [newly authored factual source specification](../research/compas-pnoc022-specification-2026-10-12.json).
