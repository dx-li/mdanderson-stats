# Binary-endpoint sample-size planning

Catalog entry **139**, MD Anderson's
[Nbinary calculator](https://biostatistics.mdanderson.org/shinyapps/Nbinary/),
is implemented for one- and two-sample proportion comparisons, Fisher's exact
test, paired McNemar tests and binary Cohen's kappa. The
[source record](binary-sample-size-sources.json) pins four technical documents
and the application's help examples.

## Proportion comparisons

```python
from mdanderson_stats import binary_proportion_sample_size, binary_proportion_power

one = binary_proportion_sample_size(0.2, 0.5)
assert one.group_sizes == (18,)
two = binary_proportion_sample_size(0.2, 0.5, allocation_ratio=1)
assert two.group_sizes == (29, 29)
corrected = binary_proportion_sample_size(0.2, 0.5, allocation_ratio=1, method="score-continuity")
assert corrected.group_sizes == (37, 37)
```

These are **normal-approximation planning calculations**, not exact guarantees of
binomial type I error or power. With `method="wald"`, the working variance uses
the specified alternative response rates. The one-sample variance is `pt*(1-pt)/n`;
the independent two-sample variance is `pc*(1-pc)/nc + pt*(1-pt)/nt`.

For two-group equality, `method="score"` uses a pooled null boundary with the
enrollment-weighted response rate. `"score-continuity"` additionally shifts that
boundary by `0.5*(1/nc + 1/nt)`. Both retain the alternative variance for power.
The corrected source example is reproduced by this pooled-boundary calculation;
its uncorrected example uses the Wald calculation. They are different planning
approximations, not an exact discrete-test pair.

The default test is one-sided. Equality uses the direction of the specified rate
difference; `sides=2` uses alpha/2 and retains the dominant rejection tail, as in
the source planning convention. Thus null two-sided *planning* power is alpha/2.
Directional margin tests use the signed difference `d=pt-pc`, with larger
response rates favorable:

| Objective | Alternative |
| --- | --- |
| `equality` | Nonzero difference; margin must be zero |
| `equivalence` | -margin < d < margin |
| `noninferiority` | d > -margin |
| `superiority` | d > margin |

Margin tests require `method="wald"` and a margin in (0, 1). They use one-sided
alpha regardless of `sides`; equivalence combines two one-sided rejection rules
into a normal interval probability. Sample-size searches require expected rates
inside the alternative. Power evaluation also accepts effects outside it.

```python
equivalence = binary_proportion_sample_size(
    0.25, 0.35, allocation_ratio=1, objective="equivalence", margin=0.2
)
assert equivalence.group_sizes == (257, 257)
print(
    binary_proportion_power(
        [100, 200, 300],
        0.25,
        0.35,
        treatment_size=[100, 200, 300],
        objective="equivalence",
        margin=0.2,
    )
)
```

Without a ratio, `group_sizes` contains treatment enrollment; historical controls
are not enrolled. A ratio selects independent groups and means **treatment/control**.
For each integer control size, treatment size is `ceil(ratio*control)`, with actual
sizes used in power. Ratios range from `1e-6` to `1e6`. Wald power is monotone;
score searches scan vectorized batches to retain effects of changing integer
allocation. `max_size` bounds single/control size, default ten million.

## Exact Fisher designs

```python
from mdanderson_stats import fisher_sample_size, fisher_power

design = fisher_sample_size(0.35, 0.6)
assert design.group_sizes == (56, 56)
print(design.power, design.previous_power)  # 0.810914, 0.799683
print(fisher_power(56, 56, [0.35, 0.35], [0.35, 0.6]))
```

`alternative="greater"` means treatment has a higher response rate; `"less"`
means lower. `"two-sided"` uses probability-ordered hypergeometric p-values, with
relative mass tolerance `1e-12` to include numerical ties, consistent with the
package's Fisher analysis. Each design's power sums independent binomial
probabilities over every rejected table. This is unconditional power of the
conditional test: enrollment is fixed, but total responses are random.

Response-rate arrays broadcast and may include zero and one. Group sizes are
scalar integers from 1 to 2,000. The search scans every control size through
`max_size` (default 500), retaining oscillations in exact-test power. Treatment
size is `ceil(allocation_ratio*control)` and must also be at most 2,000. Infeasible
bounded searches raise an error. The example above took about 0.2 seconds locally;
large enumeration searches cost more. This is numerical exact enumeration, not
a normal approximation or simulation.

## Paired shifts and rater agreement

```python
from mdanderson_stats import mcnemar_sample_size, mcnemar_power, kappa_sample_size

assert mcnemar_sample_size(0.2, 0.5).group_sizes == (46,)
print(mcnemar_power([30, 46, 60], 0.2, 0.5))
assert kappa_sample_size(0.4, 0.3, 0.3, 0.5).group_sizes == (136,)
```

McNemar inputs are the probabilities of success-to-failure (`loss_probability`)
and failure-to-success (`gain_probability`) **within complete pairs**, not
marginal arm response rates. The planning formula uses discordance `s=loss+gain`,
shift `d=gain-loss`, null variance s and alternative variance `s-d**2`. One-sided
power tests gain > loss; two-sided planning uses the absolute shift and alpha/2.
Probabilities must form a valid paired population with positive shift variance.

Kappa inputs give each rater's positive marginal probability, null kappa and
alternative kappa. The two binary joint tables are uniquely reconstructed from
these margins and kappas; infeasible negative cell probabilities are rejected
(roundoff up to `1e-14` is tolerated). Cantor planning uses separate asymptotic
variances under the null and alternative, reusing the package's stable
multinomial delta-method implementation. Degenerate zero-variance populations
are rejected. One-sided tests require an increase in kappa. Both paired methods
return numbers of subjects observed twice, not twice as many subjects, and their
power calculations broadcast over scenarios. Their powers are approximations.

All result objects report group sizes, total enrollment, power, target and method.
`previous_power` evaluates the immediately preceding reference size and is absent
at size one. Defaults are alpha .05 and target power .8; alpha must be in (0, .5).
For exact one-sample binomial design, use the existing `binomial_sample_size` API.

## Source interpretation and validation

The [one-sample document](https://biostatistics.mdanderson.org/shinyapps/Nbinary/One-sample-help.pdf)
and [two-sample document](https://biostatistics.mdanderson.org/shinyapps/Nbinary/Two-sample-help.pdf)
describe large-sample planning. The latter swaps one-/two-sided quantile labels
and contains inconsistent equivalence scaling. We use the stated hypotheses,
proper standard errors and the application's worked examples. The
[McNemar document](https://biostatistics.mdanderson.org/shinyapps/Nbinary/McNemar-Test-Help.pdf)
defines paired discordance; the
[kappa document](https://biostatistics.mdanderson.org/shinyapps/Nbinary/Cohen-Kappa-Help.pdf)
describes Cantor's variance-based design. Formula typos are not reproduced.

Fifteen focused tests reproduce all twelve help examples, compare all Fisher
tails to independent rational enumeration of every small table, verify null
rejection control in that example, check kappa variance by multinomial simulation,
and verify integer allocation and infeasible populations. The statistical APIs
cover the calculator's test families; the Shiny interface and download layouts
are not reproduced.
