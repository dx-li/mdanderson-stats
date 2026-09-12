# Paired-endpoint BOP2-DC

`bop2_dc_paired_design` supports multiple efficacy endpoints and joint
efficacy/toxicity monitoring. Both use a four-cell Dirichlet prior and derive
the two marginal Beta posteriors from it.

Counts must be in this order: **both events, first only, second only,
neither**. For efficacy/toxicity, the first event is efficacy and the second
is toxicity. Do not pass two marginal totals as four-category counts.

```python
from mdanderson_stats import bop2_dc_paired_design

design = bop2_dc_paired_design(
    20,
    "efficacy_toxicity",
    lrv=[0.3, 0.2],
    cmv=[0.45, 0.15],
    prior=[0.25, 0.25, 0.25, 0.25],
    looks=[10, 20],
)
state = design.monitor([2, 10, 0, 8])  # 12 responses and 2 toxicities in 20 patients
print(state.decision)  # final_consider
```

For multiple efficacy endpoints, either endpoint can support a final go,
while both must support no-go to stop. For efficacy/toxicity, both must
support go and either can trigger no-go. Other final combinations yield
consider. Only configured interim looks can stop early.

Each endpoint has its own two posterior cutoffs and information exponents.
Toxicity is evaluated in the lower direction: its clinically meaningful
threshold must be below its reference threshold. Efficacy uses the upper
direction. `marginal_posterior` has trailing axes `(endpoint, criterion)`,
where the criteria are LRV then CMV; its toxicity entries are safety
probabilities, not probabilities of exceeding the toxicity thresholds.

The default prior has effective sample size one and independent cell
probabilities centered on the supplied reference rates. An explicit four-cell
prior can express a different prior association. Decision composition uses
the marginal posteriors; it does not multiply them or assume posterior
independence.

Paired operating-characteristic calculations and calibration remain pending.
See [source notes](bop2-dc-paired-source.md) and
[independent references](bop2-dc-reference.md).
