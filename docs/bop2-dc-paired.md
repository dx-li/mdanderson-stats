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

`design.operating_characteristics(category_probability)` computes exact trial
outcome probabilities from four joint cell probabilities in the same order as
observed counts. Supply the joint distribution explicitly; two marginal rates
do not determine the association between endpoints.

```python
oc = design.operating_characteristics([0.06, 0.54, 0.04, 0.36])
print(oc.final_go, oc.final_consider, oc.no_go_probability)
print(oc.expected_sample_size)
```

`stop_no_go` gives early no-go probability at each look, with zero in its final
entry. `final_no_go` excludes earlier stops; `no_go_probability` includes them.
`sample_size_probability` gives the probability of stopping at each configured
look, including every final decision. Expected sample size includes early
stopping.

A forward recursion tracks the two marginal event counts, using four joint
outcome transitions. It preserves endpoint dependence without allocating the
full table of four-category counts. Work is bounded before state allocation by
`number_of_scenarios * (max_subjects + 1)**3 <= 5_000_000`, allowing up to 169
subjects for one scenario. Split a large batch if necessary; larger individual
designs exceed this OC limit even though monitoring supports them. This calculation assumes fully
observed outcomes at the configured enrollment looks and does not model
calendar-time accrual or delayed observations.

Paired parameter calibration remains pending.
See [source notes](bop2-dc-paired-source.md) and
[independent references](bop2-dc-reference.md).
