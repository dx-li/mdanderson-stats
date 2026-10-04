# Published BARD response scenarios

`bard_response_scenario` returns the paper's dose-specific toxicity truth,
printed marginal response truth, and the response-model intercepts for one of
12 published scenarios:

```python
from mdanderson_stats import bard_response_scenario

scenario = bard_response_scenario("five-dose-2")
print(scenario.toxicity_probabilities)
print(scenario.published_response_probabilities)
print(scenario.response_intercepts)
print(scenario.response_coefficients)
```

IDs `five-dose-1` through `five-dose-8` correspond to Table 3 and Supplementary
Table S1. IDs `three-dose-1` through `three-dose-4` correspond to Supplementary
Tables S4 and S5. Dose levels are ordinal labels, not physical dose values.
The response coefficients are `(1.7, -1.5, 0.4)` and the source gives each of
three binary factors marginal probability 0.5 for either coded level 1 or 2.
The source does not specify their joint dependence, so the record does not
claim independence. A product-profile marginal calculation is a separate,
explicit modeling convention; it should not overwrite the printed marginal
response probabilities.

These objects are source-data records, not complete trial scenarios. They do
not infer the true OBD, tie handling, an allocation policy, dose amounts, or a
joint toxicity-response distribution. The paper's utility row is intentionally
not included because reconstructing it would require an unstated association
between toxicity and response. See the [source audit](../research/bard-response-scenarios-audit.md)
for table-level crosswalks and rounding notes.
