# MDS-HOPE: published relative-risk score

This module implements Supplemental Equation S1 from Chien et al.,
[*Performance of molecular scoring systems in hypomethylating agent-treated
myelodysplastic neoplasms*](https://doi.org/10.1038/s41375-026-02895-5).
It computes the published Cox linear predictor, each predictor's contribution,
and a hazard ratio against an explicitly supplied reference profile.

The publisher supplement does **not** give the numeric encoding of the five
cytogenetic categories, the training-score mean and standard deviation, or a
baseline survival curve. Consequently, the module requires an already encoded
cytogenetic score and does not produce an absolute survival probability or
claim equivalence to the deployed calculator. Entry 171 remains partial.

## Inputs and relative comparisons

Age is age at diagnosis in years. ANC and platelets use 10^9/L, hemoglobin uses
g/dL, and marrow blasts use percentage points (7 means 7%). The four mutation
indicators must be explicitly supplied as 0 or 1. `tp53_hit_count` encodes wild
type as 0, single hit as 1, and multi-hit as 2; it is not an unrestricted count.
Missing or unknown mutation results are not imputed.

```python
from dataclasses import replace
import numpy as np
from mdanderson_stats import MDSHopeCovariates, mds_hope_score

# Synthetic arithmetic example. The numeric cytogenetic input is deliberately
# unlabelled: this does not establish the original application's category map.
reference = MDSHopeCovariates(
    age_years=70, anc_10e9_l=1.13, hemoglobin_g_dl=9.1,
    platelets_10e9_l=79, marrow_blast_percent=7,
    cytogenetic_risk_score=3,
    sf3b1_mutation=0, ezh2_mutation=0, tp53_hit_count=0,
    kras_mutation=0, ptpn11_mutation=0,
)
profiles = replace(reference, tp53_hit_count=[0, 1, 2])
result = mds_hope_score(profiles, reference=reference)
np.testing.assert_allclose(result.log_hazard_ratio, [0, 0.345, 0.690])
np.testing.assert_allclose(result.relative_hazard, np.exp([0, 0.345, 0.690]))
assert result.contributions.shape == (3, 11)
```

Inputs can be scalars or equally sized one-dimensional arrays; scalars and
one-element arrays broadcast across patients. Results are immutable arrays.
Scoring batches are limited to 250,000 patients before score matrices are built.
The linear predictor is not a probability. A relative hazard compares only
the supplied profiles under the published equation.

The raw-unit equation is

```text
eta = .025*age + .067*ANC - .160*hemoglobin - .0021*platelets
      + .051*marrow_blast_percent + .284*cytogenetic_risk_score
      - .294*SF3B1 + .347*EZH2 + .345*TP53 + .681*KRAS + 1.094*PTPN11.
```

The positive ANC term follows the source exactly: its neutropenia variable is
negative ANC, with coefficient −.067. Its anemia and thrombocytopenia variables
are negative hemoglobin and negative platelets divided by ten. The published
coefficients have three decimal places; unavailable full-precision fitted
coefficients are not reconstructed.

## Explicit standardization and risk groups

`mds_hope_risk_groups` accepts **already standardized** scores and returns codes
0 through 5 by default. Their labels and intervals are:

| Code | Group | Standardized score |
| ---: | --- | --- |
| 0 | very low | z ≤ −1.5 |
| 1 | low | −1.5 < z ≤ −0.5 |
| 2 | intermediate low | −0.5 < z ≤ 0 |
| 3 | intermediate high | 0 < z ≤ 0.5 |
| 4 | high | 0.5 < z ≤ 1.5 |
| 5 | very high | z > 1.5 |

Both grouping functions also accept `groups=5` for the supplement's alternative
classification. It merges the six-group intermediate-high and high categories
into `intermediate`, with interval `0 < z ≤ 1.5`. The first three groups keep
their intervals and codes; very high becomes code 4. The six-group default
is unchanged. A classification result includes `group_count` and the labels
corresponding to its codes.

`mds_hope_standardized_risk_groups` computes `(raw_score - reference_center) /
reference_sd` using the caller's explicit reference constants, then applies
these cutoffs. The prediction cohort is never used to infer the constants.

```python
from mdanderson_stats import (
    mds_hope_risk_groups, mds_hope_standardized_risk_groups,
)

z = np.array([-2, -1.5, -0.5, 0, 0.5, 1.5, 2])
np.testing.assert_array_equal(mds_hope_risk_groups(z), [0, 0, 1, 2, 3, 4, 5])
# Arbitrary constants for this arithmetic demonstration, not study calibration.
classified = mds_hope_standardized_risk_groups(
    0.25 + 2*z, reference_center=0.25, reference_sd=2,
)
np.testing.assert_array_equal(classified.group_code, mds_hope_risk_groups(z))
five = mds_hope_standardized_risk_groups(
    0.25 + 2*z, reference_center=0.25, reference_sd=2, groups=5,
)
assert five.group_count == 5
np.testing.assert_array_equal(five.group_code, [0, 0, 1, 2, 3, 3, 4])
assert five.group_labels[3] == "intermediate"
```

## Numerical scope

Inputs must be finite. Hazard-ratio overflow raises an explicit error;
underflow may return zero while retaining the finite log hazard ratio.
Reference comparisons use predictor differences before coefficient weighting.
Independent base-R fixtures cover every coefficient, TP53 coding, reference
comparisons, and exact/neighboring boundaries for both grouping schemes. The
reference generator and
synthetic fixtures are in `tools/reference_mds_hope.R` and
`tests/fixtures/mds-hope/`.

These checks validate the recovered equation and cutoffs. They do not establish
the missing app calibration, prediction accuracy in a new population, or native
plot/report parity. See the [source audit](../research/mds-hope-source-status.md).
