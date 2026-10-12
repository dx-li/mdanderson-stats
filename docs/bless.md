# BLESS published-table survival

`bless_published_prediction` implements all four rounded models in Molina et al.,
[Chest 2021](https://doi.org/10.1016/j.chest.2021.03.059), e-Appendix 1 and
 e-Tables 1–8: lung, breast, hematologic and general cancer with interactions
(GCMM). The profile is `published_tables_2021`. Catalog149 remains **partial**:
this published-grid workflow is available, while the deployed daily baseline,
full-precision fit and any additional uncertainty workflow are not recovered.

```python
from mdanderson_stats import BLESSPublishedStudy

study = BLESSPublishedStudy(
    "breast",
    {
        "ecog": 3,
        "benign_cytology": 0,
        "nlr": 10,
        "ldh_u_l": 900,
        "protein_g_dl": 2.8,
    },
)
reopened = BLESSPublishedStudy.from_json(study.to_json())
result = reopened.predict()
print(result.survival)  # 0, 30, 60, 90, 120, 150, 180 days
print(result.to_json())  # inputs, coefficients' profile, curve, provenance, limitations
print(result.to_csv())  # curve export; keep JSON for the complete audit
```

Time is days from first thoracentesis, not diagnosis. Only nonempty increasing
subsets of the seven published integer days are accepted. The result is
`S_reference(t) ** exp(eta)`, evaluated stably without overflowing the hazard
multiplier. No interpolation, extrapolation, median or individual confidence
interval is invented. Marginal coefficient HR intervals do not determine joint
survival uncertainty.

All inputs must be present and irrelevant inputs are rejected:

| Model | Required predictors |
|---|---|
| lung | ecog, surgery_within_30_days, bilateral, benign_cytology, ldh_u_l, protein_g_dl |
| breast | ecog, benign_cytology, nlr, ldh_u_l, protein_g_dl |
| hematologic | ecog, high_clinical_suspicion_pneumonia, bilateral, benign_cytology, protein_g_dl |
| gcmm | cancer, ecog, benign_cytology, nlr, bilateral, ldh_u_l, protein_g_dl |

ECOG is an integer 0–4. GCMM cancer is exactly `lung`, `breast` or
`hematologic`. Indicators are booleans or integers 0/1. **Benign cytology is1;
malignant cytology is0.** LDH is pleural-fluid U/L (division by100 happens inside
the equation), protein is pleural-fluid g/dL, and NLR is the blood
neutrophil/lymphocyte ratio. Continuous values must be finite and nonnegative;
these are mathematical input bounds, not validated clinical ranges. No missing
values are imputed. The study owns an immutable copy of its inputs, and its
versioned JSON rejects duplicate and unknown fields.

Source choices are included in every JSON report:

- Lung uses equation/table centers7.19 and3.7; earlier prose says7.31 and3.8.
- Breast uses NLR center5.398, not the baseline footnote's5.389. Its printed
  example survival contradicts the tables beyond ordinary rounding: the
  declared tables give30-day survival0.6312601492, not0.644. No coefficients
  have been adjusted to force agreement.
- Hematologic rounded coefficients give example LP−0.502; the example's
  −0.49662 is not evidence of known full-precision coefficients.
- GCMM's example requires bilateral1, omitted by its narrative. Age has no
  fitted term. Cancer-specific slopes and cancer/laterality terms are retained.

The independent reference generator uses60-digit Decimal arithmetic and the
structured source-table expressions, without importing package code. Its458
cases cover all categorical combinations, continuous changes, reference cases
and all four examples. See [source audit](../research/bless-published-parameters-audit.json)
and [recovery/validation record](../research/bless-published-recovery-2026-10-12.md).
Mathematical agreement does not establish clinical validation or exact deployed
application parity. The original PDF is not redistributed and existing package
licensing restrictions remain unchanged.
