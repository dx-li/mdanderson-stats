# ASYPOW calculation workflow

`asypow_calculate` joins an existing ASYPOW information or SMO constructor to
the corresponding significance, power, and sample-size calculations. Each
request supplies exactly two target columns; the third is computed from the
constructed model. Existing lower-level constructors and their validation
remain the source of model mathematics.

```python
from mdanderson_stats.asypow_workflow import AsyPowCalculationRequest, asypow_calculate

result = asypow_calculate(
    AsyPowCalculationRequest(
        "lr_groups",
        {
            "parameters": [0.25, 0.48],
            "contrasts": [1, -1],
            "model": "binomial",
            "group_size": [2, 1],
        },
        power=[0.8, 0.9],
        significance=0.05,
    )
)
print(result.report())
```

The request settings are bound to the selected Python constructor. The report
records all bound settings, including defaults, together with the actual
method result, degrees of freedom, and the three-column target table. Settings
are snapshotted before model construction; arrays in the recorded settings
are owned and read-only. Input vectors must be scalar or one-dimensional and
paired elementwise, with scalar broadcasting, under the same 10,000-target
limit used by ASYPOW sample-size inversion. The report is capped at two
megabytes.

| Procedure | Model arguments | Existing calculation used |
| --- | --- | --- |
| `lr_information` | `parameters`, `information`, `contrasts`, optional `null_values` and `information_observations` | `asypow_information`; information is for the specified number of observations and is normalized by `information_observations` |
| `lr_groups` | `parameters`, `contrasts`; optional group `model`, `null_values`, `group_size`, `duration` | group-information constructor, then `asypow_information` |
| `lr_regression` | `parameters`, `covariates`, `contrasts`; optional `family`, `null_values`, `observations`, `group_size`, `duration` | regression-information constructor, then `asypow_information` |
| `lr_ordinal` | cumulative probabilities, `contrasts`, optional `null_values`, `group_size` | ordinal-information constructor, then `asypow_information` |
| `lr_ordinal_regression` | parameters, covariates, contrasts; optional `quadratic`, `link`, null and allocation settings | ordinal-regression information, then `asypow_information` |
| `lr_multinomial` | probabilities, contrasts, optional `null_values`, `group_size` | multinomial-information constructor, then `asypow_information` |
| `lr_design` | coefficients, design matrix, contrasts; optional `model`, `null_values`, `observations` | design-information constructor, then `asypow_information` |
| `asypow_smo_binomial` | probabilities and optional null/constraint/allocation settings | binomial SMO constructor |
| `asypow_smo_poisson` | means and optional null/constraint/allocation settings | Poisson SMO constructor |
| `asypow_smo_exponential` | rates, duration and optional null/allocation settings | exponential-survival SMO constructor |
| `asypow_smo_multinomial`, `asypow_smo_ordinal` | probability or cumulative-probability arrays and optional null/constraint settings | categorical SMO constructors |
| `asypow_smo_regression`, `asypow_smo_ordinal_regression` | model parameters, covariates, constraints and bounds | regression SMO constructors |
| `asypow_smo_design` | coefficients, design, constraints and bounds | general-design SMO constructor |
| `asypow_smo_generic` | parameters, callback, bounds, constraints and optimizer settings | generic callback SMO constructor |

The generic callback route records the callback's qualified name but cannot
capture closure state, so its report is not a serialization of that callback.
This workflow provides readable Python reports; the native ASYPOW program
prints interactively and does not define a saved-report file format. It does
not claim byte-for-byte console parity. The lower-level
`asypow_reparameterize(information, jacobian)` remains composable with
`lr_information`: callers supply the transformed parameter vector and
information matrix explicitly. See [the ASYPOW method guide](asypow.md) for
the underlying model assumptions and numerical conventions.
