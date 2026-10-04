# Interaction-index case studies

This guide runs the two factual examples in Lee and Kong (2009), Section 4.2,
Tables 2 and 3, and saves a median-effect panel, an interaction-index panel,
and a compact JSON record for each case. The cited cases are UMSCC22B cells
with SCH66336 and 4-HPR, and the o-phenanthroline/ADP alcohol-dehydrogenase
study. The source is [Lee and Kong (2009)](https://doi.org/10.1198/sbr.2009.0001);
the cached paper text and readme are listed in
[`interaction-index-sources.json`](interaction-index-sources.json).

Run from a writable working directory in an environment with the package and
Matplotlib installed. Both single-agent curves and the fixed-ratio mixture curve are fit
to `logit(response)` against `log(dose)`. For the mixture, the x values are
**total doses** along the stated ray. The response is kept as printed: the
first case is fractional survival, not one minus survival; the second is
fractional inhibition. The first case's doses are in μM. The paper does not
state units for the second case, so the plots leave the dose label generic.

The left panel reproduces the paper's median-effect display in transformed
coordinates. The right panel shows the fixed-ray interaction-index curve with
pointwise log-delta intervals and the observed-combination indices as points
with pointwise intervals from the documented Python pooled-error fallback.
The displayed effect grid stays within the observed mixture-response range;
no curve extrapolation is drawn. The intervals are pointwise, not simultaneous.
This is a data-and-delta-analysis reproduction, not the paper's full
Monte-Carlo figure reproduction. The native pooled-error denominator is not
specified in the recovered source; therefore the observed-point intervals may
differ from the printed intervals. Printed rounded estimates are retained in
the fixture for comparison, not used to force a fit.

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mdanderson_stats import (
    fit_median_effect,
    interaction_index_pooled_error,
    interaction_index_ray,
    plot_interaction_index,
    plot_median_effect,
)

cases = [
    {
        "id": "umscc22b-sch66336-4hpr",
        "label": "UMSCC22B: SCH66336 + 4-HPR",
        "response_label": "Fractional survival",
        "dose_unit": "μM",
        "components": ["SCH66336", "4-HPR"],
        "single_doses": [[0.1, 0.5, 1, 2, 4], [0.1, 0.5, 1, 2]],
        "single_responses": [[0.6701, 0.6289, 0.5577, 0.4550, 0.3755], [0.7666, 0.5833, 0.5706, 0.4934]],
        "combination_total_dose": [0.2, 1, 2, 4],
        "combination_response": [0.6539, 0.4919, 0.3551, 0.2341],
        "ray": [1, 1],
    },
    {
        "id": "phenanthroline-adp",
        "label": "o-Phenanthroline + ADP",
        "response_label": "Fractional inhibition",
        "dose_unit": None,
        "components": ["o-Phenanthroline", "ADP"],
        "single_doses": [[8.7, 17.4, 26.1, 34.8, 43.5], [0.5, 1, 1.5, 2, 2.5]],
        "single_responses": [[0.132, 0.267, 0.411, 0.476, 0.548], [0.175, 0.400, 0.492, 0.542, 0.592]],
        "combination_total_dose": [9.2, 18.4, 27.6, 36.8, 46],
        "combination_response": [0.507, 0.769, 0.872, 0.919, 0.944],
        "ray": [17.4, 1],
    },
]

output = Path("interaction-index-case-studies")
output.mkdir(exist_ok=True)
source_url = "https://doi.org/10.1198/sbr.2009.0001"
for case in cases:
    single_fits = [
        fit_median_effect(dose, response)
        for dose, response in zip(case["single_doses"], case["single_responses"], strict=True)
    ]
    mixture_fit = fit_median_effect(case["combination_total_dose"], case["combination_response"])
    effects = np.linspace(min(case["combination_response"]), max(case["combination_response"]), 100)
    ray = interaction_index_ray(single_fits, mixture_fit, case["ray"], effects)
    components = np.asarray(case["combination_total_dose"])[:, None] * (
        np.asarray(case["ray"]) / sum(case["ray"])
    )
    observed = interaction_index_pooled_error(
        single_fits, components, case["combination_response"]
    )

    fig, (median_ax, index_ax) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for label, fit, dose, response in zip(
        [*case["components"], "Fixed-ratio mixture"],
        [*single_fits, mixture_fit],
        [*case["single_doses"], case["combination_total_dose"]],
        [*case["single_responses"], case["combination_response"]],
        strict=True,
    ):
        plot_median_effect(fit, dose, response, ax=median_ax, label=label)
    median_ax.set_title(case["label"])
    median_ax.set_xlabel("log(dose" + (f" [{case['dose_unit']}])" if case["dose_unit"] else ")"))
    median_ax.set_ylabel(f"logit({case['response_label'].lower()})")
    median_ax.legend(fontsize="small")

    plot_interaction_index(effects, ray, ax=index_ax, label="Fixed-ray curve")
    plot_interaction_index(
        case["combination_response"], observed, ax=index_ax,
        label="Observed (pooled error)", kind="points",
    )
    index_ax.set_xlabel(case["response_label"])
    index_ax.legend(fontsize="small", loc="upper left")
    fig.savefig(output / f"{case['id']}.png", dpi=160)
    plt.close(fig)

    fits = [*single_fits, mixture_fit]
    record = {
        "case": case["id"],
        "source_url": source_url,
        "inputs": case,
        "response_scale": case["response_label"],
        "dose_unit": case["dose_unit"],
        "fits": [
            {
                "intercept": fit.intercept,
                "slope": fit.slope,
                "covariance": fit.covariance.tolist(),
                "observations": fit.observations,
                "residual_variance": fit.residual_variance,
            }
            for fit in fits
        ],
        "fixed_ray": {
            "effect": effects.tolist(),
            "log_index": ray.log_index.tolist(),
            "log_interval": ray.log_interval.tolist(),
            "degrees_of_freedom": ray.degrees_of_freedom,
            "confidence": ray.confidence,
        },
        "observed_pooled_python_fallback": {
            "variance_convention": "single-agent residual mean squares weighted by residual df (n-2); native pooling denominator unspecified",
            "effect": case["combination_response"],
            "index": observed.index.tolist(),
            "interval": observed.interval.tolist(),
            "degrees_of_freedom": observed.degrees_of_freedom,
            "confidence": observed.confidence,
        },
    }
    (output / f"{case['id']}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
```

The source data and rounded printed fit summaries are also recorded in
[`tests/fixtures/interaction_index_cases.json`](../tests/fixtures/interaction_index_cases.json).
That fixture supports a focused check that the recovered table observations
produce median-effect estimates consistent with the rounded values printed in
Tables 2 and 3. Full-precision fit, observed-point, and ray references come
from the separate base-R calculation in
[`tools/reference_interaction_index_cases.R`](../tools/reference_interaction_index_cases.R).
For example, its first Table 2 lower limit is 0.202556, while the paper prints
0.202; source rows and tabulated results are rounded, so printed-value checks
allow that precision difference. The case-study coverage does not establish
native application file-format compatibility or equality to the paper's
Monte-Carlo curves.
