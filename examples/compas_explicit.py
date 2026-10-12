"""Synthetic interim data with explicit, illustrative grouping and model priors."""

import json

from mdanderson_stats import ComPASStudy, compas_pnoc022_variances

study = ComPASStudy(
    completed=(11, 11, 11),
    successes=(4, 5, 8),
    groups=("explicit_group",) * 3,
    variances=compas_pnoc022_variances("cohorts_1_2"),
    model_prior=(0.2,) * 5,  # Caller choice, not a recovered source default.
    null_response=0.4,
    equality_policy="hold",
)
fit = ComPASStudy.from_json(study.to_json()).fit()
assert fit.selected_model == 3
assert abs(sum(fit.model_probabilities) - 1) < 1e-12
assert json.loads(fit.to_json())["study"] == json.loads(study.to_json())
print(fit.to_json())
