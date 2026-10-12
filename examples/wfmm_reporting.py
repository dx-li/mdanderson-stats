"""Saved regional summaries and predictions for existing and shared new levels."""

import numpy as np

from mdanderson_stats import WFMMStudy, read_wfmm_study, run_wfmm_study

observed = np.array(
    [
        [1.0, 4, 2, 3],
        [2, 2, 3, 1],
        [4, 3, 1, 5],
        [5, 6, 4, 2],
        [6, 3, 7, 4],
        [3, 1, 2, 6],
        [2, 5, 5, 3],
        [4, 2, 6, 2],
    ]
)
study = WFMMStudy(
    observed,
    np.ones((8, 1)),
    random_design=np.repeat(np.eye(4), 2, axis=0),
    transform="identity",
    draws=16,
    warmup=8,
    seed=212,
    time_contrast=[[0.5, 0.0], [0.5, 0.0], [0.0, 0.5], [0.0, 0.5]],
    effect_contrast=[[1.0], [-1.0]],
    confidence=0.9,
    effect_sizes=[0.0, 0.5],
    summarize_random_effects=True,
    include_covariance=True,
    prediction_design=np.ones((3, 1)),
    prediction_existing_random_design=[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    prediction_new_random_design=[[0.0], [1.0], [1.0]],
    prediction_new_random_strata=[0],
)
result = run_wfmm_study(read_wfmm_study(study.to_json()))
print(result.report_json())
if result.fit is not None:
    print(result.curve_csv())
