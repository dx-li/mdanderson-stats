"""Save/reopen a PCA-hybrid study and produce qualified reports on synthetic curves."""

import json

import numpy as np

from mdanderson_stats import WFMMStudy, read_wfmm_study, run_wfmm_study

y = np.array(
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
    y,
    np.ones((8, 1)),
    transform="PCw",
    alpha_pc=0.9,
    draws=32,
    warmup=16,
    seed=212,
    prediction_design=[[1.0]],
    prediction_residual=True,
    prediction_strata=[0],
)
saved = study.to_json()
reopened = read_wfmm_study(saved)
assert reopened.to_json() == saved
result = run_wfmm_study(reopened)
assert result.fit is not None, result.initialization.holds
report = json.loads(result.report_json())
assert report["status"] == "sampled"
assert report["posterior_precision"] == "not_certified"
assert len(result.curve_csv().splitlines()) == 5
print("Retained coefficient indices:", report["retained_indices"])
print("Curve mean:", report["curves"]["mean"])
print("Predictive mean:", report["prediction"]["mean"])
print("Short chains exercise the saved workflow; they do not certify precision.")
