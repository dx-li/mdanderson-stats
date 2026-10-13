"""Small saved image study; numerical demonstration, not a clinical application."""

import numpy as np

from mdanderson_stats import WFMMStudy, read_wfmm_study, run_wfmm_study

observed = np.random.default_rng(46).normal(size=(8, 16))
observed += 5 * np.random.default_rng(411).normal(size=16)
study = WFMMStudy(
    observed,
    np.ones((8, 1)),
    transform="spatial_wavelet",
    spatial_shape=(4, 4),
    draws=8,
    warmup=4,
    thin=2,
    optimizer_iterations=50,
    prediction_design=[[1]],
)
result = run_wfmm_study(read_wfmm_study(study.to_json()))
print(result.report_json())
if result.fit is not None:
    print(result.summary.mean[0].reshape(study.spatial_shape, order="F"))
