"""Explicit shared shrinkage priors; synthetic data, not clinical validation."""

import numpy as np

from mdanderson_stats import WFMMPrior, fit_wfmm_coefficients, wfmm_mode_shrinkage_hyperprior

prior = wfmm_mode_shrinkage_hyperprior(0.2, 1, probability_variance=0.06, slab_variance=10)
fit = fit_wfmm_coefficients(
    [[1.5, -0.2], [1.2, 0.1], [1.8, -0.5], [1.5, -0.2]],
    np.ones((4, 1)),
    prior=WFMMPrior(0.4, 1),
    shrinkage_hyperprior=prior,
    residual_variance=1,
    estimate_variances=False,
    draws=100,
    warmup=50,
    rng=np.random.default_rng(20261012),
)
print("Retained partition labels:", fit.hyperparameter_partitions)
print("Posterior inclusion-probability mean:", fit.inclusion_probabilities.mean(axis=(0, 1)))
print("Posterior slab-variance mean:", fit.slab_variances.mean(axis=(0, 1)))
