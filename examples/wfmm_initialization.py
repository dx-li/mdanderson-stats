"""MOM and native EB diagnostics; not an automatically calibrated WFMM study."""

import numpy as np

from mdanderson_stats import wfmm_mom_initialization, wfmm_native_empirical_bayes

coefficients = np.array(
    [
        [1.0, 4.0, 2.0, 3.0],
        [2.0, 2.0, 3.0, 1.0],
        [4.0, 3.0, 1.0, 5.0],
        [5.0, 6.0, 4.0, 2.0],
        [6.0, 3.0, 7.0, 4.0],
        [3.0, 1.0, 2.0, 6.0],
        [2.0, 5.0, 5.0, 3.0],
        [4.0, 2.0, 6.0, 2.0],
    ]
)
fixed = np.ones((8, 1))
mom = wfmm_mom_initialization(coefficients, fixed)
assert mom.converged
# Intercept-only conditional sampling variance equals residual variance/N.
standardized = mom.fixed_effect_estimates / np.sqrt(mom.residual_variance / 8)
eb = wfmm_native_empirical_bayes(standardized, profile="legacy_3_1")
assert eb.converged
print("MOM variance:", mom.residual_variance.tolist())
print("Native EB ratio:", eb.group_slab_to_sampling_variance.tolist())
zero = wfmm_native_empirical_bayes(np.zeros((1, 4)), profile="legacy_3_1")
assert zero.legacy_stopping_satisfied and not zero.converged
print("Zero-signal native stop is not convergence:", zero.maximum_change)
