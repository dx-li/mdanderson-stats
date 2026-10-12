"""Automatic WFMM initialization, sampling and reconstruction on synthetic data.

Short chains exercise the workflow; they do not establish posterior precision.
"""

import numpy as np

from mdanderson_stats import (
    fit_wfmm_coefficients,
    initialize_wfmm,
    wfmm_basis,
    wfmm_summarize,
)

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
x = np.ones((8, 1))
z = np.repeat(np.eye(4), 2, axis=0)
initialization = initialize_wfmm(coefficients, x, z)
if not initialization.ready_for_sampling:
    raise RuntimeError(initialization.holds)
state = initialization.conditional
fit = fit_wfmm_coefficients(
    coefficients,
    x,
    z,
    prior=initialization.prior,
    random_variance=state.random_variance,
    residual_variance=state.residual_variance,
    proposal_sd=(state.proposal_sd[:1], state.proposal_sd[1:]),
    draws=32,
    warmup=16,
    chains=2,
    rng=np.random.default_rng(177),
)
summary = wfmm_summarize(fit.coefficients, wfmm_basis(4, transform="identity"))
print("Initialization ready:", initialization.ready_for_sampling)
print("Reconstructed mean:", summary.mean.tolist())
legacy = initialize_wfmm(coefficients, x, z, profile="legacy_3_1")
assert not legacy.ready_for_sampling
print("Explicit native holds:", legacy.holds)
