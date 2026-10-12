"""Illustrate disjoint complete and toxicity-only data; short chains are not precision evidence."""

import numpy as np

from mdanderson_stats import fit_uaroet

fit = fit_uaroet(
    counts=[[[4, 0], [2, 0]], [[1, 1], [3, 1]]],
    toxicity_only_counts=[[1, 0], [0, 2]],
    prior_mean=[0, 0.3, -1, 0.3],
    prior_sd=[0.7, 0.4, 0.7, 0.4],
    association=0,
    draws=64,
    warmup=32,
    chains=2,
    rng=np.random.default_rng(14),
)
assert not fit.direct_prior
print("Mean toxicity probability by dose:", fit.joint[..., 1].sum(axis=-1).mean(axis=(0, 1)))
print("Parameter diagnostics:", fit.parameter_summary)
