"""Independent six-dimensional importance check against the Python slice sampler.

Uses randomized Sobol gamma-prior importance integration with independently
written odds/likelihood algebra. This is a bounded diagnostic experiment, not
native RNG identity, universal mixing assurance, or clinical validation.
"""

import json

import numpy as np
from scipy.special import gammaincinv
from scipy.stats import qmc

from mdanderson_stats import ToxFinderPrior, fit_toxfinder
from mdanderson_stats.hierarchical_binomial import summarize_chains

prior = ToxFinderPrior(np.ones(6), np.full(6, 0.25))
doses = np.array([[0.15, 0.15], [0.3, 0.5], [0.6, 0.3], [0.8, 0.8]])
y, n = np.array([0, 1, 1, 3]), np.array([2, 3, 3, 4])
reference = []
effective_sizes = []
for seed in range(8):
    unit = qmc.Sobol(6, scramble=True, seed=90210 + seed).random_base2(15)
    theta = gammaincinv(4.0, unit) / 4.0
    a1, b1, a2, b2, a3, b3 = theta.T
    x1 = doses[:, 0][None, :] ** b1[:, None]
    x2 = doses[:, 1][None, :] ** b2[:, None]
    odds = a1[:, None] * x1 + a2[:, None] * x2 + a3[:, None] * (x1 * x2) ** b3[:, None]
    log_likelihood = (y * np.log(odds) - n * np.log1p(odds)).sum(axis=1)
    weights = np.exp(log_likelihood - log_likelihood.max())
    weights /= weights.sum()
    effective_sizes.append(float(1 / (weights @ weights)))
    reference.append(weights @ np.column_stack((np.log(theta), odds / (1 + odds))))
reference = np.array(reference)
reference_mean = reference.mean(axis=0)
reference_se = reference.std(axis=0, ddof=1) / np.sqrt(len(reference))
fit = fit_toxfinder(
    doses,
    y,
    n,
    prior=prior,
    rng=np.random.default_rng(20261012),
    draws=2000,
    warmup=1000,
    chains=4,
    max_evaluations=20_000_000,
)
summary = summarize_chains(np.concatenate((fit.log_parameters, fit.probabilities[..., 1]), axis=2))
combined_se = np.sqrt(reference_se**2 + summary.batch_mean_mcse**2)
z = np.abs(summary.mean - reference_mean) / combined_se
result = {
    "method": "8 independently scrambled Sobol gamma-prior importance integrals",
    "points_per_scramble": 32768,
    "slice_draws_per_chain": 2000,
    "slice_warmup_per_chain": 1000,
    "slice_chains": 4,
    "prior_mean": prior.mean.tolist(),
    "prior_variance": prior.variance.tolist(),
    "doses": doses.tolist(),
    "toxicities": y.tolist(),
    "subjects": n.tolist(),
    "columns": [
        "log_alpha1",
        "log_beta1",
        "log_alpha2",
        "log_beta2",
        "log_alpha3",
        "log_beta3",
        "risk1",
        "risk2",
        "risk3",
        "risk4",
    ],
    "reference_mean": reference_mean.tolist(),
    "reference_standard_error": reference_se.tolist(),
    "slice_mean": summary.mean.tolist(),
    "slice_batch_mcse": summary.batch_mean_mcse.tolist(),
    "slice_split_rhat": summary.split_rhat.tolist(),
    "absolute_difference_in_combined_se": z.tolist(),
    "importance_effective_sizes": effective_sizes,
    "objective_evaluations": fit.evaluations,
    "passed": bool(np.max(z) < 5 and np.max(summary.split_rhat) < 1.05),
}
print(json.dumps(result, indent=2))
assert result["passed"], "joint posterior comparison requires investigation"
