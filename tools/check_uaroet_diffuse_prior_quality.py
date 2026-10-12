"""Independent quadrature for a diffuse-prior, fixed-independence posterior."""

import json

import numpy as np
from scipy.integrate import quad
from scipy.special import expit, log_expit

from mdanderson_stats import fit_uaroet
from mdanderson_stats.hierarchical_binomial import summarize_chains


def density(x):
    return np.exp(50 * (log_expit(x) + log_expit(-x)) + 100 * np.log(2) - x * x / 20000)


z = quad(density, -10, 10, epsabs=1e-12)[0]
mean = quad(lambda x: expit(x) * density(x), -10, 10, epsabs=1e-12)[0] / z
second = quad(lambda x: expit(x) ** 2 * density(x), -10, 10, epsabs=1e-12)[0] / z
fit = fit_uaroet(
    np.full((1, 2, 2), 25),
    prior_mean=[0, 0],
    prior_sd=[100, 100],
    initial_parameters=[0, 0],
    association=0,
    draws=2000,
    warmup=500,
    chains=4,
    rng=np.random.default_rng(7304),
)
probability = fit.joint.sum(axis=3)[..., 0, 1]
s = summarize_chains(probability)
result = dict(
    scope="one-dose binary diffuse-prior posterior, fixed independence",
    reference_mean=mean,
    reference_variance=second - mean * mean,
    posterior_mean=float(probability.mean()),
    posterior_variance=float(probability.var(ddof=1)),
    probability_mcse=float(s.batch_mean_mcse),
    probability_split_rhat=float(s.split_rhat),
    maximum_parameter_rhat=float(np.max(fit.parameter_summary.split_rhat)),
    draws=2000,
    chains=4,
    warmup=500,
    clinical_validation=False,
)
assert abs(result["posterior_mean"] - mean) < 0.005
assert abs(result["posterior_variance"] / (second - mean * mean) - 1) < 0.08
assert result["probability_split_rhat"] < 1.03
print(json.dumps(result, indent=2))
