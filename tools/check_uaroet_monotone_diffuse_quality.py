"""Independent two-dimensional quadrature for monotone diffuse-prior inference."""

import json

import numpy as np
from scipy.integrate import quad
from scipy.special import logit

from mdanderson_stats import fit_uaroet
from mdanderson_stats.hierarchical_binomial import summarize_chains


def density(p, q):
    a, b = logit(p), logit(q)
    log_beta = 49 * (np.log(4 * p * (1 - p)) + np.log(4 * q * (1 - q)))
    return np.exp(log_beta - (a * a + (b - a) ** 2) / 20000)


def integral(power, which):
    def outer(p):
        return quad(
            lambda q: density(p, q) * (p if which == 0 else q) ** power,
            p,
            1,
            epsabs=1e-11,
            epsrel=1e-9,
        )[0]

    return quad(outer, 0, 1, epsabs=1e-11, epsrel=1e-9)[0]


z = integral(0, 0)
expected = np.array([integral(1, i) / z for i in range(2)])
fit = fit_uaroet(
    np.full((2, 2, 2), 25),
    prior_mean=[0] * 4,
    prior_sd=[100] * 4,
    initial_parameters=[-0.1, 0.2, -0.1, 0.2],
    association=0,
    draws=3000,
    warmup=1000,
    chains=4,
    rng=np.random.default_rng(2026),
)
probability = fit.joint.sum(axis=3)[..., 1]
s = summarize_chains(probability)
result = dict(
    scope="two-dose binary ordered diffuse-prior posterior, fixed independence",
    reference_means=expected.tolist(),
    posterior_means=probability.mean(axis=(0, 1)).tolist(),
    mean_absolute_errors=abs(probability.mean(axis=(0, 1)) - expected).tolist(),
    mcse=s.batch_mean_mcse.tolist(),
    split_rhat=s.split_rhat.tolist(),
    maximum_parameter_rhat=float(np.max(fit.parameter_summary.split_rhat)),
    draws=3000,
    chains=4,
    warmup=1000,
    clinical_validation=False,
)
assert max(result["mean_absolute_errors"]) < 0.004
assert max(result["split_rhat"]) < 1.05
print(json.dumps(result, indent=2))
