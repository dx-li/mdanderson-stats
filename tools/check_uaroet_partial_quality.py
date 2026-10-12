"""Independent scalar-integration check for partial-only UAROET inference."""

import json

import numpy as np
from scipy.integrate import quad
from scipy.special import expit, log_expit
from scipy.stats import norm

from mdanderson_stats import fit_uaroet
from mdanderson_stats.hierarchical_binomial import summarize_chains


def density(x):
    return norm.pdf(x, loc=-0.5, scale=0.9) * np.exp(6 * log_expit(x) + 2 * log_expit(-x))


z = quad(density, -12, 12, epsabs=1e-12)[0]
expected = quad(lambda x: expit(x) * density(x), -12, 12, epsabs=1e-12)[0] / z
fit = fit_uaroet(
    np.zeros((1, 2, 2)),
    toxicity_only_counts=[[2, 6]],
    prior_mean=[-0.2, -0.5],
    prior_sd=[0.8, 0.9],
    association=0.0,
    draws=1500,
    warmup=500,
    chains=4,
    rng=np.random.default_rng(8301),
)
risk = fit.joint.sum(axis=3)[..., 0, 1]
s = summarize_chains(risk)
result = dict(
    scope="one-dose binary partial-only posterior at fixed zero association",
    reference_risk_mean=expected,
    posterior_risk_mean=float(risk.mean()),
    absolute_difference=abs(float(risk.mean()) - expected),
    risk_split_rhat=float(s.split_rhat),
    risk_mcse=float(s.batch_mean_mcse),
    maximum_parameter_rhat=float(np.max(fit.parameter_summary.split_rhat)),
    draws_per_chain=1500,
    chains=4,
    warmup_per_chain=500,
    clinical_validation=False,
)
assert result["absolute_difference"] < 0.015
assert result["risk_split_rhat"] < 1.03
print(json.dumps(result, indent=2))
