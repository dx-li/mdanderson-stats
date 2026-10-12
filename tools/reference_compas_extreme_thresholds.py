"""Fixed-node independent evidence checks for identical-arm ComPAS stress cases.

Uses fixed Gauss-Legendre integration in mu and either theta or a standard normal
coordinate; no adaptive integration, mode finder, response threshold, or package
implementation is used. Expanded limits and increased orders are cross-checked.
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy.special import gammaln, roots_hermitenorm, roots_legendre


def evidence(y, variance, order, limit):
    z, w = roots_legendre(order)
    mu = limit * z
    logcoef = gammaln(101) - gammaln(y + 1) - gammaln(101 - y)
    if variance == 1000.0:
        # Broad conditional prior, narrow likelihood: integrate directly in theta.
        width = 60 * limit / 256
        theta = width * z - 36  # [-96,24], expanded to [-111,39] in final check.
        wt = width * w
        loglik = logcoef - y * np.logaddexp(0.0, -theta) - (100 - y) * np.logaddexp(0.0, theta)
        integrand = np.exp(
            loglik[None, :] - (theta[None, :] - mu[:, None]) ** 2 / (2 * variance)
        ) / math.sqrt(2 * math.pi * variance)
        inner = integrand @ wt
    else:
        normal, nw = roots_hermitenorm(64 if order == 1024 else 32)
        theta = mu[:, None] + math.sqrt(variance) * normal[None, :]
        loglik = logcoef - y * np.logaddexp(0.0, -theta) - (100 - y) * np.logaddexp(0.0, theta)
        inner = np.exp(loglik) @ (nw / math.sqrt(2 * math.pi))
    total = np.dot(w, inner**6 * np.exp(-(mu**2) / (2e6))) * limit / math.sqrt(2 * math.pi * 1e6)
    return math.log(total)


rows = []
for y, variance, limit in [(1, 1000.0, 256.0), (50, 1e-6, 4.0)]:
    values = [
        dict(order=order, mu_limit=limit, log_evidence=evidence(y, variance, order, limit))
        for order in (256, 512, 768)
    ]
    values.append(
        dict(
            order=1024,
            mu_limit=limit * 1.25,
            log_evidence=evidence(y, variance, 1024, limit * 1.25),
        )
    )
    rows.append(dict(completed=[100] * 6, successes=[y] * 6, variance=variance, refinements=values))
output = dict(
    method=(
        "fixed Gauss-Legendre/normal integration of normalized hierarchical likelihood; "
        "no response threshold or production imports"
    ),
    cases=rows,
)
Path("tests/fixtures/compas-extreme-threshold-reference.json").write_text(
    json.dumps(output, indent=2) + "\n"
)
print(json.dumps(output, indent=2))
