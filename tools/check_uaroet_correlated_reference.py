"""Scrambled Sobol importance integration with independent Gaussian rho quadrature."""

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit, ndtr, ndtri
from scipy.stats import qmc

nodes, weights = leggauss(256)


def evaluate(uniform):
    theta = np.empty((len(uniform), 4))
    for i, (mean, sd) in enumerate(((0, 0.7), (0.3, 0.4), (-1, 0.7), (0.3, 0.4))):
        q = uniform[:, i]
        if i in (1, 3):
            lo = ndtr(-mean / sd)
            q = lo + (1 - lo) * q
        theta[:, i] = mean + sd * ndtri(q)
    e0 = expit(-np.stack((theta[:, 0], theta[:, 0] + theta[:, 1]), axis=1))
    t0 = expit(-np.stack((theta[:, 2], theta[:, 2] + theta[:, 3]), axis=1))
    h, k = ndtri(e0), ndtri(t0)
    alpha = np.arcsin(2 * uniform[:, 4] - 1)
    angle = alpha[:, None] * (nodes[None, :] + 1) / 2
    s = np.sin(angle)
    c2 = np.cos(angle) ** 2
    joint = np.empty((len(uniform), 2, 2, 2))
    for d in range(2):
        hd = h[:, d, None]
        kd = k[:, d, None]
        f = np.exp(-(hd * hd - 2 * s * hd * kd + kd * kd) / (2 * c2))
        p00 = e0[:, d] * t0[:, d] + alpha * (f @ weights) / (4 * np.pi)
        joint[:, d, 0, 0] = p00
        joint[:, d, 0, 1] = e0[:, d] - p00
        joint[:, d, 1, 0] = t0[:, d] - p00
        joint[:, d, 1, 1] = 1 - e0[:, d] - t0[:, d] + p00
    assert joint.min() > -1e-12, joint.min()
    likelihood = joint[:, 0, 1, 0] * (1 - t0[:, 1])
    return (likelihood[:, None, None, None] * joint).sum(0), likelihood.sum()


references = []
for seed in range(8):
    sample = qmc.Sobol(5, scramble=True, seed=seed + 819).random_base2(16)
    sums = np.zeros((2, 2, 2))
    total = 0.0
    for block in np.array_split(sample, 16):
        s, t = evaluate(block)
        sums += s
        total += t
    references.append(sums / total)
reference = np.mean(references, axis=0)
error = np.std(references, axis=0, ddof=1) / np.sqrt(8)
python = np.array(
    json.loads(Path("research/uaroet-precision-quality-2026-10-12.json").read_text())[
        "posterior_mean_joint"
    ]
)
result = dict(
    scope=(
        "independent Sobol prior-importance integration and256-node correlation-angle "
        "quadrature; same correlated pending-outcome posterior"
    ),
    scrambles=8,
    points_per_scramble=65536,
    reference_mean_joint=reference.tolist(),
    scramble_standard_error=error.tolist(),
    python_mean_joint=python.tolist(),
    maximum_absolute_error=float(abs(reference - python).max()),
    maximum_reference_standard_error=float(error.max()),
    clinical_validation=False,
)
assert result["maximum_reference_standard_error"] < 0.0001
assert result["maximum_absolute_error"] < 0.008
print(json.dumps(result, indent=2))
