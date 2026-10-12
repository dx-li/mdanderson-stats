"""Independent importance integration of a dense correlated pseudo-posterior."""

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit, ndtr, ndtri
from scipy.stats import qmc

counts = np.array([[[62, 6], [27, 5]], [[40, 18], [25, 17]]])
centers = np.array([-0.8, 0.5, -2.1, 1.5])
scale = 0.5
nodes, weights = leggauss(256)
ceiling = np.sum(counts * np.log(counts / 100))


def evaluate(u):
    theta = np.empty((len(u), 4))
    for i, mu in enumerate(centers):
        q = u[:, i]
        if i in (1, 3):
            lo = ndtr(-mu / scale)
            q = lo + (1 - lo) * q
        theta[:, i] = mu + scale * ndtri(q)
    e0 = expit(-np.stack((theta[:, 0], theta[:, 0] + theta[:, 1]), axis=1))
    t0 = expit(-np.stack((theta[:, 2], theta[:, 2] + theta[:, 3]), axis=1))
    h, k = ndtri(e0), ndtri(t0)
    alpha = np.arcsin(2 * u[:, 4] - 1)
    angle = alpha[:, None] * (nodes[None, :] + 1) / 2
    s = np.sin(angle)
    c2 = np.cos(angle) ** 2
    j = np.empty((len(u), 2, 2, 2))
    for d in range(2):
        hd = h[:, d, None]
        kd = k[:, d, None]
        f = np.exp(-(hd * hd - 2 * s * hd * kd + kd * kd) / (2 * c2))
        p = e0[:, d] * t0[:, d] + alpha * (f @ weights) / (4 * np.pi)
        j[:, d, 0, 0] = p
        j[:, d, 0, 1] = e0[:, d] - p
        j[:, d, 1, 0] = t0[:, d] - p
        j[:, d, 1, 1] = 1 - e0[:, d] - t0[:, d] + p
    assert j.min() > -1e-12, j.min()
    # Every observed cell has positive count. Vanishing quadrature cells have
    # negligible likelihood; record how often this occurs for transparency.
    vanishing = np.any(j <= 0, axis=(1, 2, 3))
    valid = ~vanishing
    loglik = np.sum(counts * np.log(j[valid]), axis=(1, 2, 3))
    logw = (
        loglik
        - ceiling
        - np.sum(theta[valid] ** 2, axis=1) / 20000
        + np.sum((theta[valid] - centers) ** 2, axis=1) / (2 * scale**2)
    )
    w = np.exp(logw)
    bad = theta[vanishing]
    upper_loglik = np.sum(counts * np.log(np.maximum(j[vanishing], 2e-12)), axis=(1, 2, 3))
    upper_w = np.exp(
        upper_loglik
        - ceiling
        - np.sum(bad**2, axis=1) / 20000
        + np.sum((bad - centers) ** 2, axis=1) / (2 * scale**2)
    )
    return (w[:, None] * theta[valid]).sum(0), w.sum(), int(vanishing.sum()), float(upper_w.sum())


def integrate(order):
    global nodes, weights
    nodes, weights = leggauss(order)
    results = []
    vanishing = 0
    omitted = 0.0
    retained = 0.0
    for seed in range(8):
        sample = qmc.Sobol(5, scramble=True, seed=711 + seed).random_base2(16)
        sums = np.zeros(4)
        total = 0
        for block in np.array_split(sample, 16):
            s, t, v, b = evaluate(block)
            sums += s
            total += t
            vanishing += v
            omitted += b
            retained += t
        results.append(sums / total)
    return (
        np.mean(results, axis=0),
        np.std(results, axis=0, ddof=1) / np.sqrt(8),
        vanishing,
        omitted / retained,
    )


means, se, vanishing, omitted_bound = integrate(256)
second, _, _, _ = integrate(512)
node_difference = float(np.max(np.abs(means - second)))
python = np.array(
    json.loads(Path("research/uaroet-calibration-quality-2026-10-12.json").read_text())[
        "prior_mean"
    ]
)
r = dict(
    scope=(
        "dense correlated pseudo-posterior; independent importance integration, "
        "positive-normal proposal and256-node correlation-angle quadrature"
    ),
    counts=counts.tolist(),
    reference_parameter_means=means.tolist(),
    reference_standard_error=se.tolist(),
    python_parameter_means=python.tolist(),
    absolute_errors=abs(means - python).tolist(),
    vanishing_likelihood_points=vanishing,
    total_points=8 * 65536,
    roundoff_floor_relative_likelihood_bound=omitted_bound,
    roundoff_floor=2e-12,
    clinical_validation=False,
    quadrature_node_doubling_parameter_mean_difference=node_difference,
    roundoff_bound_is_conditional_on_declared_numerical_floor=True,
)
assert node_difference < 1e-10
assert max(se) < 0.001
assert max(abs(means - python)) < 0.015
print(json.dumps(r, indent=2))
