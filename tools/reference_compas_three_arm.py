"""Independent marginal logit-prior, Laplace-centered tensor Gaussian quadrature."""

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln, logsumexp, roots_hermitenorm


def evidence(n, y, v, order):
    n = np.asarray(n, float)
    y = np.asarray(y, float)
    k = len(n)

    def prior(t):
        mean = np.mean(t, axis=-1, keepdims=True)
        return np.sum((t - mean) ** 2, axis=-1) / v + k * mean[..., 0] ** 2 / (v + k * 1e6)

    def objective(t):
        return float(
            np.sum(y * np.logaddexp(0.0, -t) + (n - y) * np.logaddexp(0.0, t)) + 0.5 * prior(t)
        )

    def gradient(t):
        mean = t.mean()
        return n * expit(t) - y + (t - mean) / v + mean / (v + k * 1e6)

    def hessian(t):
        p = expit(t)
        return (
            np.diag(n * p * (1 - p))
            + (np.eye(k) - np.ones((k, k)) / k) / v
            + np.ones((k, k)) / (k * (v + k * 1e6))
        )

    fit = minimize(
        objective,
        np.zeros(k),
        jac=gradient,
        hess=hessian,
        method="trust-exact",
        options={"gtol": 1e-10},
    )
    mode = fit.x
    H = hessian(mode)
    cov = np.linalg.inv(H)
    C = np.linalg.cholesky(cov)
    z, w = roots_hermitenorm(order)
    w = w / math.sqrt(2 * math.pi)
    meshes = np.meshgrid(*([z] * k), indexing="ij")
    points = np.stack([m.ravel() for m in meshes], axis=-1)
    weights = np.meshgrid(*([np.log(w)] * k), indexing="ij")
    logweights = sum(m.ravel() for m in weights)
    theta = mode + points @ C.T
    lp = -np.sum(
        y * np.logaddexp(0.0, -theta) + (n - y) * np.logaddexp(0.0, theta), axis=-1
    ) - 0.5 * prior(theta)
    const = (
        float(np.sum(gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)))
        - 0.5 * ((k - 1) * math.log(v) + math.log(v + k * 1e6))
        + 0.5 * np.linalg.slogdet(cov)[1]
    )
    return float(logsumexp(logweights + lp + 0.5 * np.sum(points**2, axis=-1)) + const)


if __name__ == "__main__":
    rows = []
    for v in [0.00012, 0.042, 0.126, 0.379, 126.1]:
        a = evidence([11] * 3, [4, 5, 8], v, 48)
        b = evidence([11] * 3, [4, 5, 8], v, 80)
        row = dict(
            n=[11] * 3, y=[4, 5, 8], variance=v, log_evidence=b, refinement_difference=abs(a - b)
        )
        rows.append(row)
        print(row, flush=True)
    Path("tests/fixtures/compas-three-arm-reference.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
