"""Independent global MTADF posterior quadrature in Cauchy CDF coordinates.

Stream one intercept slice at a time to bound temporary storage. This reference
does not use the package sampler or its posterior implementation.
"""

import csv
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit


def global_posterior_means(order: int) -> np.ndarray:
    n = np.array([10.0, 10.0, 10.0])
    y = np.array([2.0, 7.0, 4.0])
    doses = np.array([-1.0, 0.0, 1.0])
    nodes, weights = leggauss(order)
    # The independent Cauchy priors become uniform probability coordinates.
    alpha = 10 * np.tan(nodes * np.pi / 2)
    beta = 2.5 * np.tan(nodes * np.pi / 2)
    gamma = beta
    linear = beta[:, None, None] * doses[None, None, :]
    quadratic = gamma[None, :, None] * doses[None, None, :] ** 2
    product_weights = weights[:, None] * weights[None, :] / 4
    # A saturated-binomial upper bound keeps likelihood weights in [0, 1].
    shift = np.sum(y * np.log(y / n) + (n - y) * np.log1p(-y / n))
    denominator = 0.0
    numerator = np.zeros(doses.size)
    for intercept, weight in zip(alpha, weights / 2, strict=True):
        eta = intercept + linear + quadratic
        log_likelihood = np.sum(
            y * -np.logaddexp(0, -eta) + (n - y) * -np.logaddexp(0, eta), axis=2
        )
        kernel = np.exp(log_likelihood - shift) * product_weights * weight
        denominator += kernel.sum()
        numerator += (kernel[:, :, None] * expit(eta)).sum(axis=(0, 1))
    return numerator / denominator


if __name__ == "__main__":
    lower = global_posterior_means(128)
    refined = global_posterior_means(192)
    np.testing.assert_allclose(lower, refined, rtol=0, atol=1e-11)
    output = Path(__file__).resolve().parents[1] / "tests/fixtures/mtadf-logistic-quadrature.csv"
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dose", "subjects", "responses", "posterior_mean_efficacy"])
        for dose, response, mean in zip([-1, 0, 1], [2, 7, 4], refined, strict=True):
            writer.writerow([dose, 10, response, repr(float(mean))])
    print(f"Wrote {output.name}; refinement difference={np.max(np.abs(lower - refined)):.3g}")
