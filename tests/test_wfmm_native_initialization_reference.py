"""Independent Gaussian calculation of original initialization traces.

These certify recovered source semantics, not an integrated WFMM initializer.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

CASES = json.loads(
    (Path(__file__).parent / "fixtures/wfmm-native-initialization.json").read_text()
)["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case"])
def test_original_initialization_conditions_on_shrunken_beta(case):
    y, x = np.array(case["y"]), np.array(case["model"]["X"])
    c = np.array(case["model"]["C"])
    components = []
    offset = 0
    if "Z" in case["model"]:
        z = np.array(case["model"]["Z"])
        for count in case["model"]["m"]:
            block = z[:, offset : offset + int(count)]
            components.append(block @ block.T)
            offset += int(count)
    components.extend(np.diag(c[:, j]) for j in range(c.shape[1]))
    fixed_beta = np.array(case["traces"][0]["beta"])
    for row in case["traces"]:
        j = row["coefficient"]
        variance = np.array(row["theta"])[:, j]
        covariance = sum(v * matrix for v, matrix in zip(variance, components, strict=True))
        np.testing.assert_array_equal(row["beta"], fixed_beta)
        residual = y[:, j] - x @ fixed_beta[:, j]
        expected = -0.5 * (
            np.linalg.slogdet(covariance)[1] + residual @ np.linalg.solve(covariance, residual)
        )
        assert row["loglik"] == pytest.approx(expected, abs=3e-12, rel=0)
    tau = np.array(case["output"]["tau_MOM"])
    alpha = np.array(case["output"]["alpha_MOM"])
    beta_ns = np.array(case["traces"][0]["betans"])
    np.testing.assert_allclose(fixed_beta, beta_ns * alpha * tau / (1 + tau), rtol=2e-14)


def test_native_scale_is_likelihood_drop_half_width_not_hessian_variance():
    case = CASES[0]
    y = np.array(case["y"])
    beta = np.array(case["traces"][0]["beta"])[0]
    n = len(y)
    for j in range(y.shape[1]):
        native = case["output"]["omega_MLE"][0][j]
        rss = np.sum((y[:, j] - beta[j]) ** 2)

        def root(v):
            return 0.5 * (n * np.log(v / native) + rss * (1 / v - 1 / native)) - 1

        lower = brentq(root, 1e-6, native)
        upper = brentq(root, native, 100 * native)
        # Native root/optimizer tolerance, not an algebraic equality claim.
        assert case["output"]["se_omega"][0][j] == pytest.approx((upper - lower) / 2, abs=3e-5)
        assert abs(native - rss / n) < 2e-4
        # Unshrunk ordinary profile ML has a different objective/optimum.
        profile = np.mean((y[:, j] - y[:, j].mean()) ** 2)
        assert native > profile + 0.009
