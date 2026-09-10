from math import comb

import numpy as np
from scipy.integrate import quad

from mdanderson_stats import cid2bp_interval


def test_supplied_native_weighted_reference_and_automatic_choice():
    for method, expected in [
        ("weighted_mid_p", [-0.0069, 0.7551]),
        ("weighted_likelihood", [-0.0247, 0.7645]),
    ]:
        ci = cid2bp_interval(7, 12, 1, 7, method=method)
        np.testing.assert_allclose([ci.lower, ci.upper], expected, atol=1e-4)
    assert cid2bp_interval(7, 12, 1, 7, method="auto") == cid2bp_interval(
        7, 12, 1, 7, method="weighted_likelihood"
    )
    assert cid2bp_interval(10, 25, 9, 25, method="auto").method == "weighted_likelihood"
    assert cid2bp_interval(10, 25, 9, 26, method="auto").method == "cox_snell"


def test_weighted_endpoints_against_independent_adaptive_integration():
    n1, x1, n2, x2 = 4, 1, 6, 3
    observed = x1 * n2 - x2 * n1
    for mid in [False, True]:
        ci = cid2bp_interval(
            x1, n1, x2, n2, method="weighted_mid_p" if mid else "weighted_likelihood"
        )
        for delta, upper_tail in [(ci.lower, True), (ci.upper, False)]:

            def likelihood(p):
                q = p - delta
                return p**x1 * (1 - p) ** (n1 - x1) * q**x2 * (1 - q) ** (n2 - x2)

            def numerator(p):
                q = p - delta
                tail = 0.0
                for i in range(n1 + 1):
                    for j in range(n2 + 1):
                        difference = i * n2 - j * n1 - observed
                        weight = (
                            0.5
                            if mid and difference == 0
                            else float(difference >= 0 if upper_tail else difference <= 0)
                        )
                        tail += (
                            weight
                            * comb(n1, i)
                            * p**i
                            * (1 - p) ** (n1 - i)
                            * comb(n2, j)
                            * q**j
                            * (1 - q) ** (n2 - j)
                        )
                return tail * likelihood(p)

            lo, hi = max(0, delta), min(1, 1 + delta)
            ratio = (
                quad(numerator, lo, hi, epsabs=1e-14)[0] / quad(likelihood, lo, hi, epsabs=1e-14)[0]
            )
            np.testing.assert_allclose(ratio, 0.025, atol=2e-10)


def test_weighted_symmetry_and_boundary_data():
    for method in ["weighted_mid_p", "weighted_likelihood"]:
        for x1, n1, x2, n2 in [(0, 8, 0, 6), (8, 8, 0, 6), (2, 6, 3, 9), (100, 100, 0, 100)]:
            forward = cid2bp_interval(x1, n1, x2, n2, method=method)
            reverse = cid2bp_interval(x2, n2, x1, n1, method=method)
            np.testing.assert_allclose(
                [forward.lower, forward.upper], [-reverse.upper, -reverse.lower], atol=2e-10
            )
            assert -1 <= forward.lower <= forward.upper <= 1
