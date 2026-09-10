import numpy as np
from scipy.stats import binom

from mdanderson_stats import cid2bp_interval


def test_exact_compiled_fortran_reference_and_separated_sample_formula():
    # Unmodified cfbwb/SETCI/CONINT/MAXTL routines, with the caller's sample swap.
    for (n1, x1, n2, x2), expected in [
        ((12, 7, 7, 1), [-0.046941234639661641, 0.81594820727563988]),
        ((10, 0, 10, 0), [-0.45608455037811801, 0.45608455037812312]),
        ((10, 10, 10, 0), [0.66313280804445918, 1]),
        ((5, 3, 4, 0), [-0.088681179142462846, 0.94725549537909703]),
        ((4, 1, 6, 3), [-0.80587987098138547, 0.41218318686050470]),
    ]:
        result = cid2bp_interval(x1, n1, x2, n2, method="exact")
        np.testing.assert_allclose([result.lower, result.upper], expected, atol=1e-6, rtol=0)
    for confidence in [1e-6, 0.95, 1 - 1e-12]:
        result = cid2bp_interval(100, 100, 0, 100, confidence=confidence, method="exact")
        expected = 2 * np.exp(np.log((1 - confidence) / 2) / 200) - 1
        np.testing.assert_allclose(result.lower, expected, atol=1e-10, rtol=0)
        assert result.upper == 1


def test_exact_endpoints_against_dense_joint_binomial_enumeration():
    n1, x1, n2, x2 = 4, 1, 6, 3
    result = cid2bp_interval(x1, n1, x2, n2, method="exact")
    i, j = np.indices((n1 + 1, n2 + 1)).reshape(2, -1)
    observed = x1 * n2 - x2 * n1
    differences = i * n2 - j * n1
    for delta in [result.lower, result.upper]:
        q = np.linspace(max(0, -delta), min(1, 1 - delta), 10001)
        joint = binom.pmf(i[:, None], n1, q + delta) * binom.pmf(j[:, None], n2, q)
        upper = joint[differences >= observed].sum(axis=0)
        lower = joint[differences <= observed].sum(axis=0)
        np.testing.assert_allclose(np.minimum(upper, lower).max(), 0.025, atol=1e-7, rtol=0)
    reverse = cid2bp_interval(x2, n2, x1, n1, method="exact")
    np.testing.assert_allclose(
        [result.lower, result.upper], [-reverse.upper, -reverse.lower], atol=1e-9
    )


def test_exact_small_sample_coverage_by_complete_outcome_enumeration():
    n1, n2 = 3, 4
    i, j = np.indices((n1 + 1, n2 + 1)).reshape(2, -1)
    intervals = [cid2bp_interval(int(a), n1, int(b), n2, method="exact") for a, b in zip(i, j)]
    lower = np.array([ci.lower for ci in intervals])
    upper = np.array([ci.upper for ci in intervals])
    for p in np.linspace(0, 1, 21):
        for q in np.linspace(0, 1, 21):
            covered = (lower <= p - q + 1e-10) & (upper >= p - q - 1e-10)
            probability = binom.pmf(i, n1, p) * binom.pmf(j, n2, q)
            assert probability[covered].sum() >= 0.95 - 1e-12


def test_exact_refines_maxima_missed_by_native_single_search():
    for n1, x1, n2, x2, side, native in [
        (86, 62, 44, 21, "lower", 0.06603128894859517),
        (89, 49, 39, 12, "upper", 0.4184685969820541),
    ]:
        result = cid2bp_interval(x1, n1, x2, n2, method="exact")
        endpoint = getattr(result, side)
        assert endpoint < native if side == "lower" else endpoint > native
        i, j = np.indices((n1 + 1, n2 + 1))
        difference = i * n2 - j * n1 - (x1 * n2 - x2 * n1)
        # Independent full joint-outcome sums, batched to keep memory bounded.
        masks = [(difference >= 0).astype(float), (difference <= 0).astype(float)]
        best = 0.0
        for u in np.array_split(np.linspace(0, 1, 10001), 20):
            p = max(endpoint, 0) + (1 - abs(endpoint)) * u
            q = max(-endpoint, 0) + (1 - abs(endpoint)) * u
            first = binom.pmf(np.arange(n1 + 1)[:, None], n1, p)
            second = binom.pmf(np.arange(n2 + 1)[:, None], n2, q)
            tails = [(first * (mask @ second)).sum(axis=0) for mask in masks]
            best = max(best, float(np.minimum(*tails).max()))
        np.testing.assert_allclose(best, 0.025, atol=1e-8, rtol=0)
