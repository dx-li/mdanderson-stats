import numpy as np
from scipy.special import ndtri

from mdanderson_stats.cid2bp import cid2bp_interval


def test_native_archive_reference_intervals():
    # CID2BP_V1.2 TestOut_Orig.txt, x1/n1=7/12, x2/n2=1/7, 95%.
    for method, expected in [
        ("wald", [0.0597, 0.8213]),
        ("continuity_corrected", [-0.0350, 0.9160]),
        ("yates", [-0.0534, 0.9344]),
        ("cox_snell", [-0.0013, 0.7523]),
    ]:
        ci = cid2bp_interval(7, 12, 1, 7, method=method)
        # Cox-Snell native root tolerance is 1e-4; other outputs round to 4 digits.
        np.testing.assert_allclose([ci.lower, ci.upper], expected, atol=1e-4)


def test_profile_exact_likelihood_limits_and_sample_exchange():
    z = -ndtri(0.025)
    zeros = cid2bp_interval(0, 10, 0, 10)
    bound = -np.expm1(-z * z / 20)
    np.testing.assert_allclose([zeros.lower, zeros.upper], [-bound, bound], atol=1e-12)
    separated = cid2bp_interval(10, 10, 0, 10)
    np.testing.assert_allclose(separated.lower, 2 * np.exp(-z * z / 40) - 1, atol=1e-12)
    assert separated.upper == 1
    for x1, n1, x2, n2 in [
        (7, 12, 1, 7),
        (0, 25, 9, 10),
        (17, 20, 5, 8),
        (500000, 1000000, 400000, 1000000),
    ]:
        forward = cid2bp_interval(x1, n1, x2, n2)
        reverse = cid2bp_interval(x2, n2, x1, n1)
        np.testing.assert_allclose(
            [forward.lower, forward.upper], [-reverse.upper, -reverse.lower], atol=2e-10
        )


def test_source_special_adjustments_for_normal_methods():
    for method in ["wald", "continuity_corrected", "yates"]:
        equal = cid2bp_interval(10, 10, 0, 10, method=method)
        np.testing.assert_allclose(equal.lower, 2 * 0.025 ** (1 / 20) - 1)
        assert equal.upper == 1
        unequal = cid2bp_interval(100, 100, 0, 2, method=method)
        np.testing.assert_allclose(unequal.lower, 0.025**0.5)
        near = cid2bp_interval(9, 10, 0, 10, method=method)
        np.testing.assert_allclose(near.upper, 0.975**0.1)


def test_profile_small_likelihood_loss_does_not_cancel():
    result = cid2bp_interval(500000, 1000000, 400000, 1000000, confidence=1e-6)
    expected_width = -ndtri((1 - 1e-6) / 2) * np.sqrt(0.49 / 1000000)
    np.testing.assert_allclose(
        [result.estimate - result.lower, result.upper - result.estimate], expected_width, rtol=0.002
    )
