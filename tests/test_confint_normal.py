import numpy as np
from scipy.special import erf
from scipy.stats import chi2, t

from mdanderson_stats import (
    confint_normal_probability,
    confint_normal_sample_size,
    confint_normal_sd_limit,
)


def test_confint_manual_examples_and_independent_r_values():
    np.testing.assert_allclose(
        confint_normal_probability(45, 1, 2, confidence=0.9), 0.52129784259065892
    )
    np.testing.assert_allclose(
        confint_normal_probability(20, 0.5, 1, confidence=0.9, target="sd"), 0.2336381350798496
    )
    np.testing.assert_allclose(
        confint_normal_probability(20, 1, 1, target="mean_difference", sample_size2=30),
        0.093011102872683868,
    )
    assert confint_normal_sample_size(1, 2, confidence=0.9) == 57
    assert confint_normal_sample_size(0.5, 1, confidence=0.9, assurance=0.8, target="sd") == 30
    np.testing.assert_allclose(
        confint_normal_sd_limit(45, 1, assurance=[0.05, 0.5, 0.95], confidence=0.9),
        [2.4261, 2.0115, 1.7026],
        atol=5e-5,
    )
    np.testing.assert_allclose(
        confint_normal_sd_limit(20, 0.5, assurance=[0.05, 0.5, 0.95], confidence=0.9, target="sd"),
        [1.1886, 0.88285, 0.68859],
        atol=5e-5,
    )


def test_confint_widths_from_independent_normal_samples():
    rng = np.random.default_rng(64)
    first = rng.normal(size=(60000, 20))
    second = rng.normal(size=(60000, 30))
    s1, s2 = first.std(axis=1, ddof=1), second.std(axis=1, ddof=1)
    widths = {
        "mean": 2 * t.isf(0.025, 19) * s1 / np.sqrt(20),
        "sd": np.sqrt(19)
        * s1
        * (1 / np.sqrt(chi2.ppf(0.025, 19)) - 1 / np.sqrt(chi2.isf(0.025, 19))),
        "mean_difference": 2
        * t.isf(0.025, 48)
        * np.sqrt((19 * s1**2 + 29 * s2**2) / 48 * (1 / 20 + 1 / 30)),
    }
    for target, width in widths.items():
        length = 1 if target != "sd" else 0.7
        expected = confint_normal_probability(
            20, length, 1, target=target, sample_size2=30 if target == "mean_difference" else None
        )
        assert abs(np.mean(width <= length) - expected) < 0.007


def test_confint_inversion_minimality_scaling_and_extreme_tail():
    for target in ["mean", "sd", "mean_difference"]:
        n = confint_normal_sample_size(1, 2, target=target)
        for size, enough in [(n, True), (n - 1, False)]:
            kwargs = {"sample_size2": size} if target == "mean_difference" else {}
            probability = confint_normal_probability(size, 1, 2, target=target, **kwargs)
            assert (float(probability) >= 0.9) == enough
        kwargs = {"sample_size2": n} if target == "mean_difference" else {}
        limit = confint_normal_sd_limit(n, 1, target=target, **kwargs)
        np.testing.assert_allclose(
            confint_normal_probability(n, 1, limit, target=target, **kwargs), 0.9
        )
        for scale in [1e-200, 1e200]:
            np.testing.assert_allclose(
                confint_normal_probability(n, scale, 2 * scale, target=target, **kwargs),
                confint_normal_probability(n, 1, 2, target=target, **kwargs),
                rtol=1e-11,
            )
    expected = erf(1e-200 / (2 * np.tan(0.9 * np.pi / 2)))
    np.testing.assert_allclose(
        confint_normal_probability(2, 1e-200, 1, confidence=0.9), expected, rtol=1e-12, atol=0
    )
