import numpy as np
from scipy.stats import gamma

from mdanderson_stats import confint_survival_fixed_events, confint_survival_probability


def test_confint_survival_independent_r_poisson_mixture_and_error_bound():
    for target, complete in [("hazard", 0.15495114420212214), ("mean", 0.12857484375969869)]:
        result = confint_survival_probability(1, 5, 10, 0, 0.5, target=target)
        np.testing.assert_allclose(result.expected_events, 45.00022699964881)
        assert result.probability <= complete + 2e-14
        assert complete <= result.probability + result.omitted_probability + 2e-14
        assert result.omitted_probability <= 1e-12
        tighter = confint_survival_probability(
            1, 5, 10, 0, 0.5, target=target, tail_tolerance=1e-14
        )
        assert abs(tighter.probability - complete) < 2e-14


def test_confint_fixed_event_widths_from_exponential_samples():
    rng = np.random.default_rng(6407)
    n = 45
    total = rng.exponential(size=(60000, n)).sum(axis=1)
    lo, hi = gamma.ppf(0.025, n), gamma.isf(0.025, n)
    lengths = {"hazard": (hi - lo) / total, "mean": total * (1 / lo - 1 / hi)}
    for target, width in lengths.items():
        expected = float(confint_survival_fixed_events(n, 1, 0.65, target=target))
        assert abs(np.mean(width <= 0.65) - expected) < 0.007
        assert confint_survival_fixed_events(0, 1, 0.65, target=target) == 0
    reciprocal_length = (hi - lo) ** 2 / (lo * hi) / 0.65
    total_prob = float(confint_survival_fixed_events(n, 1, 0.65)) + float(
        confint_survival_fixed_events(n, 1, reciprocal_length, target="mean")
    )
    np.testing.assert_allclose(total_prob, 1, atol=1e-13)


def test_confint_survival_zero_accrual_units_and_tiny_event_probability():
    assert confint_survival_probability(1, 0, 10, 0, 0.5).probability == 0
    for target in ["hazard", "mean"]:
        ordinary = confint_survival_probability(1, 5, 10, 2, 0.5, target=target)
        for scale in [1e-200, 1e200]:
            result = confint_survival_probability(
                1 / scale,
                5 / scale,
                10 * scale,
                2 * scale,
                0.5 / scale if target == "hazard" else 0.5 * scale,
                target=target,
            )
            np.testing.assert_allclose(result.probability, ordinary.probability, rtol=2e-11)
            np.testing.assert_allclose(result.expected_events, ordinary.expected_events, rtol=1e-12)
    rare = confint_survival_probability(1e-300, 1e300, 1e-100, 0, 0.5)
    np.testing.assert_allclose(rare.expected_events, 5e-201, rtol=1e-12, atol=0)
    np.testing.assert_allclose(rare.probability, rare.expected_events, rtol=1e-12, atol=0)
    assert (
        rare.event_probability == 0
    )  # Unrepresentable alone; expected count remains representable.
