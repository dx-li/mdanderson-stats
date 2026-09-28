import pytest

from mdanderson_stats.stplan_discrete_significance import (
    STPLAN_MAX_SIGNIFICANCE,
    stplan_exact_binomial_significance,
    stplan_exact_poisson_significance,
)


def test_inverse_binomial_selects_minimum_alpha_exact_tail() -> None:
    upper = stplan_exact_binomial_significance(0.2, 0.4, 40, target_power=0.8)
    assert (upper.critical_tail, upper.critical_count) == ("upper", 13)
    assert upper.significance == pytest.approx(0.04324162237632384, abs=1e-15)
    assert upper.achieved_power == pytest.approx(0.87149032192931575, abs=1e-15)
    assert upper.target_attained

    lower = stplan_exact_binomial_significance(0.6, 0.4, 50, target_power=0.8)
    assert (lower.critical_tail, lower.critical_count) == ("lower", 23)
    assert lower.significance == pytest.approx(0.03140555309922028, abs=1e-15)
    assert lower.achieved_power == pytest.approx(0.8438316691000456, abs=1e-15)
    assert lower.target_attained


def test_inverse_poisson_selects_both_directions() -> None:
    upper = stplan_exact_poisson_significance(1, 2, 10, target_power=0.8)
    assert (upper.critical_tail, upper.critical_count) == ("upper", 16)
    assert upper.significance == pytest.approx(0.048740403303978934, abs=1e-15)
    assert upper.achieved_power == pytest.approx(0.8434868653602569, abs=1e-15)

    lower = stplan_exact_poisson_significance(2, 1, 12.5, target_power=0.8)
    assert (lower.critical_tail, lower.critical_count) == ("lower", 15)
    assert lower.significance == pytest.approx(0.022293021307365317, abs=1e-15)
    assert lower.achieved_power == pytest.approx(0.8060290010444158, abs=1e-15)


def test_unattainable_target_returns_best_allowed_region_and_validates_direction() -> None:
    result = stplan_exact_binomial_significance(0.2, 0.4, 2, target_power=0.8)
    assert (result.critical_tail, result.critical_count) == ("upper", 1)
    assert result.significance == pytest.approx(0.36)
    assert result.achieved_power == pytest.approx(0.64)
    assert result.significance <= STPLAN_MAX_SIGNIFICANCE
    assert not result.target_attained

    with pytest.raises(ValueError, match="must differ"):
        stplan_exact_poisson_significance(1, 1, 10, target_power=0.8)


def test_poisson_brackets_tiny_target_by_survival_probability() -> None:
    target = 1e-100
    result = stplan_exact_poisson_significance(1.0, 2.0, 10.0, target_power=target)
    assert result.critical_tail == "upper"
    assert result.target_attained
    assert result.achieved_power >= target
    from scipy.stats import poisson

    assert poisson.sf(result.critical_count, 20.0) < target


def test_poisson_rejects_positive_rate_exposure_underflow() -> None:
    with pytest.raises(ArithmeticError, match="underflowed to zero"):
        stplan_exact_poisson_significance(1e-300, 2e-300, 1e-100, target_power=0.8)
