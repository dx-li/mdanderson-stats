from __future__ import annotations

from fractions import Fraction
from math import comb, factorial

import numpy as np
import pytest

from mdanderson_stats.success_calibration import BinarySuccessTable
from mdanderson_stats.success_calibration_two_arm_search import (
    _search_table,
    calibrate_binary_two_arm_success_cutoff,
)


def _beta(a: int, b: int) -> Fraction:
    return Fraction(factorial(a - 1) * factorial(b - 1), factorial(a + b - 1))


def _beta_order(a: int, b: int, c: int, d: int) -> Fraction:
    """Exact integral of Beta(a,b)'s density times Beta(c,d)'s CDF."""
    degree = c + d - 1
    total = Fraction(0)
    for power in range(c, degree + 1):
        total += comb(degree, power) * _beta(a + power, b + degree - power)
    return total / _beta(a, b)


def _uniform_two_by_two_table() -> tuple[
    BinarySuccessTable, list[tuple[Fraction, Fraction]], list[Fraction]
]:
    posterior: list[Fraction] = []
    predictive: list[Fraction] = []
    null_mass: list[Fraction] = []
    for treatment_count in range(3):
        treatment_mass = Fraction(1, 3)
        for control_count in range(3):
            posterior.append(
                _beta_order(
                    1 + treatment_count,
                    1 + 2 - treatment_count,
                    1 + control_count,
                    1 + 2 - control_count,
                )
            )
            predictive.append(treatment_mass * Fraction(1, 3))
            null_mass.append(Fraction(comb(2, treatment_count) * comb(2, control_count), 16))

    pa = np.array([float(value) for value in posterior])
    effective = np.array([float(m * q) for m, q in zip(predictive, posterior, strict=True)])
    ineffective = np.array([float(m * (1 - q)) for m, q in zip(predictive, posterior, strict=True)])
    null = np.array([float(value) for value in null_mass])
    table = BinarySuccessTable(
        pa.reshape(3, 3),
        np.zeros((3, 3)),
        effective.reshape(3, 3),
        ineffective.reshape(3, 3),
        null.reshape(3, 3),
    )
    return table, list(zip(posterior, predictive, strict=True)), null_mass


def test_integer_beta_states_match_independent_fraction_oracle() -> None:
    table, states, null_mass = _uniform_two_by_two_table()
    lower = Fraction(1, 4)
    target = Fraction(1, 5)
    candidate_values = sorted(
        {lower} | {posterior for posterior, _ in states if lower < posterior <= Fraction(99, 100)}
    )
    expected: tuple[Fraction, Fraction, Fraction] | None = None
    for cutoff in candidate_values:
        selected = [(posterior, mass) for posterior, mass in states if posterior > cutoff]
        tp = sum((mass * posterior for posterior, mass in selected), Fraction(0))
        fp = sum((mass * (1 - posterior) for posterior, mass in selected), Fraction(0))
        if tp + fp > 0 and fp / (tp + fp) <= target:
            expected = cutoff, tp, fp
            break
    assert expected is not None

    result = _search_table(table, float(target), (float(lower), 0.99))
    assert result.cutoff == float(expected[0])
    assert result.operating_characteristics.true_positive == pytest.approx(float(expected[1]))
    assert result.operating_characteristics.false_positive == pytest.approx(float(expected[2]))
    assert result.operating_characteristics.incorrect_decision_probability == pytest.approx(
        float(expected[2] / (expected[1] + expected[2])), rel=0, abs=5e-16
    )
    expected_alpha = sum(
        (
            null_probability
            for (posterior, _), null_probability in zip(states, null_mass, strict=True)
            if posterior > expected[0]
        ),
        Fraction(0),
    )
    assert result.operating_characteristics.frequentist_type1_error == pytest.approx(
        float(expected_alpha), rel=0, abs=5e-16
    )


def test_search_scans_nonmonotone_pid_and_respects_strict_ties() -> None:
    table = BinarySuccessTable(
        np.array([0.2, 0.5, 0.8]),
        np.zeros(3),
        np.array([0.0, 0.4, 0.0]),
        np.array([0.4, 0.0, 0.2]),
        np.array([0.2, 0.3, 0.5]),
    )
    at_lower = table.evaluate(0.1)
    at_middle = table.evaluate(0.2)
    at_upper = table.evaluate(0.5)
    assert at_lower.incorrect_decision_probability == pytest.approx(0.6)
    assert at_middle.incorrect_decision_probability == pytest.approx(1 / 3)
    assert at_upper.incorrect_decision_probability == pytest.approx(1.0)

    result = _search_table(table, 0.55, (0.1, 0.8))
    assert result.cutoff == 0.2
    assert result.operating_characteristics == at_middle


def test_error_gaps_zero_endpoint_and_unresolved_range() -> None:
    table = BinarySuccessTable(
        np.array([0.2, 0.8]),
        np.array([0.01, 0.02]),
        np.array([0.0, 0.9]),
        np.array([0.1, 0.0]),
        np.array([0.4, 0.6]),
    )
    result = _search_table(table, 0.1, (0.2, 0.9))
    conservative_upper = np.nextafter(
        float(table.posterior_probability[0] + np.nextafter(table.posterior_error[0], np.inf)),
        np.inf,
    )
    assert result.cutoff == np.nextafter(conservative_upper, np.inf)
    assert not np.any(
        (table.posterior_error > 0)
        & (np.abs(table.posterior_probability - result.cutoff) <= table.posterior_error)
    )
    np.testing.assert_array_equal(table.posterior_probability > result.cutoff, [False, True])
    assert table.evaluate(result.cutoff) == result.operating_characteristics

    zero_endpoint = _search_table(table, 0.2, (0.0, 0.9))
    assert zero_endpoint.cutoff == 0.0
    assert zero_endpoint.operating_characteristics == table.evaluate(0.0)

    unresolved = BinarySuccessTable(
        np.array([0.5]),
        np.array([0.5]),
        np.array([0.5]),
        np.array([0.5]),
        np.array([1.0]),
    )
    with pytest.raises(ArithmeticError, match="unresolved"):
        _search_table(unresolved, 0.1, (0.2, 0.8))

    with pytest.raises(ValueError, match="no cutoff"):
        _search_table(unresolved, 0.1, (0.0, 0.0))
    with pytest.raises(ValueError, match="no cutoff"):
        _search_table(unresolved, 0.1, (1.0, 1.0))

    public = calibrate_binary_two_arm_success_cutoff(
        1, 1, 0.3, cutoff_range=(0.49, 0.99), margin=0.0
    )
    assert public.cutoff == 0.5
    with pytest.raises(ValueError, match="40000"):
        calibrate_binary_two_arm_success_cutoff(200, 200, 0.1)
