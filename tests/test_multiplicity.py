import itertools
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.multiplicity import (
    METHODS,
    multiple_testing,
    rom_critical_values,
    sharpened_testing,
)

REFERENCE = json.loads((Path(__file__).parent / "fixtures/multi.json").read_text())


@pytest.mark.parametrize("family", REFERENCE["adjustments"])
def test_original_adjustment_routines(family):
    for method, expected in family["sorted_adjusted"].items():
        result = multiple_testing(family["pvalues"], method)
        np.testing.assert_allclose(
            result.adjusted_pvalues[result.order], expected, atol=2e-14, rtol=2e-12
        )
        assert result.critical_values is None
        np.testing.assert_array_equal(result.reject, result.adjusted_pvalues <= 0.05)


@pytest.mark.parametrize("case", REFERENCE["rom"])
def test_original_rom_thresholds(case):
    np.testing.assert_allclose(
        rom_critical_values(case["n"], case["alpha"]), case["thresholds"], rtol=5e-10, atol=2e-14
    )


@pytest.mark.parametrize("case", REFERENCE["sharpened"])
def test_original_sharpened_decisions(case):
    for method in ("holm", "hochberg"):
        result = sharpened_testing(
            case["pvalues"], case["null_estimate"], method, alpha=case["alpha"]
        )
        expected = np.zeros(len(result), dtype=bool)
        expected[np.argsort(case["pvalues"], kind="stable")[: case[method]]] = True
        np.testing.assert_array_equal(result, expected)


def test_holm_against_exhaustive_closed_bonferroni_testing():
    p = np.array([0.03, 0.001, 0.2, 0.014, 0.011])
    expected = np.zeros(p.size)
    for size in range(1, p.size + 1):
        for subset in itertools.combinations(range(p.size), size):
            intersection_p = min(1, size * min(p[list(subset)]))
            expected[list(subset)] = np.maximum(expected[list(subset)], intersection_p)
    np.testing.assert_allclose(multiple_testing(p, "holm").adjusted_pvalues, expected)


def test_hand_calculated_step_up_and_historical_harmonic_formula():
    p = [0.04, 0.01, 0.02]
    bh = multiple_testing(p, "simes")
    np.testing.assert_allclose(bh.adjusted_pvalues, [0.04, 0.03, 0.03])
    harmonic = 1 + 1 / 2 + 1 / 3
    np.testing.assert_allclose(
        multiple_testing(p, "multi-hommel").adjusted_pvalues,
        np.array([0.04, 0.03, 0.03]) * harmonic,
    )
    with pytest.raises(ValueError, match="Unknown"):
        multiple_testing(p, "hommel")


def test_rom_small_case_from_exact_polynomial():
    a = 0.05
    np.testing.assert_allclose(
        rom_critical_values(3, a), [(a + a * a / 4) / 3, a / 2, a], rtol=1e-14
    )
    np.testing.assert_array_equal(rom_critical_values(3, 0), 0)


def test_rom_decision_uses_step_up_rejection_not_pointwise_thresholds():
    result = multiple_testing([0.02, 0.021, 0.8], "rom")
    assert result.adjusted_pvalues is None
    assert 0.02 > result.critical_values[0]
    np.testing.assert_array_equal(result.reject, [True, True, False])


@pytest.mark.parametrize("method", ["sidak", "holm-sidak", "finner"])
def test_tiny_pvalues_preserved_against_high_precision_decimal(method):
    p = [1e-30, 0.1, 1.0]
    with localcontext() as context:
        context.prec = 90
        expected = float(1 - (1 - Decimal("1e-30")) ** 3)
    actual = multiple_testing(p, method).adjusted_pvalues
    assert actual[0] == pytest.approx(expected, rel=2e-15, abs=0)
    assert actual[-1] == 1


@pytest.mark.parametrize("method", METHODS)
def test_batch_shape_input_order_and_permutation_invariance(method):
    values = np.array([[[0.1, 0.001, 0.03, 0.1], [1, 0, 0.5, 0.02]]])
    result = multiple_testing(values, method)
    assert result.reject.shape == values.shape
    for i in range(2):
        single = multiple_testing(values[0, i], method)
        np.testing.assert_array_equal(single.reject, result.reject[0, i])
    permutation = [2, 0, 3, 1]
    permuted = multiple_testing(values[..., permutation], method)
    np.testing.assert_array_equal(permuted.reject, result.reject[..., permutation])
    field = "critical_values" if method == "rom" else "adjusted_pvalues"
    if method != "rom":  # Equal p-values can receive different rank cutoffs for Rom.
        np.testing.assert_allclose(
            getattr(permuted, field), getattr(result, field)[..., permutation]
        )
    np.testing.assert_array_equal(values, [[[0.1, 0.001, 0.03, 0.1], [1, 0, 0.5, 0.02]]])


@pytest.mark.parametrize("method", ["bonferroni", "sidak", "holm", "holm-sidak", "hochberg", "rom"])
def test_independent_global_null_error_rate(method):
    values = np.random.default_rng(85021).uniform(size=(20000, 12))
    rejections = multiple_testing(values, method).reject
    # A fixed simulation checks calibration, separately from algebra/reference parity.
    empirical = np.mean(np.any(rejections, axis=1))
    assert empirical < 0.05 + 6 * math.sqrt(0.05 * 0.95 / 20000)


def test_step_up_simes_false_discovery_rate_with_true_and_false_nulls():
    rng = np.random.default_rng(289)
    p = np.column_stack([np.zeros((20000, 4)), rng.uniform(size=(20000, 16))])
    rejects = multiple_testing(p, "simes").reject
    fdp = rejects[:, 4:].sum(axis=1) / rejects.sum(axis=1)
    assert abs(fdp.mean() - 0.05 * 16 / 20) < 6 * fdp.std(ddof=1) / math.sqrt(20000)


def test_sharpened_source_holm_stops_after_its_second_threshold():
    # First threshold .01 rejects 3; second .025 rejects one more. The source
    # returns here, although a further denominator update would reject .04.
    values = [0.001, 0.002, 0.003, 0.02, 0.04]
    np.testing.assert_array_equal(sharpened_testing(values, 5, "holm"), [True] * 4 + [False])
    np.testing.assert_array_equal(sharpened_testing([0.9, 1], 0.9), True)


@pytest.mark.parametrize("values", [[], 0.1, [0.1, np.nan], [np.inf], [-0.1], [1.1]])
def test_invalid_pvalues(values):
    with pytest.raises(ValueError):
        multiple_testing(values)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        multiple_testing([0.1, 0.2], alpha=alpha)


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_invalid_rom_family_size(n):
    with pytest.raises(ValueError):
        rom_critical_values(n)
