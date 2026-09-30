"""Checks for the archived STPLAN matched-pairs approximation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import ndtr, ndtri

from mdanderson_stats.stplan_matched_pairs import (
    stplan_matched_pairs_initial_size,
    stplan_matched_pairs_power,
)
from mdanderson_stats.stplan_planning import STPLAN_METHODS, stplan_solve


def test_forward_power_matches_original_psimpd_and_pmpd_equations():
    delta, n, alpha = 0.15, 700.0, 0.05
    z11, z10, z01, z00 = 100.0, 10.0, 30.0, 80.0
    total = z11 + z10 + z01 + z00
    a = z10 + z01
    b = z10 - z01
    r = (a + delta * b) / (2 * total)
    s = r**2 - delta * (b - delta * (z11 + z00)) / total
    psi = r + np.sqrt(s)
    denominator = np.sqrt(psi**2 - 0.25 * delta**2 * (3 + psi))
    expected = ndtr((-ndtri(1 - alpha) * psi + abs(delta) * np.sqrt(n * psi)) / denominator)

    assert_allclose(stplan_matched_pairs_power(delta, n, z11, z10, z01, z00), expected, atol=2e-15)
    assert_allclose(
        stplan_matched_pairs_power(delta, n, z11, z10, z01, z00, sides=2),
        ndtr((-ndtri(1 - alpha / 2) * psi + delta * np.sqrt(n * psi)) / denominator),
        atol=2e-15,
    )
    assert_allclose(stplan_matched_pairs_power(0, n, z11, z10, z01, z00), alpha, atol=2e-15)


def test_table_scale_extremes_and_broadcasting_preserve_pair_information():
    baseline = stplan_matched_pairs_power(0.12, 450, 100, 15, 25, 90)
    scaled = stplan_matched_pairs_power(0.12, 450, 1e250, 1.5e249, 2.5e249, 9e249)
    assert_allclose(scaled, baseline, rtol=2e-14, atol=0)

    got = stplan_matched_pairs_power(
        np.array([0.05, 0.1, 0.15]),
        500,
        100,
        15,
        25,
        90,
    )
    assert got.shape == (3,)
    assert np.all(np.diff(got) > 0)


def test_generic_inverse_registration_and_no_pilot_modes_recover_target():
    assert len(STPLAN_METHODS) == 26
    solved = stplan_solve(
        "stplan_matched_pairs_power",
        compute="sample_size",
        target_power=0.8,
        bounds=(2, 5000),
        parameters={"difference": 0.15, "z11": 100, "z10": 10, "z01": 30, "z00": 80},
    )
    assert_allclose(solved.achieved_power, 0.8, atol=1e-8, rtol=0)
    assert solved.value > 1

    default = stplan_matched_pairs_initial_size(0.15, 0.8)
    estimated = stplan_matched_pairs_initial_size(0.15, 0.8, theta1=0.8, theta2=0.2)
    assert default.input_mode == "source_default"
    assert default.psi == pytest.approx(0.82)
    assert default.recommended_preliminary_size == pytest.approx(default.sample_size / 6)
    assert default.achieved_power == pytest.approx(0.8, abs=1e-10)
    assert estimated.input_mode == "estimated_group_proportions"
    assert estimated.psi == pytest.approx(0.68)
    assert estimated.recommended_preliminary_size == pytest.approx(estimated.sample_size / 4)
    assert estimated.achieved_power == pytest.approx(0.8, abs=1e-10)
    # QMPD's squared closed form can produce a spurious positive n on its
    # negative square-root branch for targets below the null-side power.
    with pytest.raises(ValueError, match="positive sample-size solution"):
        stplan_matched_pairs_initial_size(0.15, 0.01)


def test_invalid_or_unidentified_inputs_fail_clearly():
    with pytest.raises(ValueError, match="theta1 and theta2"):
        stplan_matched_pairs_initial_size(0.1, 0.8, theta1=0.5)
    with pytest.raises(ValueError, match="pilot table total"):
        stplan_matched_pairs_power(0.1, 100, 0, 0, 0, 0)
    # All-concordant pilot data still has a defined source psi at positive
    # delta; only the zero-effect/zero-discordance boundary is unidentified.
    assert np.isfinite(stplan_matched_pairs_power(0.1, 100, 20, 0, 0, 80))
    with pytest.raises(ArithmeticError, match="variance parameter"):
        stplan_matched_pairs_power(0.0, 100, 20, 0, 0, 80)
    with pytest.raises(ValueError, match="supported numeric design parameter"):
        stplan_solve(
            "stplan_matched_pairs_power",
            compute="z10",
            target_power=0.8,
            bounds=(0.1, 0.9),
            parameters={"difference": 0.1, "sample_size": 500, "z11": 100, "z01": 25, "z00": 80},
        )
    with pytest.raises(ArithmeticError, match="denominator"):
        stplan_matched_pairs_initial_size(0.9, 0.8)
    with pytest.raises(ValueError, match="positive normal tail"):
        stplan_matched_pairs_power(0.1, 100, 100, 10, 30, 80, alpha=np.nextafter(0.0, 1.0), sides=2)
