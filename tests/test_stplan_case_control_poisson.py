"""Focused formula and finite-mixture checks for STPLAN case-control methods."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import ndtr, ndtri
from scipy.stats import binom, poisson

from mdanderson_stats.stplan_case_control import (
    stplan_case_control_power,
    stplan_matched_case_control_power,
)
from mdanderson_stats.stplan_poisson import stplan_poisson_two_sample_power


def _conditional_exact_power(null, alternative, n, alpha, *, fixed_upper=False):
    if n == 0:
        return 0.0
    if alternative <= null and not fixed_upper:
        critical = max(k for k in range(-1, n + 1) if binom.cdf(k, n, null) <= alpha)
        return 0.0 if critical < 0 else float(binom.cdf(critical, n, alternative))
    critical = min(k for k in range(n + 2) if binom.sf(k - 1, n, null) <= alpha)
    return 0.0 if critical > n else float(binom.sf(critical - 1, n, alternative))


def _exposure_rates(freq, risk_exposed, risk_unexposed):
    disease = freq * risk_exposed + (1 - freq) * risk_unexposed
    nondisease = 1 - disease
    return (
        freq * risk_exposed / disease,
        freq * (1 - risk_exposed) / nondisease,
    )


def test_unmatched_case_control_matches_bayes_and_signed_arcsine_power():
    freq, risk_e, risk_u, nc, n0, alpha = 0.3, 0.2, 0.05, 120.0, 150.0, 0.05
    p_case, p_control = _exposure_rates(freq, risk_e, risk_u)
    difference = 2 * np.arcsin(np.sqrt(p_case)) - 2 * np.arcsin(np.sqrt(p_control))
    expected = ndtr(difference / np.sqrt(1 / nc + 1 / n0) + ndtri(alpha))
    assert_allclose(stplan_case_control_power(freq, risk_e, risk_u, nc, n0), expected)
    assert stplan_case_control_power(freq, risk_u, risk_e, nc, n0) < alpha
    assert stplan_case_control_power(freq, risk_u, risk_e, nc, n0, sides=2) > alpha


def test_matched_case_control_correct_upper_tail_discordance_mixture():
    freq, risk_e, risk_u, n_pairs, alpha = 0.3, 0.05, 0.1, 24, 0.05
    p_case, p_control = _exposure_rates(freq, risk_e, risk_u)
    p_case_only = p_case * (1 - p_control)
    p_discordant = p_case_only + (1 - p_case) * p_control
    conditional = p_case_only / p_discordant
    expected = sum(
        binom.pmf(k, n_pairs, p_discordant)
        * _conditional_exact_power(0.5, conditional, k, alpha, fixed_upper=True)
        for k in range(n_pairs + 1)
    )
    assert_allclose(
        stplan_matched_case_control_power(freq, risk_e, risk_u, n_pairs), expected, atol=1e-14
    )
    assert stplan_matched_case_control_power(freq, risk_e, risk_u, n_pairs) < 0.05


def test_matched_case_control_two_sided_extension_uses_alternative_direction():
    one_sided = stplan_matched_case_control_power(0.3, 0.05, 0.1, 60)
    two_sided = stplan_matched_case_control_power(0.3, 0.05, 0.1, 60, sides=2)
    assert two_sided > one_sided


def test_two_sample_poisson_matches_explicit_conditional_finite_sum():
    rate1, rate2, exposure1, exposure2, alpha = 0.4, 0.6, 10.0, 8.0, 0.05
    mean1, mean2 = rate1 * exposure1, rate2 * exposure2
    total = mean1 + mean2
    pnull = exposure1 / (exposure1 + exposure2)
    palt = mean1 / total
    cutoff = int(np.ceil(poisson.isf(1e-10, total)))
    expected = sum(
        poisson.pmf(n, total) * _conditional_exact_power(pnull, palt, n, alpha)
        for n in range(cutoff + 1)
    )
    assert_allclose(
        stplan_poisson_two_sample_power(rate1, rate2, exposure1, exposure2),
        expected,
        atol=1.1e-10,
    )


def test_two_sample_poisson_equal_rates_and_work_bound():
    power = stplan_poisson_two_sample_power(0.5, 0.5, 10, 20)
    assert 0 <= power <= 0.05
    with pytest.raises(ValueError, match="200000"):
        stplan_poisson_two_sample_power(10_000, 10_000, 20, 20)
