from fractions import Fraction
from math import comb, exp, log, log1p

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import betaln

from mdanderson_stats.mtpi import MTPIDesign
from mdanderson_stats.mtpi_isotonic_posterior import mtpi_isotonic_posterior_intervals
from mdanderson_stats.mtpi_simulation import simulate_mtpi


def integer_beta_cdf(x, alpha, beta):
    degree = alpha + beta - 1
    return sum(
        Fraction(comb(degree, k)) * x**k * (1 - x) ** (degree - k) for k in range(alpha, degree + 1)
    )


def test_common_beta_prior_changes_posterior_upm_and_safety_consistently():
    design = MTPIDesign(
        target=0.3,
        lower=0.25,
        upper=0.35,
        elimination_probability=0.8,
        prior_alpha=1,
        prior_beta=3,
    )
    result = design.posterior(3, 0)  # Posterior Beta(1, 6).
    lower = integer_beta_cdf(Fraction(1, 4), 1, 6)
    at_upper = integer_beta_cdf(Fraction(7, 20), 1, 6)
    at_target = integer_beta_cdf(Fraction(3, 10), 1, 6)
    expected_mass = [lower, at_upper - lower, 1 - at_upper]
    expected_mass_float = np.asarray(expected_mass, dtype=float)
    expected_upm = expected_mass_float / [0.25, 0.10, 0.65]
    np.testing.assert_allclose(result.probability, expected_mass_float, atol=2e-15)
    np.testing.assert_allclose(result.unit_probability_mass, expected_upm, atol=2e-14)
    assert result.move == 1 - int(np.argmax(expected_upm))
    assert result.overdose_probability == pytest.approx(float(1 - at_target), abs=2e-15)
    assert result.unsafe == (1 - float(at_target) > design.elimination_probability)


def test_fractional_table3_prior_matches_independent_density_quadrature():
    design = MTPIDesign(prior_alpha=0.05, prior_beta=0.05)
    result = design.posterior(2, 1)  # Posterior Beta(1.05, 1.05).
    a = b = 1.05
    normalizer = float(betaln(a, b))

    def density(x):
        if x <= 0 or x >= 1:
            return 0.0
        return exp((a - 1) * log(x) + (b - 1) * log1p(-x) - normalizer)

    lower = quad(density, 0.0, design.lower, epsabs=1e-13)[0]
    center = quad(density, design.lower, design.upper, epsabs=1e-13)[0]
    upper = quad(density, design.upper, 1.0, epsabs=1e-13)[0]
    target_tail = quad(density, design.target, 1.0, epsabs=1e-13)[0]
    np.testing.assert_allclose(result.probability, [lower, center, upper], atol=2e-12)
    assert result.overdose_probability == pytest.approx(target_tail, abs=2e-12)


def test_final_isotonic_mean_uses_nonuniform_posterior():
    design = MTPIDesign(prior_alpha=1, prior_beta=3)
    result = design.select_mtd([2, 2], [1, 0])
    # Posterior means are 1/3 and 1/6, so equal weights pool them to 1/4.
    np.testing.assert_allclose(result.isotonic_mean, [0.25, 0.25], atol=1e-15)


def test_default_prior_is_explicit_default_and_seeded_simulation_replays():
    default = MTPIDesign()
    explicit = MTPIDesign(prior_alpha=1, prior_beta=1)
    first = simulate_mtpi(default, [0.15, 0.35], trials=64, rng=148)
    second = simulate_mtpi(explicit, [0.15, 0.35], trials=64, rng=148)
    np.testing.assert_array_equal(first.patients, second.patients)
    np.testing.assert_array_equal(first.toxicities, second.toxicities)
    np.testing.assert_array_equal(first.selected_dose, second.selected_dose)
    nonuniform = simulate_mtpi(
        MTPIDesign(prior_alpha=1, prior_beta=3), [0.15, 0.35], trials=64, rng=148
    )
    repeat = simulate_mtpi(
        MTPIDesign(prior_alpha=1, prior_beta=3), [0.15, 0.35], trials=64, rng=148
    )
    np.testing.assert_array_equal(nonuniform.patients, repeat.patients)
    np.testing.assert_array_equal(nonuniform.selected_dose, repeat.selected_dose)


def test_isotonic_interval_draws_use_the_nonuniform_prior_including_untried_dose():
    seed = 886
    rng = np.random.default_rng(seed)
    result = mtpi_isotonic_posterior_intervals(
        MTPIDesign(prior_alpha=1, prior_beta=3),
        [0],
        [0],
        draws=16,
        rng=rng,
        retain_draws=True,
    )
    reference_rng = np.random.default_rng(seed)
    expected = np.array(
        [reference_rng.beta(np.array([1.0]), np.array([3.0]))[0] for _ in range(16)]
    )
    np.testing.assert_array_equal(result.transformed_draws[:, 0], expected)


@pytest.mark.parametrize("prior", [(0, 1), (1, 0), (float("inf"), 1), (1e-7, 1), (1e6, 1)])
def test_invalid_or_unrepresentable_common_prior_is_rejected(prior):
    with pytest.raises(ValueError):
        MTPIDesign(prior_alpha=prior[0], prior_beta=prior[1])
