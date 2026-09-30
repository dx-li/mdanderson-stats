import numpy as np
import pytest
from scipy.stats import beta as beta_distribution

from mdanderson_stats.mtpi import MTPIDesign
from mdanderson_stats.mtpi_isotonic_posterior import mtpi_isotonic_posterior_intervals


def test_single_dose_intervals_match_independent_beta_quantiles():
    draws = 12_000
    result = mtpi_isotonic_posterior_intervals(
        MTPIDesign(), [8], [3], draws=draws, rng=np.random.default_rng(781), confidence=0.9
    )
    expected = beta_distribution.ppf([0.05, 0.5, 0.95], 4, 6)
    np.testing.assert_allclose(
        [result.lower[0], result.median[0], result.upper[0]],
        expected,
        atol=0.012,
    )
    assert result.transformed_draws is None


def test_joint_draws_are_isotonic_and_untried_dose_uses_prior():
    result = mtpi_isotonic_posterior_intervals(
        MTPIDesign(),
        [8, 0, 5],
        [6, 0, 1],
        draws=64,
        rng=np.random.default_rng(84),
        weights=[1, 2, 1],
        retain_draws=True,
    )
    assert result.transformed_draws is not None
    assert result.transformed_draws.shape == (64, 3)
    assert np.all(np.diff(result.transformed_draws, axis=1) >= 0)
    assert np.all((result.lower >= 0) & (result.upper <= 1))
    assert not result.transformed_draws.flags.writeable
    assert not result.lower.flags.writeable


def test_two_uniform_dose_posteriors_match_closed_form_projection():
    result = mtpi_isotonic_posterior_intervals(
        MTPIDesign(),
        [0, 0],
        [0, 0],
        draws=24_000,
        rng=np.random.default_rng(521),
        retain_draws=True,
    )
    assert result.transformed_draws is not None
    np.testing.assert_allclose(result.mean, [5 / 12, 7 / 12], atol=0.009)
    # For two independent uniforms with equal-weight isotonic projection,
    # P(projected first dose <= .5) = .5 + .5**2 / 2 = 5/8.
    empirical_cdf = np.mean(result.transformed_draws[:, 0] <= 0.5)
    assert empirical_cdf == pytest.approx(5 / 8, abs=0.012)


def test_work_rejection_does_not_consume_generator_state():
    rng = np.random.default_rng(14)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match="max_work"):
        mtpi_isotonic_posterior_intervals(
            MTPIDesign(), [3, 3], [0, 3], draws=8, rng=rng, max_work=1
        )
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("confidence", [np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)])
def test_unrepresentable_quantile_levels_rejected_before_rng(confidence):
    rng = np.random.default_rng(141)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match="interior quantile"):
        mtpi_isotonic_posterior_intervals(
            MTPIDesign(), [3], [1], draws=8, rng=rng, confidence=confidence
        )
    assert rng.bit_generator.state == before
