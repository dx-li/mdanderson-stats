"""Normal conjugacy, data reduction, credible mass and independent density identities."""

from math import factorial, pi

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import NormalInverseGamma, NormalMeanPosterior, NormalSample


def test_manual_posterior_examples_and_corrected_variance_interval():
    sample = NormalSample.from_data([1, 2, 3, 4, 5])
    known = NormalMeanPosterior().update(sample, observation_variance=4)
    assert_allclose(known.location, 25 / 9)
    assert_allclose(known.variance, 20 / 27)
    assert_allclose(known.credible_interval(), [1.09091, 4.46465], atol=5e-5)
    unknown = NormalInverseGamma().update(sample)
    assert_allclose(
        [unknown.location, unknown.mean_precision, unknown.shape, unknown.scale],
        [5 / 3, 9, 4.5, 18],
    )
    assert_allclose(unknown.mean_scale, 2 / 3)
    assert_allclose(unknown.variance_expectation, 36 / 7)
    assert_allclose(unknown.mean_interval(), [0.15856189, 3.17477144], atol=1e-8)
    # The manual's claimed 95% IG(4.5,18) interval has only about 76% mass.
    mass = unknown.variance_cdf(6.53) - unknown.variance_cdf(1.87)
    assert 0.76 < mass < 0.77
    bounds = unknown.variance_interval()
    assert_allclose(unknown.variance_cdf(bounds[1]) - unknown.variance_cdf(bounds[0]), 0.95)
    assert_allclose(unknown.variance_pdf(bounds[0]), unknown.variance_pdf(bounds[1]), rtol=1e-10)


def test_sequential_and_pooled_analysis_are_equivalent():
    chunks = [[-2, 1, 3], [4, 8], [-1, 0, 5, 9]]
    known, unknown = NormalMeanPosterior(), NormalInverseGamma()
    for chunk in chunks:
        sample = NormalSample.from_data(chunk)
        known = known.update(sample, observation_variance=3)
        unknown = unknown.update(sample)
    pooled = NormalSample.from_data(np.concatenate(chunks))
    k = NormalMeanPosterior().update(pooled, observation_variance=3)
    u = NormalInverseGamma().update(pooled)
    assert_allclose([known.location, known.variance], [k.location, k.variance], rtol=2e-14)
    assert_allclose(
        [unknown.location, unknown.mean_precision, unknown.shape, unknown.scale],
        [u.location, u.mean_precision, u.shape, u.scale],
        rtol=2e-14,
    )


def test_raw_summary_batching_and_shift_invariance():
    data = np.array([[1, 2, 3, 4, 5], [-1, 4, 2, 5, 9]], dtype=float)
    raw = NormalSample.from_data(data)
    summary = NormalSample(data.mean(axis=-1), 5, sample_sd=data.std(axis=-1, ddof=1))
    assert_allclose(raw.sum_squares, summary.sum_squares)
    shifted = NormalSample.from_data(data + 1e12)
    assert_allclose(shifted.sum_squares, raw.sum_squares, rtol=1e-8)
    posterior = NormalInverseGamma(location=[0, 1]).update(raw)
    assert posterior.location.shape == (2,)
    assert not raw.mean.flags.writeable
    assert not posterior.scale.flags.writeable


def test_normal_and_inverse_gamma_density_identities():
    x = np.array([0.1, 0.5, 1, 3, 10])
    p = NormalInverseGamma(shape=3, scale=2)
    expected = 2**3 / factorial(2) * x**-4 * np.exp(-2 / x)
    assert_allclose(p.variance_pdf(x), expected, rtol=2e-14)
    # IG integer-shape CDF is a finite Poisson sum at scale/x.
    z = 2 / x
    assert_allclose(p.variance_cdf(x), np.exp(-z) * (1 + z + z**2 / 2), rtol=2e-14)
    cauchy = NormalInverseGamma(location=0, mean_precision=2, shape=0.5, scale=1)
    assert_allclose(cauchy.mean_pdf(x), 1 / (pi * (1 + x * x)), rtol=2e-14)
    assert_allclose(cauchy.mean_cdf(x), 0.5 + np.arctan(x) / pi)
    assert np.isnan(cauchy.mean_expectation)
    assert np.isinf(cauchy.variance_expectation)
    normal = NormalMeanPosterior(0, 1)
    assert_allclose(normal.pdf(x), np.exp(-x * x / 2) / np.sqrt(2 * pi))
    assert normal.sf(10) > 0 and normal.cdf(10) == 1


@pytest.mark.parametrize("shape", [0.25, 0.5, 1, 2, 4.5, 50])
def test_inverse_gamma_intervals_mass_and_density(shape):
    posterior = NormalInverseGamma(shape=shape, scale=3)
    mass = np.array([0.5, 0.8, 0.95])
    for method in ["highest-density", "equal-tail"]:
        bounds = posterior.variance_interval(mass, method=method)
        assert_allclose(
            posterior.variance_cdf(bounds[:, 1]) - posterior.variance_cdf(bounds[:, 0]),
            mass,
            atol=2e-10,
        )
        if method == "highest-density":
            assert_allclose(
                posterior.variance_pdf(bounds[:, 0]),
                posterior.variance_pdf(bounds[:, 1]),
                rtol=3e-8,
            )
        else:
            assert_allclose(posterior.variance_cdf(bounds[:, 0]), (1 - mass) / 2, atol=1e-12)
    mean_bounds = posterior.mean_interval(mass)
    assert_allclose(
        posterior.mean_cdf(mean_bounds[:, 1]) - posterior.mean_cdf(mean_bounds[:, 0]), mass
    )


def test_frequentist_intervals_and_missing_summary():
    sample = NormalSample(3, 5)
    known = sample.confidence_interval(observation_variance=4)
    assert_allclose(known, [1.2469549, 4.7530451], atol=1e-7)
    with pytest.raises(ValueError):
        NormalInverseGamma().update(sample)
    with pytest.raises(ValueError):
        sample.confidence_interval()
    full = NormalSample.from_data([1, 2, 3, 4, 5])
    assert_allclose(full.confidence_interval(), [1.0367568, 4.9632432], atol=1e-7)
    singleton = NormalSample(2, 1)
    assert np.isnan(singleton.sample_variance)
    assert NormalInverseGamma().update(singleton).shape == 2.5


def test_extreme_scales_without_reciprocal_or_square_overflow():
    known = NormalMeanPosterior(1e300, 1e-300).update(
        NormalSample(-1e300, 1), observation_variance=1e-300
    )
    assert_allclose(known.location, 0, atol=0)
    assert_allclose(known.variance / 1e-300, 0.5, rtol=1e-12)
    unknown = NormalInverseGamma(location=1e154, scale=1).update(NormalSample(-1e154, 1))
    assert_allclose(unknown.scale / 1e308, 1.6, rtol=3e-13)
    with pytest.raises(ArithmeticError):
        NormalSample.from_data([0, 1e-200])
    with pytest.raises(ArithmeticError):
        NormalSample.from_data([-1e308, 1e308])


def test_invalid_inputs():
    for factory in [
        lambda: NormalSample(0, 0),
        lambda: NormalSample(0, 2, sample_sd=-1),
        lambda: NormalInverseGamma(shape=0),
        lambda: NormalMeanPosterior(variance=0),
        lambda: NormalSample.from_data([]),
    ]:
        with pytest.raises(ValueError):
            factory()


def test_updating_preserves_location_shift_and_unit_changes():
    sample = NormalSample.from_data([1, 2, 3, 4, 5])
    baseline = NormalInverseGamma().update(sample)
    shifted = NormalInverseGamma(location=1e12).update(
        NormalSample.from_data(1e12 + np.arange(1, 6))
    )
    assert_allclose(shifted.scale, baseline.scale, rtol=2e-14)
    factor = 1e100
    scaled = NormalInverseGamma(scale=3 * factor**2).update(
        NormalSample.from_data(np.arange(1, 6) * factor)
    )
    assert_allclose(scaled.scale / factor**2, baseline.scale, rtol=2e-13)
    assert_allclose(scaled.mean_scale / factor, baseline.mean_scale, rtol=2e-13)
