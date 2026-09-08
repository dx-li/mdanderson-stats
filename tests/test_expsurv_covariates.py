"""GEN-DATA censoring, independent scalar normalization and row alignment."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import generate_exploratory_data
from mdanderson_stats.expsurv_simulation import _study_followup


def test_scalar_sample_sd_and_study_end_calculation():
    lifetime = np.array([1.0, 4.0, 2.0, 8.0])
    arrival = np.array([0.2, 1.5, 0.1, 1.9])
    mean = sum(lifetime) / 4
    sd = math.sqrt(sum((float(t) - mean) ** 2 for t in lifetime) / 3)
    end = [float(a) + float(t) / sd for t, a in zip(lifetime, arrival, strict=True)]
    expected_time = [min(2, t) - a for t, a in zip(end, arrival, strict=True)]
    duration, status = _study_followup(lifetime, arrival)
    assert_allclose(duration, expected_time, rtol=1e-14)
    assert_array_equal(status, [int(t < 2) for t in end])
    # Normalization must not overflow when the arbitrary latent scale changes.
    scaled = _study_followup(lifetime * 1e300, arrival)
    assert_allclose(scaled[0], duration)
    assert_array_equal(scaled[1], status)


def test_exact_study_end_is_censored():
    duration, status = _study_followup(np.array([0.0, 1.0, 2.0]), np.zeros(3))
    assert_array_equal(duration, [0, 1, 2])
    assert_array_equal(status, [1, 1, 0])


@pytest.mark.parametrize("lifetime", [[1, 1], [0, 0], [1, np.inf]])
def test_undefined_normalization_fails(lifetime):
    with pytest.raises(RuntimeError):
        _study_followup(np.array(lifetime), np.zeros(2))


def test_seeded_generation_alignment_censoring_and_uniform_covariates():
    table = generate_exploratory_data(10000, rng=109)
    assert table.names == ("length", "arrive", "status", "x", "y", "z")
    t, a, d, x, y, z = table.values.T
    assert np.all(np.diff(t) >= 0)
    assert np.all((a >= 0) & (a < 2))
    assert np.all((x >= 0) & (x < 1) & (y >= 0) & (y < 1))
    assert_array_equal(z, x * y)
    assert_allclose((t + a)[d == 0], 2, atol=1e-15)
    assert np.all((t + a)[d == 1] < 2)
    assert 0 < d.sum() < d.size
    assert abs(x.mean() - 0.5) < 0.015 and abs(y.mean() - 0.5) < 0.015
    assert abs(np.mean(x * y) - 0.25) < 0.015
    assert_array_equal(table.values, generate_exploratory_data(10000, rng=109).values)
    assert not table.values.flags.writeable


@pytest.mark.parametrize("n", [0, 1, -1, 2.5, np.inf, [2]])
def test_invalid_size_does_not_advance_rng(n):
    generator = np.random.default_rng(6)
    state = generator.bit_generator.state
    with pytest.raises(ValueError):
        generate_exploratory_data(n, rng=generator)
    assert generator.bit_generator.state == state
