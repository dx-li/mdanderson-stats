"""Independent bootstrap statistics and generation/fit workflow checks."""

from copy import deepcopy
from decimal import Decimal, localcontext

import numpy as np
import pytest

from mdanderson_stats import (
    sppcr_bootstrap,
    sppcr_bootstrap_summary,
    sppcr_fit_means,
    sppcr_frequencies,
    sppcr_generate,
)
from mdanderson_stats.sppcr_bootstrap import _series


@pytest.mark.parametrize("scale", [1e-150, 1, 1e150])
def test_population_statistics_against_independent_replicates(scale):
    means = np.array([[1, 2, 3], [2, 3, 5], [3, 4, 7], [1, 4, 2]]) * scale
    result = sppcr_bootstrap_summary(means, progenitor=(0, 1))
    # Extreme scales inherit log-domain ratio rounding from sppcr_frequencies;
    # small replicate variances amplify that input rounding.
    p = means / means.sum(axis=-1, keepdims=True)
    for series, values in [
        (result.mu, means),
        (result.calibration, means.sum(axis=-1)),
        (result.frequency, p),
        (result.mutant, p[:, 2]),
        (result.transformed_frequency, 2 * np.arcsin(np.sqrt(p))),
        (result.transformed_mutant, 2 * np.arcsin(np.sqrt(p[:, 2]))),
    ]:
        np.testing.assert_allclose(series.values, values, rtol=2e-12)
        np.testing.assert_allclose(series.mean, values.mean(axis=0), rtol=2e-12)
        np.testing.assert_allclose(series.variance, values.var(axis=0), rtol=2e-12)
        np.testing.assert_allclose(series.standard_deviation, values.std(axis=0), rtol=2e-12)


@pytest.mark.parametrize("base,step", [(1e12, 0.001), (1e150, 1e135), (0, 1e-170)])
def test_centered_variance_against_decimal(base, step):
    values = np.array([base, base + step, base + 3 * step, base + 8 * step])
    with localcontext() as ctx:
        ctx.prec = 100
        numbers = [Decimal.from_float(float(x)) for x in values]
        mean = sum(numbers) / 4
        var = sum((x - mean) ** 2 for x in numbers) / 4
        sd = var.sqrt()
    result = _series(values)
    assert result.mean == pytest.approx(float(mean), rel=2e-15, abs=0)
    assert result.variance == pytest.approx(float(var), rel=3e-15, abs=0)
    assert result.standard_deviation == pytest.approx(float(sd), rel=3e-15, abs=0)


def test_large_constant_and_true_variance_overflow():
    s = _series(np.full(100, 1e308))
    assert s.mean == 1e308 and s.variance == 0
    with pytest.raises(ArithmeticError):
        _series(np.array([0, 1e308]))


def test_all_zero_replicate_is_retained_not_conditioned_away():
    m = np.array([[[1, 2], [2, 3]], [[0, 0], [4, 5]], [[3, 2], [6, 7]]])
    s = sppcr_bootstrap_summary(m, progenitor=(0, 0))
    np.testing.assert_array_equal(s.defined, [[True, True], [False, True], [True, True]])
    assert np.isnan(s.frequency.values[1, 0]).all()
    assert np.isnan(s.frequency.mean[0]).all()
    assert np.isnan(s.mutant.variance[0])
    assert np.isfinite(s.frequency.mean[1]).all()
    np.testing.assert_allclose(s.mu.mean, m.mean(axis=0))
    np.testing.assert_allclose(s.calibration.mean, m.sum(axis=-1).mean(axis=0))


def test_zero_allele_and_structural_mutant_constant():
    s = sppcr_bootstrap_summary([[0, 1], [0, 2], [0, 3]], progenitor=(1, 1))
    np.testing.assert_array_equal(s.frequency.mean, [0, 1])
    np.testing.assert_array_equal(s.frequency.variance, [0, 0])
    assert s.mutant.mean == 0 and s.mutant.variance == 0
    assert np.all(s.defined)


def test_bootstrap_matches_separate_generation_and_scalar_fits():
    seen = np.array([[4, 8, 2], [10, 15, 5]])
    n = np.array([20, 30])
    a, b = np.random.default_rng(93), np.random.default_rng(93)
    result = sppcr_bootstrap([1, 2], seen, n, progenitor=(0, 1), rng=a, replicates=30)
    generated = sppcr_generate(seen / n[:, None], n, rng=b, replicates=30)
    np.testing.assert_array_equal(result.samples.seen, generated.seen)
    expected = np.stack(
        [sppcr_fit_means([1, 2], row, n, saturation="half").mu for row in generated.seen]
    )
    np.testing.assert_array_equal(result.fit.mu, expected)
    assert a.bit_generator.state == b.bit_generator.state
    for i in range(30):
        individual = sppcr_frequencies(expected[i], progenitor=(0, 1))
        np.testing.assert_array_equal(
            result.summary.frequency.values[i], individual.frequency.value
        )


def test_override_model_boundaries_and_zero_data():
    s = sppcr_bootstrap(
        [1, 2], [[0, 0], [0, 0]], 10, progenitor=(0, 1), rng=np.random.default_rng(1), replicates=3
    )
    assert not np.any(s.summary.defined)
    assert np.isnan(s.summary.frequency.mean).all()
    assert s.summary.calibration.mean == 0
    s = sppcr_bootstrap(
        [1, 2],
        [[0, 0], [0, 0]],
        10,
        progenitor=(0, 1),
        rng=np.random.default_rng(1),
        replicates=3,
        probability=[[0, 1], [0, 1]],
    )
    np.testing.assert_array_equal(s.fit.adjusted, [[False, True]] * 3)
    assert np.all(s.summary.defined)
    assert np.all(s.fit.mu[:, 0] == 0)
    assert s.summary.mutant.variance == 0


def test_multidimensional_batches_single_replicate_and_ownership():
    m = np.arange(1, 25, dtype=float).reshape(1, 2, 3, 4)
    s = sppcr_bootstrap_summary(m, progenitor=(0, 0))
    assert s.frequency.mean.shape == (2, 3, 4)
    assert np.all(s.frequency.variance == 0)
    m[:] = 0
    assert np.all(s.mu.values > 0)
    for array in [s.defined, s.mu.values, s.frequency.variance]:
        with pytest.raises(ValueError):
            array.setflags(write=True)
    empty = sppcr_bootstrap_summary(np.empty((3, 0, 2)), progenitor=(0, 1))
    assert empty.frequency.mean.shape == (0, 2)


@pytest.mark.parametrize(
    "mu",
    [
        [],
        [1, 2],
        np.empty((0, 2)),
        np.empty((2, 0)),
        [[-1, 2]],
        [[float("nan"), 2]],
        [[float("inf"), 2]],
    ],
)
def test_invalid_summary(mu):
    with pytest.raises(ValueError):
        sppcr_bootstrap_summary(mu, progenitor=(0, 0))


@pytest.mark.parametrize(
    "options",
    [
        dict(replicates=0),
        dict(replicates=True),
        dict(replicates=1.5),
        dict(progenitor=(0, 3)),
        dict(probability=1.1),
        dict(saturation="unknown"),
    ],
)
def test_invalid_bootstrap_does_not_consume_rng(options):
    rng = np.random.default_rng(3)
    before = deepcopy(rng.bit_generator.state)
    kwargs = dict(progenitor=(0, 1), replicates=3)
    kwargs.update(options)
    with pytest.raises(ValueError):
        sppcr_bootstrap([1], [[1, 2]], 10, rng=rng, **kwargs)
    assert rng.bit_generator.state == before
