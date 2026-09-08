"""Distribution moments, censoring endpoints, reproducibility and file use."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import ExploratoryTable, generate_exponential_samples


def test_defaults_replay_and_file_roundtrip(tmp_path):
    first, second = generate_exponential_samples(rng=42)
    replay = generate_exponential_samples(rng=42)
    assert first.values.shape == (100, 2) and second.values.shape == (50, 2)
    for table, other in zip((first, second), replay, strict=True):
        assert_array_equal(table.values, other.values)
        assert np.all(np.diff(table.column("time")) >= 0)
        assert not table.values.flags.writeable
        path = tmp_path / "generated.txt"
        table.write(path)
        assert_array_equal(ExploratoryTable.read(path, table.names).values, table.values)
    generator = np.random.default_rng(42)
    a = generate_exponential_samples(rng=generator)[0]
    b = generate_exponential_samples(rng=generator)[0]
    assert not np.array_equal(a.values, b.values)


@pytest.mark.parametrize("p", [0, 0.4, 1])
def test_analytic_followup_and_status_moments(p):
    # E[Tobs]=(1-p/2)/rate; E[Tobs^2]=(2-4p/3)/rate^2.
    tables = generate_exponential_samples(100000, 100000, p, 2, 5, rng=102)
    for table, rate in zip(tables, [2, 5], strict=True):
        t, d = table.column("time"), table.column("status")
        assert abs(d.mean() - (1 - p)) < 0.008
        assert_allclose(t.mean(), (1 - p / 2) / rate, rtol=0.018)
        assert_allclose(np.mean(t * t), (2 - 4 * p / 3) / rate**2, rtol=0.045)
        if p == 0:
            assert np.all(d == 1)
        if p == 1:
            assert np.all(d == 0)


def test_inverse_rate_scaling_same_seed():
    a = generate_exponential_samples(30, 40, 0.3, 1, 2, rng=3)
    b = generate_exponential_samples(30, 40, 0.3, 4, 8, rng=3)
    for first, second in zip(a, b, strict=True):
        assert_array_equal(first.column("status"), second.column("status"))
        assert_allclose(first.column("time"), second.column("time") * 4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_first": 0},
        {"n_second": -1},
        {"n_first": 1.5},
        {"censor_probability": -1},
        {"censor_probability": 1.1},
        {"rate_first": 0},
        {"rate_second": np.inf},
        {"rate_first": np.nextafter(0.0, 1.0)},
    ],
)
def test_invalid_settings_preserve_supplied_rng(kwargs):
    rng = np.random.default_rng(12)
    state = rng.bit_generator.state
    with pytest.raises(ValueError):
        generate_exponential_samples(**kwargs, rng=rng)
    assert state == rng.bit_generator.state
