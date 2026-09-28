import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.bacis import bacis_classify
from mdanderson_stats.bacis_theta import bacis_theta_posterior, sample_bacis_theta


def test_theta_curves_and_moments_match_independent_half_normal_reference():
    fixture_dir = Path(__file__).parent / "fixtures"
    inputs = defaultdict(list)
    with (fixture_dir / "bacis-classification.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            inputs[row["case"]].append(row)
    curves = defaultdict(list)
    with (fixture_dir / "bacis-theta-curves.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            curves[(row["case"], float(row["latent_precision"]))].append(row)
    moments = defaultdict(list)
    with (fixture_dir / "bacis-theta-moments.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            moments[(row["case"], float(row["latent_precision"]))].append(row)

    for (case, precision), rows in curves.items():
        source = inputs[case]
        classification = bacis_classify(
            [int(row["y"]) for row in source],
            [int(row["n"]) for row in source],
            phi_low=float(source[0]["phi_low"]),
            phi_high=float(source[0]["phi_high"]),
            classification_precision=float(source[0]["precision"]),
        )
        theta = np.array([float(row["theta"]) for row in rows[: len(rows) // len(source)]])
        result = bacis_theta_posterior(classification, theta, latent_precision=precision)
        expected = {
            field: np.array(
                [
                    [float(row[field]) for row in rows if int(row["group"]) == group]
                    for group in range(1, len(source) + 1)
                ]
            ).T
            for field in ("density", "cdf", "survival")
        }
        for field, values in expected.items():
            np.testing.assert_allclose(getattr(result, field), values, rtol=2e-12, atol=2e-14)
        np.testing.assert_allclose(result.cdf + result.survival, 1, atol=3e-16)
        assert not result.cdf.flags.writeable

        moment_rows = moments[(case, precision)]
        np.testing.assert_allclose(
            result.mean, [float(row["mean"]) for row in moment_rows], rtol=5e-13
        )
        np.testing.assert_allclose(
            result.variance, [float(row["variance"]) for row in moment_rows], rtol=5e-13
        )
        np.testing.assert_allclose(
            result.mean**2 + result.variance,
            [float(row["second_moment"]) for row in moment_rows],
            rtol=5e-13,
        )


def test_theta_samples_are_reproducible_and_respect_component_masses():
    classification = bacis_classify([2, 7], [25, 25])
    first = sample_bacis_theta(classification, draws=12_000, rng=np.random.default_rng(17))
    second = sample_bacis_theta(classification, draws=12_000, rng=np.random.default_rng(17))
    np.testing.assert_array_equal(first, second)
    observed = np.mean(first > 0, axis=0)
    np.testing.assert_allclose(observed, classification.high_probability, atol=0.015)
    assert not first.flags.writeable
    many_groups = bacis_classify([0] * 6, [1] * 6)
    with pytest.raises(ValueError, match="500,000"):
        sample_bacis_theta(many_groups, draws=100_000)
