"""Independent R references and ToxFinder's important numerical boundaries."""

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.toxfinder_model import (
    ToxFinderPrior,
    fit_toxfinder,
    toxfinder_log_probabilities,
    toxfinder_probabilities,
)


def _rows(name):
    with (Path(__file__).parent / "fixtures" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_official_surfaces_against_independent_r():
    rows = _rows("toxfinder-surfaces.csv")
    names = list(dict.fromkeys(row["scenario"] for row in rows))
    parameters = []
    expected = []
    for name in names:
        group = [r for r in rows if r["scenario"] == name]
        parameters.append(
            [float(group[0][k]) for k in ("alpha1", "beta1", "alpha2", "beta2", "alpha3", "beta3")]
        )
        expected.append([float(r["probability"]) for r in group])
    doses = [[float(r["x1"]), float(r["x2"])] for r in rows if r["scenario"] == names[0]]
    # Separate chains/draw dimensions and Cartesian dose axis.
    parameters = np.array(parameters).reshape(2, 3, 6)
    actual = toxfinder_probabilities(doses, parameters)
    assert actual.shape == (2, 3, 10, 2)
    np.testing.assert_allclose(actual[..., 1].reshape(6, 10), expected, rtol=3e-14, atol=1e-16)
    np.testing.assert_allclose(actual.sum(axis=-1), 1, atol=2e-16)


def test_tiny_gamma_prior_against_analytic_log_moments():
    prior = ToxFinderPrior([0.4, 2, 0.4, 2, 1, 0.05], [0, 0, 0, 0, 0, 3])
    fit = fit_toxfinder(
        [[0, 0]], [0], [0], prior=prior, draws=512, warmup=0, rng=np.random.default_rng(726)
    )
    reference = _rows("toxfinder-log-prior.csv")[0]
    values = fit.log_parameters[..., 5]
    mean, sd = float(reference["mean"]), float(reference["sd"])
    assert abs(values.mean() - mean) < 5 * sd / np.sqrt(values.size)
    assert abs(values.std(ddof=1) - sd) < 0.18 * sd
    assert np.count_nonzero(values < -745) > values.size / 3
    assert np.all(np.isfinite(values))


def test_nonconjugate_posterior_against_independent_r_integrals():
    prior = ToxFinderPrior([0.5, 2, 0.4, 3, 1, 0.5], [0.125, 0, 0, 0, 0, 0])
    fit = fit_toxfinder(
        [[0.5, 0], [1, 0], [1.5, 0]],
        [0, 1, 3],
        [2, 4, 4],
        prior=prior,
        draws=512,
        warmup=128,
        rng=np.random.default_rng(138),
    )
    reference = {r["quantity"]: float(r["value"]) for r in _rows("toxfinder-posterior.csv")}
    values = fit.log_parameters[..., 0]
    mcse = fit.parameter_summary.batch_mean_mcse[0]
    assert abs(values.mean() - reference["log_alpha1_mean"]) < max(5 * mcse, 0.025)
    assert abs(values.std(ddof=1) - reference["log_alpha1_sd"]) < 0.06
    assert abs(np.exp(values).mean() - reference["alpha1_mean"]) < 0.07
    assert fit.parameter_summary.split_rhat[0] < 1.1
    assert fit.evaluations > values.size


def test_log_domain_extremes_preserve_model_instead_of_clipping():
    # beta1,beta2 overflow on exponentiation; their products with beta3 equal1.
    logs = [-np.inf, 1000, -np.inf, 1000, 0, -1000]
    np.testing.assert_allclose(
        toxfinder_probabilities([[0.5, 0.5], [1, 1]], logs, log_parameters=True)[:, 1], [0.2, 0.5]
    )
    smallest = np.nextafter(0.0, 1.0)
    actual = toxfinder_log_probabilities([[smallest, 0]], [1, 0.25, 0, 1, 0, 1])[0, 1]
    expected = -np.logaddexp(0, -0.25 * np.log(smallest))
    assert actual == pytest.approx(expected, abs=1e-12)
    # A finite log probability survives an ordinary-probability underflow.
    actual = toxfinder_log_probabilities([[1e-100, 0]], [1, 8, 0, 1, 0, 1])[0, 1]
    assert actual == pytest.approx(8 * np.log(1e-100), abs=1e-12)
