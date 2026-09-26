"""Independent EffTox formula and posterior integration references from base R."""

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.efftox_decision import EffToxContour
from mdanderson_stats.efftox_model import (
    EffToxPrior,
    efftox_log_likelihood,
    efftox_predict,
    efftox_standardize,
    fit_efftox,
)

COUNTS = np.array([[[4, 1], [2, 1]], [[2, 1], [4, 1]], [[1, 2], [4, 3]]])
BASE = np.array([-1, 0.8, 0.2, 1.1, -0.3, 0.0])


def _rows(name):
    with (Path(__file__).parent / "fixtures" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_joint_cells_and_likelihood_against_r():
    x = efftox_standardize([1, 2, 4])
    rows = _rows("efftox-joint.csv")
    for psi in (-5, 0, 2):
        theta = BASE.copy()
        theta[5] = psi
        actual = efftox_predict(x, theta)
        records = [r for r in rows if float(r["psi"]) == psi]
        expected = np.array([float(r["probability"]) for r in records]).reshape(3, 2, 2)
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-16)
        assert float(efftox_log_likelihood(x, COUNTS, theta)) == pytest.approx(
            float(records[0]["loglikelihood"]), abs=2e-13
        )


def test_contours_against_r_and_official_tutorial():
    for row in _rows("efftox-contour.csv"):
        contour = EffToxContour.from_points(*[float(row[k]) for k in ("e0", "t1", "em", "tm")])
        actual = contour.utility(float(row["e"]), float(row["t"]))
        assert float(actual) == pytest.approx(float(row["utility"]), abs=2e-12)
    contour = EffToxContour.from_points(0.5, 0.65, 0.7, 0.25)
    # Tutorial Scenario 2, native utilities rounded to two decimal places.
    actual = contour.utility([0.2, 0.4, 0.6, 0.8, 0.9], [0.05, 0.1, 0.15, 0.2, 0.3])
    np.testing.assert_allclose(actual, [-0.68, -0.37, -0.04, 0.28, 0.33], atol=0.005, rtol=0)


@pytest.mark.parametrize("case", ["independent_intercepts", "association", "positive_slope"])
def test_posterior_against_independent_integration(case):
    mean = BASE.copy()
    sd = np.zeros(6)
    if case == "independent_intercepts":
        sd[[0, 2]] = [0.9, 1.1]
    elif case == "association":
        mean[5], sd[5] = 0.4, 1.3
    else:
        mean[1], sd[1], mean[5] = -1, 0.8, 1.2
    fit = fit_efftox(
        [1, 2, 4],
        COUNTS,
        prior=EffToxPrior(mean, sd),
        draws=1800,
        warmup=300,
        chains=2,
        rng=np.random.default_rng(2026),
    )
    records = [r for r in _rows("efftox-posterior.csv") if r["case"] == case]
    for row in records:
        j, d = int(row["parameter"]) - 1, int(row["dose"]) - 1
        expected = float(row["value"])
        if row["quantity"] == "mean":
            assert abs(fit.summary.mean[j] - expected) < max(
                0.025, 6 * fit.summary.batch_mean_mcse[j]
            )
            assert fit.summary.split_rhat[j] < 1.08
        elif row["quantity"] == "sd":
            assert fit.summary.standard_deviation[j] == pytest.approx(expected, rel=0.10)
        else:
            probability = (
                fit.toxicity_probabilities[..., d] if j == 0 else fit.efficacy_probabilities[..., d]
            )
            if row["quantity"] == "probability":
                assert probability.mean() == pytest.approx(expected, abs=0.015)
            else:
                acceptable = probability < 0.3 if j == 0 else probability > 0.45
                assert acceptable.mean() == pytest.approx(expected, abs=0.04)
    assert np.all(fit.parameters[..., 1] > 0)
    np.testing.assert_array_equal(
        fit.parameters[..., sd == 0],
        np.broadcast_to(mean[sd == 0], fit.parameters[..., sd == 0].shape),
    )
