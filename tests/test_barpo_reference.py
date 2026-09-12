"""BARPO probabilities and allocation formulas against independent base R."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.barpo import barpo_allocation, barpo_monitor, barpo_posterior

_FIXTURES = Path(__file__).parent / "fixtures"
_PRIOR = [[1, 1]] * 3
_SUCCESS = [1, 4, 8]
_FAILURE = [3, 2, 1]


def _rows(name):
    with (_FIXTURES / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_posterior_and_all_four_allocation_formulas_against_base_r():
    expected = _rows("barpo-posterior.csv")
    posterior = barpo_posterior(_SUCCESS, _FAILURE, prior=_PRIOR)
    np.testing.assert_allclose(
        posterior.best_probability, [float(r["best"]) for r in expected], rtol=1e-9, atol=5e-9
    )
    np.testing.assert_allclose(
        posterior.variance, [float(r["variance"]) for r in expected], rtol=1e-12
    )
    allocation_rows = _rows("barpo-allocation.csv")
    for method in ("BARCP", "BARN2N", "BARMTV", "DBCD"):
        for floor_name, floor in (
            ("none", [0, 0, 0]),
            ("control", [0.4, 0, 0]),
            ("all", [0.15] * 3),
        ):
            result = barpo_allocation(
                posterior,
                [5, 8, 12],
                method=method.lower(),
                tau=2 if method == "DBCD" else 0.7,
                tau1=0.5,
                max_n=50,
                target_probability=[0.2, 0.3, 0.5] if method == "DBCD" else None,
                minimum_probability=floor,
            )
            reference = [
                float(r["probability"])
                for r in allocation_rows
                if r["method"] == method and r["floor"] == floor_name
            ]
            np.testing.assert_allclose(result, reference, rtol=1e-8, atol=5e-9)


def test_monitoring_with_and_without_control_against_base_r():
    expected = _rows("barpo-posterior.csv")
    result = barpo_monitor(
        _SUCCESS,
        _FAILURE,
        prior=_PRIOR,
        theta_fut=0.25,
        pfut=0.35,
        theta_eff=0.65,
        peff=0.9,
        theta_final=0.5,
        pfinal=0.95,
    )
    for field, column in (
        ("futility_probability", "futility"),
        ("efficacy_probability", "efficacy"),
        ("final_efficacy_probability", "final_efficacy"),
    ):
        np.testing.assert_allclose(
            getattr(result, field), [float(r[column]) for r in expected], rtol=1e-12
        )
    np.testing.assert_array_equal(result.futile, [True, False, False])
    np.testing.assert_array_equal(result.efficacious, [False, False, True])
    np.testing.assert_array_equal(result.final_efficacious, [False, False, True])
    controlled = barpo_monitor(
        _SUCCESS, _FAILURE, prior=_PRIOR, control=True, pfut=0.8, peff=0.95, pfinal=0.85
    )
    np.testing.assert_allclose(
        controlled.efficacy_probability[1:],
        [float(r["above_control"]) for r in expected[1:]],
        rtol=1e-9,
        atol=5e-9,
    )
    np.testing.assert_array_equal(controlled.efficacious, [False, False, True])
    np.testing.assert_array_equal(controlled.final_efficacious, [False, True, True])


def test_zero_power_needs_no_resolution_of_rare_best_arm_events():
    posterior = barpo_posterior([0, 50], [50, 0], prior=[[1, 1]] * 2)
    assert posterior.best_probability[0] < posterior.best_probability_error[0]
    np.testing.assert_array_equal(barpo_allocation(posterior, [50, 50], tau=0), [0.5, 0.5])
