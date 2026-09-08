"""Fixed-design information checked against native SINGLE and analytic identities."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import single_design_precision

CASES = json.loads((Path(__file__).parent / "fixtures/single.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_native_response_and_information(case):
    args = ([-1, 0, 1, 2], [10, 20, 30, 40], case["parameters"])
    if case["form"] == "centered" and case["parameters"][0] == 0:
        with pytest.raises(ValueError, match="Slope"):
            single_design_precision(*args, model=case["model"], form=case["form"])
        return
    r = single_design_precision(*args, model=case["model"], form=case["form"])
    assert_allclose(r.probability, case["probabilities"], rtol=1e-13)
    assert_allclose(r.information, case["information"], rtol=1e-12, atol=1e-13)


def test_supplied_single_example():
    # Rounded doses and published criterion from SINGLE's test.out.
    r = single_design_precision([-1.111111, 3.333333], [50, 50], [0, 1])
    assert_allclose(r.quantile_sd, 0.562565, atol=1e-6)
    optimized = single_design_precision([-2.399239, 2.399495], [90.741824, 9.258176], [0, 1])
    assert_allclose(optimized.quantile_sd, 0.444280, atol=1e-6)


def test_symmetric_logistic_closed_form():
    r = single_design_precision([-1, 1], [50, 50], [0, 1], quantile=0.1)
    w = np.exp(-1) / (1 + np.exp(-1)) ** 2
    assert_allclose(r.information, np.eye(2) * 100 * w, atol=1e-14)
    assert_allclose(r.slope_variance, 1 / (100 * w))
    assert_allclose(r.quantile_variance, (1 + np.log(0.1 / 0.9) ** 2) / (100 * w))


@pytest.mark.parametrize("model", ["logistic", "loglog"])
def test_parameterization_and_allocation_scaling(model):
    x = [-1, 0, 2]
    n = [20, 40, 40]
    a, b = -0.3, 1.2
    linear = single_design_precision(x, n, [a, b], model=model, quantile=[0.05, 0.5, 0.9])
    centered = single_design_precision(
        x, n, [b, -a / b], model=model, form="centered", quantile=[0.05, 0.5, 0.9]
    )
    assert_allclose(linear.probability, centered.probability)
    assert_allclose(linear.quantile_dose, centered.quantile_dose)
    assert_allclose(linear.quantile_variance, centered.quantile_variance)
    assert_allclose(linear.slope_variance, centered.slope_variance)
    double = single_design_precision(
        x, np.array(n) * 2, [a, b], model=model, quantile=[0.05, 0.5, 0.9]
    )
    assert_allclose(double.quantile_variance, linear.quantile_variance / 2)


@pytest.mark.parametrize("model,u", [("logistic", 40), ("loglog", 300)])
def test_information_survives_probability_rounding_to_one(model, u):
    r = single_design_precision([0, 1], [50, 50], [u, 1], model=model)
    assert np.all(r.probability == 1)
    assert np.all(np.isfinite(r.information))
    assert np.all(np.diagonal(r.information) > 0)
    assert np.isfinite(r.quantile_variance)


@pytest.mark.parametrize(
    "changes",
    [
        {"subjects": [-1, 2]},
        {"subjects": [0, 0]},
        {"doses": [1, 1]},
        {"parameters": [0, 0]},
        {"quantile": 1},
        {"model": "probit"},
        {"form": "quadratic"},
        {"parameters": [1]},
        {"doses": [np.nan, 1]},
    ],
)
def test_invalid_design(changes):
    args = dict(doses=[-1, 1], subjects=[50, 50], parameters=[0, 1])
    args.update(changes)
    with pytest.raises(ValueError):
        single_design_precision(**args)
