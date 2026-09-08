"""Joint optimization examples and the analytic objective/gradient contract."""

import importlib

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    single_design_precision,
    single_optimize_design,
    single_two_sample_precision,
)


def test_original_quantile_design_example():
    r = single_optimize_design([-1.111111, 3.333333], [0, 1], [-10, 10])
    assert_allclose(r.doses, [-2.399239, 2.399495], atol=0.001)
    assert_allclose(r.subjects, [90.741824, 9.258176], atol=0.005)
    assert_allclose(r.value, 0.444280, atol=1e-6)
    assert r.value < r.initial_value and r.stationarity < 1e-4


def test_symmetric_slope_design_and_boundary_optimum():
    r = single_optimize_design([-1, 1], [0, 1], [-5, 5], criterion="slope")
    assert_allclose(r.doses, [-2.399363, 2.399363], atol=0.001)
    assert_allclose(r.subjects, [50, 50], atol=0.001)
    assert_allclose(r.value, 0.150888, atol=1e-6)
    bounded = single_optimize_design([-0.5, 0.5], [0, 1], [-1, 1], criterion="slope")
    assert_allclose(bounded.doses, [-1, 1], atol=1e-8)


def test_two_sample_joint_slope_design():
    x = ([-1, 1], [-1, 1])
    r = single_optimize_design(x, [0, 1, 1], [-5, 5], comparison="slope")
    assert_allclose(r.doses, [[-2.399363, 2.399363]] * 2, atol=0.002)
    assert_allclose(r.subjects, [[25, 25]] * 2, atol=0.002)
    actual = single_two_sample_precision(r.doses, r.subjects, [0, 1, 1], comparison="slope")
    assert_allclose(r.value, actual.sd)


@pytest.mark.parametrize("aggregation", ["arithmetic", "harmonic"])
@pytest.mark.parametrize("measure", ["sd", "variance"])
def test_uncertain_prior_value_and_scaling(aggregation, measure):
    options = dict(
        parameters=[[0, 0.8], [0, 1.2]],
        prior_weights=[0.4, 0.6],
        dose_bounds=[-5, 5],
        criterion="slope",
        aggregation=aggregation,
        measure=measure,
    )
    r = single_optimize_design([-1, 1], **options)
    local = single_design_precision(r.doses, r.subjects, options["parameters"]).slope_variance
    if measure == "sd":
        local = np.sqrt(local)
    expected = (
        np.dot([0.4, 0.6], local)
        if aggregation == "arithmetic"
        else 1 / np.dot([0.4, 0.6], 1 / local)
    )
    assert_allclose(r.value, expected)
    double = single_optimize_design([-1, 1], total_subjects=200, **options)
    assert_allclose(double.doses, r.doses, atol=1e-6)
    assert_allclose(double.value, r.value / (np.sqrt(2) if measure == "sd" else 2))


@pytest.mark.parametrize("model", ["logistic", "loglog"])
@pytest.mark.parametrize("form", ["linear", "centered"])
@pytest.mark.parametrize("comparison", [None, "location", "slope"])
@pytest.mark.parametrize("measure", ["sd", "variance"])
@pytest.mark.parametrize("aggregation", ["arithmetic", "harmonic"])
def test_analytic_gradient_delivered_to_solver(
    monkeypatch, model, form, comparison, measure, aggregation
):
    # Intercept the external solver boundary to check both dose and allocation
    # derivatives numerically before optimization changes the evaluation point.
    class Checked(Exception):
        pass

    def check(fun, start, **kwargs):
        _, derivative = fun(start)
        numerical = []
        for i in range(start.size):
            delta = np.zeros_like(start)
            delta[i] = 1e-6
            numerical.append((fun(start + delta)[0] - fun(start - delta)[0]) / 2e-6)
        assert_allclose(derivative, numerical, rtol=2e-5, atol=2e-7)
        raise Checked

    module = importlib.import_module("mdanderson_stats.single_optimize")
    monkeypatch.setattr(module, "minimize", check)
    b = np.array([[0.3, 0.8, 0.5], [0.5, 1.1, 0.7]])[:, : 2 if comparison is None else 3]
    x = [-1, 1] if comparison is None else ([-1, 1], [-0.5, 1.5])
    with pytest.raises(Checked):
        single_optimize_design(
            x,
            b,
            [-3, 3],
            prior_weights=[0.4, 0.6],
            model=model,
            form=form,
            comparison=comparison,
            measure=measure,
            aggregation=aggregation,
        )


def test_failed_iterations_are_explicit():
    with pytest.raises(RuntimeError, match="failed"):
        single_optimize_design([-1, 1], [0, 1], [-5, 5], max_iterations=1)


@pytest.mark.parametrize(
    "changes",
    [
        {"dose_bounds": [1, 1]},
        {"initial_doses": [-6, 1]},
        {"initial_doses": [1, 1]},
        {"total_subjects": 0},
        {"parameters": [1]},
        {"parameters": [[0, 1], [0, 2]]},
        {"prior_weights": [0.5]},
        {"initial_subjects": [20, 20]},
        {"max_iterations": True},
        {"measure": "power"},
        {"comparison": "location"},
    ],
)
def test_invalid_design_requests(changes):
    args = dict(initial_doses=[-1, 1], parameters=[0, 1], dose_bounds=[-5, 5])
    args.update(changes)
    with pytest.raises(ValueError):
        single_optimize_design(**args)
