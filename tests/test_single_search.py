"""Incremental search retains only sufficiently improved designs."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    single_design_precision,
    single_search_design,
    single_two_sample_precision,
)


def test_original_point_prior_design_stops_at_two_doses():
    r = single_search_design([0, 1], [-10, 10], max_doses=4)
    assert r.stop_reason == "relative_improvement"
    assert r.best is r.steps[0].design
    assert len(r.steps) == 2 and not r.steps[1].accepted
    assert r.steps[1].relative_improvement < 0.01
    assert_allclose(r.best.value, 0.444280, atol=1e-6)
    assert_allclose(r.best.doses, [-2.39936, 2.39936], atol=0.001)
    assert r.scan_evaluations == 45 + 70


def test_broad_prior_benefits_from_added_doses():
    b = [[-3, 1], [0, 1], [3, 1]]
    r = single_search_design(b, [-8, 8], prior_weights=[1 / 3] * 3, criterion="slope", max_doses=4)
    assert r.stop_reason == "max_doses"
    assert len(r.steps) == 3 and all(step.accepted for step in r.steps)
    assert r.best.value < r.steps[0].design.value * 0.65
    for step in r.steps:
        local = single_design_precision(step.design.doses, step.design.subjects, b).slope_sd
        assert_allclose(step.design.value, np.mean(local))
        assert_allclose(step.design.subjects.sum(), 100)
    for previous, current in zip(r.steps[:-1], r.steps[1:], strict=True):
        expected = (previous.design.value - current.design.value) / previous.design.value
        assert_allclose(current.relative_improvement, expected)
        assert expected >= 0.01


def test_high_threshold_reverts_to_previous_support_size():
    r = single_search_design(
        [[-3, 1], [0, 1], [3, 1]],
        [-8, 8],
        prior_weights=[1 / 3] * 3,
        criterion="slope",
        max_doses=4,
        relative_improvement=0.9,
    )
    assert r.stop_reason == "relative_improvement"
    assert r.best is r.steps[0].design
    assert r.steps[1].design.value < r.best.value
    assert not r.steps[1].accepted


def test_two_sample_search_and_limit():
    r = single_search_design([0, 1, 1], [-5, 5], comparison="slope", max_doses=3, scan_points=6)
    assert r.stop_reason == "relative_improvement"
    assert len(r.best.doses) == 2
    assert_allclose(sum(n.sum() for n in r.best.subjects), 100)
    value = single_two_sample_precision(
        r.best.doses, r.best.subjects, [0, 1, 1], comparison="slope"
    ).sd
    assert_allclose(r.best.value, value)
    limited = single_search_design([0, 1], [-5, 5], criterion="slope", max_doses=2, scan_points=4)
    assert limited.stop_reason == "max_doses" and len(limited.steps) == 1
    assert limited.scan_evaluations == 6


def test_all_failed_optimizations_are_not_convergence():
    with pytest.raises(RuntimeError, match="All feasible"):
        single_search_design([0, 1], [-5, 5], scan_points=3, max_iterations=1)


def test_no_identifiable_seed_is_explicit():
    with pytest.raises(ValueError, match="No feasible"):
        single_search_design([0, 0], [-5, 5])


@pytest.mark.parametrize(
    "changes",
    [
        {"max_doses": 1},
        {"max_doses": 11},
        {"scan_points": 21},
        {"scan_points": True},
        {"relative_improvement": -1},
        {"relative_improvement": 1},
        {"dose_bounds": [1, 1]},
        {"parameters": [[0, 1], [0, 2]]},
        {"prior_weights": [0.5]},
        {"model": "probit"},
        {"max_iterations": 0},
        {"aggregation": "geometric"},
    ],
)
def test_invalid_search(changes):
    args = dict(parameters=[0, 1], dose_bounds=[-5, 5])
    args.update(changes)
    with pytest.raises(ValueError):
        single_search_design(**args)
