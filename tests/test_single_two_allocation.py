"""Two-group allocation optima and independent precision checks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import single_optimize_two_sample_allocations, single_two_sample_precision


def test_reference_design_optimal_between_group_split():
    r = single_optimize_two_sample_allocations(([-1, 1], [0]), [0, 1, 0], measure="variance")
    w = np.exp(-1) / (1 + np.exp(-1)) ** 2
    # Var(a1-a2)=1/(N1*w)+4/N2, with equal within-group-1 allocation.
    fraction = (1 / np.sqrt(w)) / (1 / np.sqrt(w) + 2)
    assert_allclose(r.subjects[0], [50 * fraction, 50 * fraction], atol=0.002)
    assert_allclose(r.subjects[1], [100 * (1 - fraction)], atol=0.002)
    assert_allclose(r.value, (1 / np.sqrt(w) + 2) ** 2 / 100, rtol=1e-8)


@pytest.mark.parametrize("form", ["linear", "centered"])
def test_unequal_slopes_closed_form(form):
    b = [0, 0.8, 1.2] if form == "linear" else [0.8, 0, 1.2]
    r = single_optimize_two_sample_allocations(
        ([-1, 1], [-1, 1]), b, form=form, comparison="slope", measure="variance"
    )
    slopes = np.array([0.8, 1.2])
    w = np.exp(-slopes) / (1 + np.exp(-slopes)) ** 2
    proportions = (1 / np.sqrt(w)) / np.sum(1 / np.sqrt(w))
    assert_allclose(np.sum(r.subjects, axis=1), 100 * proportions, atol=0.002)
    assert_allclose(r.value, np.sum(1 / np.sqrt(w)) ** 2 / 100, rtol=1e-8)


@pytest.mark.parametrize("form", ["linear", "centered"])
@pytest.mark.parametrize("comparison", ["location", "slope"])
@pytest.mark.parametrize("measure,power", [("sd", 0.5), ("variance", 1)])
@pytest.mark.parametrize("aggregation", ["arithmetic", "harmonic"])
def test_symmetric_uncertain_prior_objective(form, comparison, measure, power, aggregation):
    if comparison == "location":
        b = [[0, 0.8, 0], [0, 1.2, 0]] if form == "linear" else [[0.8, 0, 0], [1.2, 0, 0]]
    else:
        b = [[0, 0.8, 0.8], [0, 1.2, 1.2]] if form == "linear" else [[0.8, 0, 0.8], [1.2, 0, 1.2]]
    x = ([-1, 1], [-1, 1])
    r = single_optimize_two_sample_allocations(
        x,
        b,
        prior_weights=[0.3, 0.7],
        form=form,
        comparison=comparison,
        measure=measure,
        aggregation=aggregation,
    )
    assert_allclose(r.subjects, [[25, 25], [25, 25]], atol=1e-8)
    local = (
        single_two_sample_precision(x, r.subjects, b, form=form, comparison=comparison).variance
        ** power
    )
    expected = (
        np.dot([0.3, 0.7], local)
        if aggregation == "arithmetic"
        else 1 / np.dot([0.3, 0.7], 1 / local)
    )
    assert_allclose(r.value, expected)
    assert r.optimality_gap < 1e-8


def test_group_swap_and_total_scaling():
    x = ([-1, 1], [0])
    r = single_optimize_two_sample_allocations(x, [0, 1, 0])
    swapped = single_optimize_two_sample_allocations(x[::-1], [0, 1, 0], total_subjects=200)
    for a, b in zip(r.subjects, swapped.subjects[::-1], strict=True):
        assert_allclose(2 * a, b, atol=0.002)
    assert_allclose(swapped.value, r.value / np.sqrt(2), rtol=1e-8)


def test_iteration_failure():
    with pytest.raises(RuntimeError, match="failed"):
        single_optimize_two_sample_allocations(([-1, 1], [0]), [0, 1, 0], max_iterations=1)


@pytest.mark.parametrize("form", ["linear", "centered"])
@pytest.mark.parametrize("comparison", ["location", "slope"])
def test_loglog_model_matches_independent_precision(form, comparison):
    x = ([-1, 0, 1], [-0.5, 0.5, 1.5])
    b = [[0.2, 0.7, 0.4], [0.3, 0.8, 0.5]]
    r = single_optimize_two_sample_allocations(
        x, b, prior_weights=[0.4, 0.6], model="loglog", form=form, comparison=comparison
    )
    local = single_two_sample_precision(
        x, r.subjects, b, model="loglog", form=form, comparison=comparison
    )
    assert_allclose(r.value, np.dot([0.4, 0.6], local.sd))
    assert r.value < r.initial_value


@pytest.mark.parametrize(
    "changes",
    [
        {"doses": ([0], [0])},
        {"doses": ([], [1])},
        {"doses": ([1],)},
        {"parameters": [0, 1]},
        {"parameters": [[0, 1, 0], [0, 2, 0]]},
        {"prior_weights": [0.5]},
        {"comparison": "ratio"},
        {"initial_subjects": ([20, 20], [20])},
        {"initial_subjects": ([0, 0], [100])},
        {"total_subjects": -1},
        {"max_iterations": True},
        {"measure": "power"},
    ],
)
def test_invalid_requests(changes):
    args = dict(doses=([-1, 1], [0]), parameters=[0, 1, 0])
    args.update(changes)
    with pytest.raises(ValueError):
        single_optimize_two_sample_allocations(**args)
