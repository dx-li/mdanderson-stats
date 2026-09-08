"""Fixed group sizes constrain the feasible designs and survive study replay."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    SingleStudySpecification,
    single_optimize_design,
    single_optimize_two_sample_allocations,
    single_two_sample_precision,
)


@pytest.mark.parametrize("measure", ["sd", "variance"])
@pytest.mark.parametrize("aggregation", ["arithmetic", "harmonic"])
def test_fixed_allocation_matches_symmetric_slope_information(measure, aggregation):
    totals = [50, 150]
    result = single_optimize_two_sample_allocations(
        ([-1, 1], [-1, 1]),
        [0, 1, 1],
        comparison="slope",
        total_subjects=200,
        group_totals=totals,
        measure=measure,
        aggregation=aggregation,
        initial_subjects=([10, 40], [100, 50]),
        tolerance=1e-12,
    )
    weight = np.exp(-1) / (1 + np.exp(-1)) ** 2
    variance = (1 / 50 + 1 / 150) / weight
    assert_allclose([n.sum() for n in result.subjects], totals)
    # Matching within-group proportions eliminate the common-intercept
    # nuisance contribution; the symmetric solution is not unique.
    assert_allclose(result.subjects[0] / 50, result.subjects[1] / 150, atol=1e-5)
    assert_allclose(result.value, variance ** (0.5 if measure == "sd" else 1), rtol=1e-9)
    assert result.optimality_gap < 1e-5


def test_joint_fixed_totals_and_scaling():
    def run(scale):
        return single_optimize_design(
            ([-1, 1], [-1, 1]),
            [0, 1, 1],
            [-5, 5],
            comparison="slope",
            total_subjects=200 * scale,
            group_totals=np.array([50, 150]) * scale,
        )

    result = run(1)
    for doses, counts, total in zip(result.doses, result.subjects, [50, 150], strict=True):
        assert_allclose(np.sort(doses), [-2.39935728, 2.39935728], atol=2e-4)
        assert_allclose(counts.sum(), total)
    independent = single_two_sample_precision(
        result.doses, result.subjects, [0, 1, 1], comparison="slope"
    )
    assert_allclose(result.value, independent.sd)
    assert_allclose(run(4).value, result.value / 2, rtol=1e-8)
    assert result.stationarity < 1e-4


def test_study_search_preserves_each_total_including_rejected_stages():
    totals = np.array([100.0, 100.0])
    study = SingleStudySpecification(
        [0, 1, 1],
        [-5, 5],
        comparison="slope",
        total_subjects=200,
        group_totals=totals,
        scan_points=4,
        max_doses=3,
    ).run()
    totals[:] = 1
    assert_allclose(study.specification.group_totals, [100, 100])
    assert not study.specification.group_totals.flags.writeable
    assert len(study.search.steps) == 2
    for step in study.search.steps:
        assert_allclose([n.sum() for n in step.design.subjects], [100, 100])
    replay = SingleStudySpecification.from_json(study.specification.to_json()).run()
    assert_allclose(replay.search.best.value, study.search.best.value)
    assert "group_totals\t100,100" in study.report()
    revised = study.revise(group_totals=[50, 150])
    assert_allclose([n.sum() for n in revised.search.best.subjects], [50, 150])


@pytest.mark.parametrize("totals", [[0, 200], [-1, 201], [100], [100, 101], [np.nan, 100]])
def test_invalid_group_totals(totals):
    with pytest.raises(ValueError, match="group_totals"):
        single_optimize_two_sample_allocations(
            ([-1, 1], [-1, 1]),
            [0, 1, 1],
            comparison="slope",
            total_subjects=200,
            group_totals=totals,
        )


@pytest.mark.parametrize("joint", [False, True])
def test_initial_counts_cannot_transfer_subjects_between_fixed_groups(joint):
    kwargs = dict(
        total_subjects=200,
        group_totals=[50, 150],
        initial_subjects=([50, 50], [50, 50]),
        comparison="slope",
    )
    with pytest.raises(ValueError, match="group_totals"):
        if joint:
            single_optimize_design(([-1, 1], [-1, 1]), [0, 1, 1], [-5, 5], **kwargs)
        else:
            single_optimize_two_sample_allocations(([-1, 1], [-1, 1]), [0, 1, 1], **kwargs)


def test_one_sample_rejects_group_totals():
    with pytest.raises(ValueError, match="two-sample"):
        single_optimize_design([-1, 1], [0, 1], [-5, 5], group_totals=[50, 50])


@pytest.mark.parametrize("model", ["logistic", "loglog"])
@pytest.mark.parametrize("form", ["linear", "centered"])
@pytest.mark.parametrize("comparison", ["location", "slope"])
@pytest.mark.parametrize("aggregation", ["arithmetic", "harmonic"])
@pytest.mark.parametrize("measure", ["sd", "variance"])
def test_weighted_prior_with_unequal_group_sizes(model, form, comparison, aggregation, measure):
    parameters = [[0.6, 1, 0.9], [0.9, 1.2, 1.1]]
    weights = np.array([0.3, 0.7])
    result = single_optimize_two_sample_allocations(
        ([-2, -0.3, 2], [-1, 1]),
        parameters,
        prior_weights=weights,
        total_subjects=200,
        group_totals=[70, 130],
        model=model,
        form=form,
        comparison=comparison,
        aggregation=aggregation,
        measure=measure,
    )
    assert_allclose([n.sum() for n in result.subjects], [70, 130])
    precision = single_two_sample_precision(
        result.doses,
        result.subjects,
        parameters,
        model=model,
        form=form,
        comparison=comparison,
    )
    local = getattr(precision, measure)
    expected = weights @ local if aggregation == "arithmetic" else 1 / (weights @ (1 / local))
    assert_allclose(result.value, expected, rtol=1e-12)
    assert result.value <= result.initial_value * (1 + 1e-10)
