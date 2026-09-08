"""End-to-end uncertain-prior examples printed in SINGLE's single.doc."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    SingleStudySpecification,
    single_normal_criterion,
    single_prior_parameters,
)


@pytest.mark.parametrize(
    "variance,criterion,doses,subjects,printed",
    [
        (
            1,
            "quantile",
            [-1.639367, -0.513355, 2.172973],
            [70.424138, 9.588770, 19.987092],
            0.495569,
        ),
        (0.5, "slope", [-2.563760, 2.563760], [50, 50], 0.157458),
    ],
)
def test_manual_normal_lognormal_study(variance, criterion, doses, subjects, printed):
    # single.doc: mean(a)=0, mean(b)=1; a normal, b log-normal;
    # equal marginal variances and zero correlation. PEIGEN sets order=8.
    mean, covariance = single_prior_parameters(
        [0, 1],
        [variance, variance],
        np.eye(2),
        lognormal=[False, True],
        conversion="legacy",
    )
    prior = single_normal_criterion(
        doses,
        subjects,
        mean,
        covariance,
        lognormal=[False, True],
        criterion=f"{criterion}_sd",
        order=8,
        legacy_scale=True,
    )
    assert_allclose(prior.value, printed, atol=5e-7, rtol=0)
    study = SingleStudySpecification(
        prior.parameters,
        [-10, 10],
        prior_weights=prior.weights,
        criterion=criterion,
        aggregation="harmonic",
        max_doses=5,
        support_stopping="original",
    ).run()
    best = study.search.best
    assert best.doses.size == len(doses)
    assert_allclose(best.value, printed, atol=5e-7, rtol=0)
    # The modern local solver reaches a slightly different point from the
    # original. Compare at the original printed design/optimization precision.
    order = np.argsort(best.doses)
    assert_allclose(best.doses[order], doses, atol=0.003, rtol=0)
    assert_allclose(best.subjects[order], subjects, atol=0.03, rtol=0)
    assert best.value <= prior.value * (1 + 1e-10)
    checked = single_normal_criterion(
        best.doses,
        best.subjects,
        mean,
        covariance,
        lognormal=[False, True],
        criterion=f"{criterion}_sd",
        order=8,
        legacy_scale=True,
    )
    assert_allclose(best.value, checked.value, rtol=1e-12)
    assert study.search.stop_reason == "relative_improvement"
