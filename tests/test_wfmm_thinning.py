"""Thinning retains exact subsequences of the same complete transition stream."""

import json
from dataclasses import replace

import numpy as np
import pytest

from mdanderson_stats import (
    WFMMPrior,
    WFMMStudy,
    fit_wfmm_coefficients,
    read_wfmm_study,
    run_wfmm_study,
)


@pytest.mark.parametrize("thin", [1, 2, 3, 5])
@pytest.mark.parametrize("random_effects", [False, True])
@pytest.mark.parametrize("estimate", [False, True])
def test_exact_dense_subsequence_rng_and_proposal_counts(thin, random_effects, estimate):
    kwargs = dict(
        coefficients=[[1.0, 2.0], [3.0, 0.0], [2.0, 1.0], [0.0, 3.0]],
        fixed_design=np.ones((4, 1)),
        random_design=np.repeat(np.eye(2), 2, axis=0) if random_effects else None,
        prior=WFMMPrior(
            0.7, 2.0, random_shape=2.0, random_scale=1.0, residual_shape=2.0, residual_scale=1.0
        ),
        random_variance=1.0 if random_effects else None,
        residual_variance=2.0,
        estimate_variances=estimate,
        proposal_sd=(0.2 if random_effects else None, 0.2) if estimate else None,
        sample_random_effects=random_effects,
        warmup=3,
    )
    dense_rng = np.random.default_rng(219)
    thin_rng = np.random.default_rng(219)
    dense = fit_wfmm_coefficients(**kwargs, draws=8 * thin, rng=dense_rng)
    sparse = fit_wfmm_coefficients(**kwargs, draws=8, thin=thin, rng=thin_rng)
    for field in (
        "coefficients",
        "inclusion_indicators",
        "random_variances",
        "residual_variances",
        "log_likelihood",
    ):
        np.testing.assert_array_equal(
            getattr(sparse, field), getattr(dense, field)[:, thin - 1 :: thin]
        )
    if random_effects:
        np.testing.assert_array_equal(
            sparse.random_effects, dense.random_effects[:, thin - 1 :: thin]
        )
    np.testing.assert_array_equal(sparse.variance_proposals, dense.variance_proposals)
    np.testing.assert_array_equal(sparse.variance_acceptance, dense.variance_acceptance)
    assert sparse.likelihood_evaluations == dense.likelihood_evaluations
    assert thin_rng.bit_generator.state == dense_rng.bit_generator.state
    assert sparse.thin == thin


@pytest.mark.parametrize("thin", [0, -1, 101, True, 1.2, np.nan])
def test_invalid_thinning_rejected(thin):
    with pytest.raises(ValueError):
        fit_wfmm_coefficients(
            np.ones((3, 1)),
            np.ones((3, 1)),
            prior=WFMMPrior(0.5, 1),
            residual_variance=1,
            estimate_variances=False,
            thin=thin,
            draws=8,
            rng=np.random.default_rng(0),
        )


def test_discarded_transitions_count_toward_work_budget():
    with pytest.raises(ValueError, match="work"):
        fit_wfmm_coefficients(
            np.ones((5, 2)),
            np.ones((5, 1)),
            prior=WFMMPrior(0.5, 1),
            residual_variance=1,
            estimate_variances=False,
            thin=100,
            draws=5000,
            rng=np.random.default_rng(0),
        )


def test_saved_thinning_replay_and_historical_default():
    y = np.array(
        [
            [1.0, 4, 2, 3],
            [2, 2, 3, 1],
            [4, 3, 1, 5],
            [5, 6, 4, 2],
            [6, 3, 7, 4],
            [3, 1, 2, 6],
            [2, 5, 5, 3],
            [4, 2, 6, 2],
        ]
    )
    base = WFMMStudy(y, np.ones((8, 1)), draws=24, warmup=4, prediction_design=[[1]])
    thin = replace(base, draws=8, thin=3)
    a = run_wfmm_study(base)
    b = run_wfmm_study(read_wfmm_study(thin.to_json()))
    np.testing.assert_array_equal(b.fit.coefficients, a.fit.coefficients[:, 2::3])
    report = json.loads(b.report_json())
    assert (
        report["thinning"] == 3
        and report["retained_draws_per_chain"] == 8
        and report["iterations_per_chain"] == 28
    )
    assert b.report_json() == run_wfmm_study(thin).report_json()
    data = json.loads(base.to_json())
    data.pop("thin")
    assert read_wfmm_study(json.dumps(data)).thin == 1
