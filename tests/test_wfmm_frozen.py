"""Conditional sampling, source floor freezes and independent conjugate checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import WFMMPrior, fit_wfmm_coefficients

CASES = json.loads((Path(__file__).parent / "fixtures/wfmm-native-frozen.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: str(c["multiplier"]))
def test_native_floor_components_remain_conditioned(case):
    q = np.array(case["initial"])
    sd = np.array(case["proposal_sd"])
    mask = sd == 0
    assert np.array_equal(np.argwhere(mask), case["frozen_indices"])
    expected = np.broadcast_to(q[mask], (32, mask.sum()))
    np.testing.assert_array_equal(case["frozen_draws"], expected)
    fit = fit_wfmm_coefficients(
        np.zeros((8, 16)),
        np.ones((8, 1)),
        np.repeat(np.eye(4), 2, axis=0),
        prior=WFMMPrior(0, 1, random_shape=2, random_scale=1, residual_shape=2, residual_scale=1),
        random_variance=q[:1],
        residual_variance=q[1:],
        proposal_sd=(sd[:1], sd[1:]),
        zero_proposal_policy="freeze",
        draws=32,
        warmup=8,
        chains=2,
        rng=np.random.default_rng(41),
    )
    draws = np.concatenate((fit.random_variances, fit.residual_variances), axis=2)
    np.testing.assert_array_equal(draws[:, :, mask], np.broadcast_to(expected, (2, 32, 8)))
    np.testing.assert_array_equal(fit.variance_proposals[mask], 0)
    np.testing.assert_array_equal(fit.variance_proposals[~mask], 80)
    assert not fit.variance_proposals.flags.writeable


def test_all_frozen_matches_fixed_variance_conditional_and_rng():
    def run(estimate):
        return fit_wfmm_coefficients(
            [[1.0, 2.0], [3.0, 0.0], [2.0, 1.0]],
            np.ones((3, 1)),
            prior=WFMMPrior(0.7, 2, residual_shape=2, residual_scale=1),
            residual_variance=[2.0, 3.0],
            estimate_variances=estimate,
            proposal_sd=(None, 0) if estimate else None,
            zero_proposal_policy="freeze" if estimate else "error",
            draws=32,
            warmup=8,
            rng=np.random.default_rng(812),
        )

    frozen, fixed = run(True), run(False)
    np.testing.assert_array_equal(frozen.coefficients, fixed.coefficients)
    np.testing.assert_array_equal(frozen.log_likelihood, fixed.log_likelihood)
    np.testing.assert_array_equal(frozen.variance_proposals, 0)
    np.testing.assert_array_equal(frozen.variance_acceptance, 0)


def test_active_residual_matches_conjugate_precision_with_other_component_frozen():
    observed = np.array([0.2, -0.4, 0.1, 0.7, -0.3])
    fit = fit_wfmm_coefficients(
        np.column_stack((observed, observed)),
        np.zeros((5, 1)),
        prior=WFMMPrior(0, 1, residual_shape=4, residual_scale=2),
        residual_variance=[1.0, 7.0],
        proposal_sd=(None, [0.2, 0.0]),
        zero_proposal_policy="freeze",
        draws=3000,
        warmup=500,
        rng=np.random.default_rng(912),
    )
    expected = (4 + 5 / 2) / (2 + observed @ observed / 2)
    assert np.mean(1 / fit.residual_variances[:, :, 0, 0]) == pytest.approx(expected, rel=0.08)
    np.testing.assert_array_equal(fit.residual_variances[:, :, 0, 1], 7.0)
    assert 0 < fit.variance_acceptance[0, 0] < 1
    assert fit.variance_proposals[0, 1] == 0


@pytest.mark.parametrize(
    "policy,scale,estimate",
    [
        ("error", 0, True),
        ("freeze", -1, True),
        ("freeze", np.nan, True),
        ("freeze", np.inf, True),
        ("unknown", 1, True),
        ("freeze", 0, False),
    ],
)
def test_reject_invalid_or_implicit_freezes(policy, scale, estimate):
    with pytest.raises(ValueError):
        fit_wfmm_coefficients(
            np.zeros((3, 1)),
            np.ones((3, 1)),
            prior=WFMMPrior(0, 1, residual_shape=2, residual_scale=1),
            residual_variance=1,
            proposal_sd=(None, scale),
            zero_proposal_policy=policy,
            estimate_variances=estimate,
            draws=8,
            warmup=0,
            rng=np.random.default_rng(1),
        )
