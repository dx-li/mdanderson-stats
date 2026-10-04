from types import SimpleNamespace

import numpy as np

from mdanderson_stats.parallel_phase12_summary import (
    _COMPONENT_LABELS,
    Phase12ProbabilitySummary,
    _Phase12ProbabilityMoments,
)
from mdanderson_stats.phase12_calendar_oc import _LaplaceParameterMoments


def test_probability_moments_merge_trial_summaries_for_a_a_b_calls():
    zero = np.zeros(60)
    summaries = (
        Phase12ProbabilitySummary(2, 1, _COMPONENT_LABELS, np.full(60, 0.5), np.full(60, 0.5)),
        Phase12ProbabilitySummary(1, 0, _COMPONENT_LABELS, zero, zero),
    )
    pooled = _Phase12ProbabilityMoments()
    for summary in summaries:
        pooled.merge(summary)
    result = pooled.to_summary()

    assert result is not None
    assert result.analysis_call_count == 3
    assert result.nonconverged_analysis_call_count == 1
    np.testing.assert_allclose(result.component_means, np.full(60, 1 / 3), rtol=0, atol=1e-16)
    np.testing.assert_allclose(
        result.component_sample_variances, np.full(60, 1 / 3), rtol=0, atol=1e-16
    )


def test_available_fit_laplace_mixture_variance_has_hand_computable_values():
    fits = (
        SimpleNamespace(
            posterior_mode=np.array([1.0, 2.0, 0.0, -1.0]),
            proposal_covariance=np.diag([0.25, 1.0, 0.0, 0.5]),
            converged=True,
        ),
        SimpleNamespace(
            posterior_mode=np.array([3.0, 2.0, 4.0, 1.0]),
            proposal_covariance=np.diag([0.75, 3.0, 2.0, 1.5]),
            converged=False,
        ),
    )
    moments = _LaplaceParameterMoments()
    for fit in fits:
        moments.add(fit)

    assert moments.count == 2
    assert moments.nonconverged_count == 1
    np.testing.assert_allclose(moments.mode_mean, [2.0, 2.0, 2.0, 0.0], rtol=0, atol=0)
    np.testing.assert_allclose(moments.mixture_variance(), [1.5, 2.0, 5.0, 2.0], rtol=0, atol=0)
