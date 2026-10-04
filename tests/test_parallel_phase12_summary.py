import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.parallel_phase12_importance import Phase12ImportanceFit
from mdanderson_stats.parallel_phase12_summary import summarize_phase12_importance_fits

REFERENCE = Path(__file__).parent / "fixtures" / "phase12-probability-summary.json"


def _fit_from_native_vector(values, *, converged=True):
    row = np.asarray(values, dtype=float)
    assert row.shape == (60,)
    return Phase12ImportanceFit(
        posterior_mode=np.zeros(4),
        proposal_covariance=np.eye(4),
        response_probability_mean=np.full(6, 0.5),
        response_probability_second_moment=row[54:60],
        reference_superiority=row[:6],
        efficacy_probability=row[6:12],
        future_probability=row[12:18],
        pairwise_superiority=row[18:54].reshape(6, 6),
        toxicity_probability=np.full(6, 0.1),
        response_probability_mc_se=np.zeros(6),
        response_probability_second_moment_mc_se=np.zeros(6),
        reference_superiority_mc_se=np.zeros(6),
        efficacy_probability_mc_se=np.zeros(6),
        future_probability_mc_se=np.zeros(6),
        pairwise_superiority_mc_se=np.zeros((6, 6)),
        log_evidence=0.0,
        integrations=200,
        target_relative_error=0.001,
        component_relative_error=np.zeros(67),
        log_integral_mc_se=np.zeros(61),
        ratio_mc_se=np.zeros(60),
        converged=converged,
        mode_iterations=0,
        mode_gradient_norm=0.0,
    )


def test_streamed_probability_moments_match_native_running_variance_reference():
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert "synthetic posterior-summary vectors" in reference["scope"]

    for case in reference["cases"]:
        fits = (
            _fit_from_native_vector(row, converged=i % 2 == 0) for i, row in enumerate(case["rows"])
        )
        summary = summarize_phase12_importance_fits(fits)
        assert summary.analysis_call_count == len(case["rows"])
        assert summary.nonconverged_analysis_call_count == len(case["rows"]) // 2
        np.testing.assert_allclose(
            summary.component_means, case["component_means"], rtol=0, atol=2e-16
        )
        np.testing.assert_allclose(
            summary.component_sample_variances,
            case["component_sample_variances"],
            rtol=2e-12,
            atol=1e-24,
        )
        assert len(summary.component_labels) == 60
        assert summary.component_labels[0] == "reference_superiority[0]"
        assert summary.component_labels[18] == "pairwise_superiority[0,0]"
        assert summary.component_labels[54] == "response_probability_second_moment[0]"


def test_single_fit_and_invalid_or_oversized_iterables():
    one = _fit_from_native_vector(np.linspace(0, 1, 60), converged=False)
    summary = summarize_phase12_importance_fits(iter([one]))
    assert summary.analysis_call_count == 1
    assert summary.nonconverged_analysis_call_count == 1
    np.testing.assert_array_equal(summary.component_sample_variances, np.zeros(60))
    with pytest.raises(ValueError, match="at least one"):
        summarize_phase12_importance_fits(iter(()))
    with pytest.raises(ValueError, match="bounded limit"):
        summarize_phase12_importance_fits(iter([one, one]), max_fits=1)

    malformed = replace(one, pairwise_superiority=np.zeros((6, 5)))
    with pytest.raises(ValueError, match="shape"):
        summarize_phase12_importance_fits([malformed])
