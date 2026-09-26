"""Dose-selection contracts independently checked on simple analytic surfaces."""

import numpy as np
import pytest

from mdanderson_stats.toxfinder_decision import toxfinder_contour, toxfinder_stage1


def test_stage1_interpolation_on_offset_line_and_direct_midpoint_prediction():
    # L1 has an offset; the first agent alone determines p=x1/(1+x1).
    theta = [1, 1, 0, 1, 0, 1]
    levels = np.array([[0.1, 0.2], [0.3, 0.3], [0.6, 0.45], [1, 0.65]])
    result = toxfinder_stage1(levels, theta, levels[:3], [0, 0, 1], target=0.2)
    np.testing.assert_allclose(result.dose, [0.256, 0.278], atol=1e-14)
    assert result.estimated_toxicity == pytest.approx(0.2, abs=1e-14)
    midpoint = np.flatnonzero(np.isclose(result.candidate_doses[:, 0], 0.8)).item()
    assert result.mean_toxicity[midpoint] == pytest.approx(4 / 9, abs=1e-14)
    result = toxfinder_stage1(levels, theta, levels[:3], [0, 0, 1], target=0.49)
    np.testing.assert_allclose(result.dose, [0.8, 0.55], atol=1e-14)


def test_stage1_initial_choice_nearest_target_and_previously_tried_levels():
    theta = [1, 1, 0, 1, 0, 1]
    levels = np.column_stack(([0.1, 0.3, 0.6, 1], [0.1, 0.3, 0.6, 1]))
    first = toxfinder_stage1(levels, theta, [], [], target=0.9, start_index=1)
    np.testing.assert_allclose(first.dose, levels[1], atol=1e-14)
    # Nearest eligible dose can have posterior mean above target.
    next_dose = toxfinder_stage1(levels, theta, levels[:2], [0, 0], target=0.31)
    np.testing.assert_allclose(next_dose.dose, levels[2], atol=1e-14)
    tied = toxfinder_stage1(levels, theta, levels[:2], [0, 0], target=(0.3 / 1.3 + 0.6 / 1.6) / 2)
    np.testing.assert_allclose(tied.dose, levels[1], atol=1e-14)
    history = levels[[0, 1, 2, 1]]
    revisited = toxfinder_stage1(levels, theta, history, [0, 0, 0, 0], target=0.5)
    np.testing.assert_allclose(revisited.dose, levels[3], atol=1e-14)
    # First toxicity can occur after an earlier higher dose with no toxicity.
    expanded = toxfinder_stage1(levels, theta, history, [0, 0, 0, 1], target=0.9)
    np.testing.assert_allclose(expanded.dose, [0.45, 0.45], atol=1e-14)


def test_contour_endpoints_unattainable_points_and_dose_pairs():
    result = toxfinder_contour([1, 1, 1, 1, 0, 1], [0, 0.25, 1, 1.2], target=0.5)
    np.testing.assert_allclose(result.doses[:3], [[0, 1], [0.25, 0.75], [1, 0]], atol=1e-9)
    np.testing.assert_allclose(result.mean_toxicity[:3], 0.5, atol=1e-10)
    assert result.attainable.tolist() == [True, True, True, False]
    assert np.isnan(result.doses[3, 1])
