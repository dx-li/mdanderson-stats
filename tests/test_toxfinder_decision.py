import numpy as np

from mdanderson_stats.toxfinder_decision import toxfinder_contour, toxfinder_stage1


def test_stage1_offset_segment_interpolation_and_midpoint() -> None:
    levels = np.array([[0.1, 0.2], [0.3, 0.3], [0.6, 0.45], [1, 0.65]])
    draws = np.tile([1, 1, 0, 1, 0, 1], (2, 1))
    result = toxfinder_stage1(
        levels,
        draws,
        levels[:3],
        [0, 0, 1],
        target=0.2,
    )
    np.testing.assert_allclose(result.dose, [0.256, 0.278], atol=1e-12)
    midpoint = np.flatnonzero(np.all(np.isclose(result.candidate_doses, [0.8, 0.55]), axis=1))[0]
    np.testing.assert_allclose(result.mean_toxicity[midpoint], 4 / 9)
    assert result.first_toxic_dose.tolist() == [0.6, 0.45]
    assert not result.candidate_doses.flags.writeable
    assert not result.eligible.flags.writeable


def test_stage1_empty_start_and_upward_no_skip() -> None:
    levels = np.array([[0, 0], [0.5, 0.5], [1, 1]])
    draws = np.tile([1, 1, 1, 1, 0, 1], (2, 1))
    initial = toxfinder_stage1(levels, draws[0], np.empty((0, 2)), [], target=0.6, start_index=1)
    np.testing.assert_array_equal(initial.dose, levels[1])
    result = toxfinder_stage1(
        levels, draws, [[0, 0], [0.5, 0.5], [1, 1], [0.5, 0.5]], [0, 0, 0, 0], target=0.8
    )
    np.testing.assert_array_equal(result.dose, levels[2])
    after_toxicity = toxfinder_stage1(levels, draws, [[0, 0], [0.5, 0.5]], [0, 1], target=0.8)
    np.testing.assert_array_equal(after_toxicity.dose, [0.75, 0.75])


def test_contour_solves_mean_target_and_marks_unattainable() -> None:
    draws = np.tile([1, 1, 1, 1, 0, 1], (3, 1))
    result = toxfinder_contour(draws, [0.2, 1.2], target=0.5)
    np.testing.assert_allclose(result.doses[0], [0.2, 0.8], atol=1e-9)
    np.testing.assert_allclose(result.mean_toxicity[0], 0.5, atol=1e-9)
    assert result.attainable.tolist() == [True, False]
    assert np.isnan(result.doses[1, 1])
    boundary = toxfinder_contour(draws[:1], [1.0], target=0.5)
    np.testing.assert_array_equal(boundary.doses, [[1, 0]])
    assert boundary.attainable.tolist() == [True]
