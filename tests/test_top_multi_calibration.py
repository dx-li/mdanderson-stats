import numpy as np
import pytest

from mdanderson_stats.top_endpoints import TOPMultiEndpointDesign
from mdanderson_stats.top_multi_calibration import optimize_top_multiendpoint
from mdanderson_stats.top_multi_simulation import simulate_top_multiendpoint


def test_composite_null_finite_grid_matches_exact_final_look_reference():
    design = TOPMultiEndpointDesign(
        12,
        [0.05, 0.15, 0.25, 0.55],
        0.8,
        0,
        mode="efficacy_toxicity",
        windows=[1, 1],
        looks=[12],
    )
    null = np.array(
        [
            [0.05, 0.15, 0.25, 0.55],  # both marginal boundaries
            [0.02, 0.18, 0.08, 0.72],  # efficacy null, toxicity safe
            [0.15, 0.35, 0.15, 0.35],  # efficacy safe, toxicity null
        ]
    )
    alternative = np.array([0.05, 0.45, 0.05, 0.45])
    result = optimize_top_multiendpoint(
        design,
        null,
        alternative,
        1,
        cutoff_scales=[0.8, 0.95],
        gammas=[0],
        type1_error=0.1,
        trials=5000,
        validation_trials=2000,
        arrival="fixed",
        rng=184,
    )
    # Independent base-R complete-data enumeration for C=.8 and C=.95.
    exact = np.array(
        [
            [0.057319368045, 0.182654914272, 0.234360321242, 0.824225267212],
            [0.002122218968, 0.012788122710, 0.052102752776, 0.403831946287],
        ]
    )
    for candidate in range(2):
        tolerance = 5 * np.sqrt(exact[candidate] * (1 - exact[candidate]) / 5000)
        assert np.all(
            np.abs(result.calibration_probability[candidate] - exact[candidate]) < tolerance
        )
    # The boundary-only null would admit C=.8, but the two additional supplied
    # null points correctly make it infeasible.
    assert result.calibration_probability[0, 0] < 0.1
    assert np.any(result.calibration_probability[0, 1:3] > 0.1)
    assert result.feasible.tolist() == [False, True]
    assert result.selected_index == 1
    assert result.design.cutoff_scale == 0.95
    assert result.calibration_decision_probability.shape == (2, 4, 4)
    np.testing.assert_allclose(result.calibration_decision_probability.sum(axis=-1), 1)
    assert result.validation_seed != result.calibration_seed
    holdout_replay = simulate_top_multiendpoint(
        result.design,
        alternative,
        1,
        trials=2000,
        arrival="fixed",
        rng=result.validation_seed,
    )
    assert holdout_replay.success_probability == result.validation_probability[-1]
    np.testing.assert_allclose(holdout_replay.patients.mean(), result.validation_mean_patients[-1])


def test_hypothesis_validation_reproducibility_and_timing_input_ownership():
    design = TOPMultiEndpointDesign(
        4,
        [0.25] * 4,
        0.8,
        0,
        mode="efficacy_toxicity",
        windows=[2, 3],
        looks=[4],
    )
    null = np.array([[0.25, 0.25, 0.25, 0.25]])
    alternative = np.array([0.3, 0.4, 0.05, 0.25])
    timing = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
    original_timing = timing.copy()

    def run():
        return optimize_top_multiendpoint(
            design,
            null,
            alternative,
            1,
            cutoff_scales=[0.8],
            gammas=[0],
            type1_error=0.99,
            trials=100,
            validation_trials=100,
            arrival="exponential",
            truth_timing_probabilities=timing,
            rng=93,
        )

    first, replay = run(), run()
    assert first.calibration_seed == replay.calibration_seed
    assert first.validation_seed == replay.validation_seed
    np.testing.assert_array_equal(first.calibration_probability, replay.calibration_probability)
    np.testing.assert_array_equal(first.validation_probability, replay.validation_probability)
    np.testing.assert_array_equal(timing, original_timing)
    with pytest.raises(ValueError, match="null rows"):
        optimize_top_multiendpoint(
            design,
            [[0.3, 0.3, 0.1, 0.3]],  # neither endpoint is null
            alternative,
            1,
            cutoff_scales=[0.8],
            gammas=[0],
        )
    coprimary = TOPMultiEndpointDesign(4, [0.25] * 4, 0.8, 0, looks=[4])
    with pytest.raises(ValueError, match="coprimary"):
        optimize_top_multiendpoint(
            coprimary,
            [[0.4, 0.2, 0.1, 0.3]],  # efficacy exceeds both-null requirement
            [0.35, 0.25, 0.2, 0.2],
            1,
            cutoff_scales=[0.8],
            gammas=[0],
        )
    with pytest.raises(ValueError, match="max_work"):
        optimize_top_multiendpoint(
            design,
            null,
            alternative,
            1,
            cutoff_scales=[0.8],
            gammas=[0],
            trials=100,
            validation_trials=100,
            max_work=1,
        )


def test_runtime_scan_budget_exhaustion_is_distinct_from_preflight():
    design = TOPMultiEndpointDesign(4, [0.25] * 4, 0.5, 0, windows=[1, 2], looks=[4])
    # The one-scan-per-look preflight is exactly 3,200 cells. Complete-data
    # suspension rechecks require additional scans and exhaust this budget.
    with pytest.raises(ValueError, match="scan work exceeds max_work"):
        optimize_top_multiendpoint(
            design,
            [[0, 0, 0, 1]],
            [0.35, 0.25, 0.2, 0.2],
            1,
            cutoff_scales=[0.5],
            gammas=[0],
            trials=100,
            validation_trials=100,
            arrival="fixed",
            rng=9,
            max_work=3200,
        )
