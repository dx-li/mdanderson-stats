import numpy as np
import pytest
from scipy.stats import binom

from mdanderson_stats import (
    TOPBinaryDesign,
    TOPInfeasibleError,
    optimize_top_binary,
    simulate_top_binary,
)


def test_final_only_calibration_matches_exact_binomial_power():
    result = optimize_top_binary(
        8,
        0.2,
        0.5,
        2,
        3,
        cutoff_scales=[0.6, 0.8, 0.95],
        gammas=[0, 1],
        looks=[8],
        trials=20000,
        validation_trials=20000,
        rng=134,
    )
    for i, (c, gamma) in enumerate(result.parameter_pairs):
        design = TOPBinaryDesign(8, 0.2, c, gamma, looks=[8])
        minimum = design.boundaries().complete_go_min[0]
        exact = binom.sf(minimum - 1, 8, [0.2, 0.5])
        np.testing.assert_allclose(result.calibration_probability[i], exact, atol=0.012)
    feasible = np.flatnonzero(result.feasible)
    assert result.selected_index == max(
        feasible, key=lambda i: (result.calibration_probability[i, 1], -i)
    )
    minimum = result.design.boundaries().complete_go_min[0]
    np.testing.assert_allclose(
        result.validation_probability, binom.sf(minimum - 1, 8, [0.2, 0.5]), atol=0.012
    )
    np.testing.assert_array_equal(result.calibration_mean_patients, 8)


def test_delayed_calibration_and_holdout_are_reproducible():
    result = optimize_top_binary(
        20,
        0.2,
        0.4,
        4,
        2,
        cutoff_scales=[0.8, 0.95],
        gammas=[0.5, 1],
        looks=[5, 10, 15, 20],
        trials=1500,
        validation_trials=2000,
        rng=134,
    )
    assert result.calibration_seed != result.validation_seed
    assert not result.parameter_pairs.flags.writeable
    assert not result.validation_probability.flags.writeable
    for j, p in enumerate([0.2, 0.4]):
        for seed, trials, probability, mean_n in [
            (
                result.calibration_seed,
                result.trials,
                result.calibration_probability[result.selected_index],
                result.calibration_mean_patients[result.selected_index],
            ),
            (
                result.validation_seed,
                result.validation_trials,
                result.validation_probability,
                result.validation_mean_patients,
            ),
        ]:
            replay = simulate_top_binary(result.design, p, 4, 2, trials=trials, rng=seed)
            assert replay.success_probability == probability[j]
            assert replay.patients.mean() == mean_n[j]
    assert result.calibration_mean_patients[result.selected_index, 0] < 20


def test_infeasible_grid_does_not_return_an_unqualified_design():
    with pytest.raises(TOPInfeasibleError, match="extend the grid"):
        optimize_top_binary(
            4,
            0.2,
            0.4,
            1,
            2,
            cutoff_scales=[0.1],
            gammas=[0],
            type1_error=1e-8,
            trials=1000,
            validation_trials=100,
            rng=134,
        )
