"""Grid selection versus independent full trial simulation and holdout replay."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import bop2_survival_design, optimize_bop2_survival, simulate_bop2_survival


@pytest.mark.parametrize("arrival", ["fixed", "poisson"])
@pytest.mark.parametrize("objective", ["power", "expected_sample_size"])
def test_grid_optimum_and_independent_holdout_match_full_simulation(arrival, objective):
    options = dict(accrual_rate=2, final_followup=2, arrival=arrival)
    candidates = []
    for scale in [0.5, 0.8, 0.95]:
        for gamma in [0, 0.5, 1]:
            rng = np.random.default_rng(98)
            design = bop2_survival_design(8, 1, cutoff_scale=scale, gamma=gamma, looks=[4, 8])
            oc = [
                simulate_bop2_survival(design, median, n_trials=257, rng=rng, **options)
                for median in [1, 2.5]
            ]
            error, power = [r.success_probability for r in oc]
            if error <= 0.4 and power >= 0.1:
                en = oc[0].expected_sample_size
                key = (-power, en) if objective == "power" else (en, -power)
                candidates.append((key, scale, gamma, oc))
    _, scale, gamma, expected = min(candidates, key=lambda c: c[0])
    result = optimize_bop2_survival(
        8,
        1,
        2.5,
        looks=[4, 8],
        cutoff_scales=[0.5, 0.8, 0.95],
        gammas=[0, 0.5, 1],
        type1_error=0.4,
        minimum_power=0.1,
        objective=objective,
        n_trials=257,
        n_validation=193,
        rng=98,
        **options,
    )
    assert (result.cutoff_scale, result.gamma) == (scale, gamma)
    assert_array_equal(
        result.calibration_oc.success_probability, [r.success_probability for r in expected]
    )
    assert_allclose(
        result.calibration_oc.expected_sample_size,
        [r.expected_sample_size for r in expected],
        rtol=1e-15,
    )
    # The generator now follows exactly one null and alternative calibration sample.
    holdout = [
        simulate_bop2_survival(result.calibration_design, median, n_trials=193, rng=rng, **options)
        for median in [1, 2.5]
    ]
    assert_array_equal(
        result.validation_oc.success_probability, [r.success_probability for r in holdout]
    )
    assert_array_equal(
        result.validation_oc.expected_sample_size, [r.expected_sample_size for r in holdout]
    )
    assert result.analysis_oc is result.validation_oc


def test_analysis_prior_does_not_change_selection_and_closest_ranking():
    options = dict(
        accrual_rate=1.5,
        final_followup=3,
        looks=[3, 6],
        cutoff_scales=[0.6, 0.9],
        gammas=[0, 1],
        type1_error=0.15,
        error_control="closest",
        n_trials=6001,
        n_validation=311,
        rng=121,
    )
    plain = optimize_bop2_survival(6, 1, 3, **options)
    informed = optimize_bop2_survival(6, 1, 3, analysis_prior=[2, 10], **options)
    assert (plain.cutoff_scale, plain.gamma) == (informed.cutoff_scale, informed.gamma)
    assert_array_equal(
        plain.calibration_oc.success_probability, informed.calibration_oc.success_probability
    )
    assert_array_equal(
        plain.validation_oc.success_probability, informed.validation_oc.success_probability
    )
    assert np.any(
        informed.analysis_oc.success_probability != informed.validation_oc.success_probability
    )
    keys = []
    for scale in [0.6, 0.9]:
        for gamma in [0, 1]:
            rng = np.random.default_rng(121)
            design = bop2_survival_design(6, 1, cutoff_scale=scale, gamma=gamma, looks=[3, 6])
            results = [
                simulate_bop2_survival(
                    design, median, accrual_rate=1.5, final_followup=3, n_trials=6001, rng=rng
                )
                for median in [1, 3]
            ]
            keys.append(
                (
                    (
                        abs(results[0].success_probability - 0.15),
                        -results[1].success_probability,
                        results[0].expected_sample_size,
                    ),
                    scale,
                    gamma,
                )
            )
    best = min(keys, key=lambda item: item[0])
    assert (plain.cutoff_scale, plain.gamma) == best[1:]
