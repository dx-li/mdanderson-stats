"""Full size/parameter simulation enumeration and independent stream isolation."""

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats import (
    BOP2InfeasibleError,
    bop2_survival_design,
    optimize_bop2_survival_sample_size,
    simulate_bop2_survival,
)


@pytest.mark.parametrize("arrival", ["fixed", "poisson"])
def test_both_size_objectives_against_full_grid_simulation(arrival):
    sizes = [4, 6, 8]
    seeds = np.random.default_rng(44).integers(0, np.iinfo(np.int64).max, size=len(sizes))
    candidates = []
    for n, seed in zip(sizes, seeds):
        looks = [v for v in [2, 4, 6] if v < n] + [n]
        for scale, gamma in product([0.5, 0.8, 0.95], [0, 0.5, 1]):
            design = bop2_survival_design(n, 1, cutoff_scale=scale, gamma=gamma, looks=looks)
            rng = np.random.default_rng(int(seed))
            results = [
                simulate_bop2_survival(
                    design,
                    median,
                    accrual_rate=2,
                    final_followup=2,
                    n_trials=503,
                    arrival=arrival,
                    rng=rng,
                )
                for median in [1, 3]
            ]
            error, power = [r.success_probability for r in results]
            if error <= 0.2 and power >= 0.3:
                candidates.append((results[0].expected_sample_size, n, -power, scale, gamma))
    for objective in ["expected_sample_size", "minimax"]:
        result = optimize_bop2_survival_sample_size(
            sizes,
            1,
            3,
            minimum_power=0.3,
            type1_error=0.2,
            accrual_rate=2,
            final_followup=2,
            interim_looks=[2, 4, 6],
            cutoff_scales=[0.5, 0.8, 0.95],
            gammas=[0, 0.5, 1],
            n_trials=503,
            n_validation=157,
            arrival=arrival,
            rng=44,
            objective=objective,
        )
        key = (
            (lambda c: (c[0], c[1], c[2]))
            if objective == "expected_sample_size"
            else (lambda c: (c[1], c[0], c[2]))
        )
        best = min(candidates, key=key)
        fit = result.best
        assert (fit.calibration_design.max_subjects, fit.cutoff_scale, fit.gamma) == (
            best[1],
            best[3],
            best[4],
        )
        assert_array_equal(result.simulation_seeds, seeds)
        assert_array_equal(result.feasible_sample_sizes, sorted({c[1] for c in candidates}))
        for fit in result.feasible_designs:
            per_n = min(
                [c for c in candidates if c[1] == fit.calibration_design.max_subjects],
                key=lambda c: (c[0], c[2]),
            )
            assert (fit.cutoff_scale, fit.gamma) == per_n[3:]
            assert fit.calibration_oc.expected_sample_size[0] == per_n[0]


def test_validation_size_and_analysis_prior_do_not_change_search():
    options = dict(
        minimum_power=0.3,
        type1_error=0.2,
        accrual_rate=2,
        final_followup=2,
        interim_looks=[2, 4, 6],
        cutoff_scales=[0.5, 0.8, 0.95],
        gammas=[0, 0.5, 1],
        n_trials=503,
        rng=44,
    )
    plain = optimize_bop2_survival_sample_size([4, 6, 8], 1, 3, n_validation=157, **options)
    changed = optimize_bop2_survival_sample_size(
        [4, 6, 8], 1, 3, n_validation=211, analysis_prior=[2, 10], **options
    )
    assert_array_equal(plain.simulation_seeds, changed.simulation_seeds)
    assert_array_equal(plain.feasible_sample_sizes, changed.feasible_sample_sizes)
    assert (
        plain.best.calibration_design.max_subjects == changed.best.calibration_design.max_subjects
    )
    for a, b in zip(plain.feasible_designs, changed.feasible_designs):
        assert (a.cutoff_scale, a.gamma) == (b.cutoff_scale, b.gamma)
        assert_array_equal(
            a.calibration_oc.success_probability, b.calibration_oc.success_probability
        )
    with pytest.raises(BOP2InfeasibleError, match="no sample size"):
        optimize_bop2_survival_sample_size(
            [1, 2],
            1,
            1.01,
            minimum_power=1,
            type1_error=0.001,
            accrual_rate=2,
            final_followup=2,
            interim_looks=[],
            cutoff_scales=[0.99],
            gammas=[0],
            n_trials=1000,
            n_validation=100,
            rng=1,
        )
    with pytest.raises(ValueError, match="gammas") as error:
        optimize_bop2_survival_sample_size(
            [4, 6], 1, 3, minimum_power=0.3, accrual_rate=2, final_followup=2, gammas=[2]
        )
    assert not isinstance(error.value, BOP2InfeasibleError)
