"""Whole-grid checks for correlated paired and joint endpoint sample-size searches."""

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats import (
    BOP2InfeasibleError,
    bop2_efftox_design,
    bop2_paired_design,
    optimize_bop2_efftox_sample_size,
    optimize_bop2_paired_sample_size,
)


def assert_optima(results, candidates):
    for objective, result in results.items():
        key = (
            (lambda c: (c[0], c[1], c[2]))
            if objective == "expected_sample_size"
            else (lambda c: (c[1], c[0], c[2]))
        )
        best = min(candidates, key=key)
        fit = result.best
        assert fit.calibration_design.max_subjects == best[1]
        scales = (fit.cutoff_scale,) if hasattr(fit, "cutoff_scale") else tuple(fit.cutoff_scales)
        assert (*scales, fit.gamma) == best[3:]
        assert_array_equal(result.feasible_sample_sizes, sorted({c[1] for c in candidates}))
        for fit in result.feasible_designs:
            per_n = min(
                [c for c in candidates if c[1] == fit.calibration_design.max_subjects],
                key=lambda c: (c[0], c[2]),
            )
            scales = (
                (fit.cutoff_scale,) if hasattr(fit, "cutoff_scale") else tuple(fit.cutoff_scales)
            )
            assert (*scales, fit.gamma) == per_n[3:]
            assert fit.objective == "expected_sample_size"


@pytest.mark.parametrize("endpoint", ["ordinal", "multiple"])
def test_paired_sample_sizes_against_all_parameters(endpoint):
    sizes, scales, gammas = [8, 12, 16], [0.7, 0.9, 0.99], [0, 0.5, 1]
    settings = dict(
        endpoint=endpoint,
        null_joint_rate=0.05 if endpoint == "multiple" else None,
        alternative_joint_rate=0.4 if endpoint == "multiple" else None,
    )
    p = (
        [[0.1, 0.1, 0.8], [0.5, 0.2, 0.3]]
        if endpoint == "ordinal"
        else [[0.05, 0.05, 0.15, 0.75], [0.4, 0.1, 0.3, 0.2]]
    )
    candidates = []
    for n, s, g in product(sizes, scales, gammas):
        looks = [v for v in [4, 8, 12] if v < n] + [n]
        d = bop2_paired_design(
            n,
            [0.1, 0.2],
            cutoff_scale=s,
            gamma=g,
            looks=looks,
            endpoint=endpoint,
            null_joint_rate=settings["null_joint_rate"],
        )
        oc = d.operating_characteristics(p)
        if oc.success_probability[0] <= 0.1 and oc.success_probability[1] >= 0.7:
            candidates.append(
                (float(oc.expected_sample_size[0]), n, -float(oc.success_probability[1]), s, g)
            )
    results = {
        objective: optimize_bop2_paired_sample_size(
            sizes,
            [0.1, 0.2],
            [0.5, 0.7],
            minimum_power=0.7,
            objective=objective,
            interim_looks=[4, 8, 12],
            cutoff_scales=scales,
            gammas=gammas,
            **settings,
        )
        for objective in ["expected_sample_size", "minimax"]
    }
    assert_optima(results, candidates)


def test_joint_sample_sizes_preserve_three_null_limits_and_different_schedules():
    sizes, scales, gammas = [8, 12, 16], [0.7, 0.9, 0.99], [0, 0.5, 1]
    alpha = [0.05, 0.15, 0.15]
    joint = [0.18, 0.06, 0.37, 0.08]
    rates = np.array([[0.3, 0.5], [0.3, 0.1], [0.7, 0.5], [0.7, 0.1]])
    p = np.array([[j, e - j, t - j, 1 - e - t + j] for (e, t), j in zip(rates, joint)])
    candidates = []
    for n, se, st, g in product(sizes, scales, scales, gammas):
        e_looks = [v for v in [4, 8, 12] if v < n] + [n]
        t_looks = [v for v in [3, 6, 9, 12] if v < n] + [n]
        d = bop2_efftox_design(
            n,
            [0.3, 0.5],
            cutoff_scales=[se, st],
            gamma=g,
            null_joint_rate=joint[0],
            efficacy_looks=e_looks,
            toxicity_looks=t_looks,
        )
        oc = d.operating_characteristics(p)
        if np.all(oc.success_probability[:3] <= alpha) and oc.success_probability[3] >= 0.5:
            candidates.append(
                (float(oc.expected_sample_size[0]), n, -float(oc.success_probability[3]), se, st, g)
            )
    settings = dict(
        type1_error=alpha,
        joint_rates=joint,
        efficacy_interim_looks=[4, 8, 12],
        toxicity_interim_looks=[3, 6, 9, 12],
        efficacy_scales=scales,
        toxicity_scales=scales,
        gammas=gammas,
    )
    results = {
        objective: optimize_bop2_efftox_sample_size(
            sizes, [0.3, 0.5], [0.7, 0.1], minimum_power=0.5, objective=objective, **settings
        )
        for objective in ["expected_sample_size", "minimax"]
    }
    assert_optima(results, candidates)
    changed = optimize_bop2_efftox_sample_size(
        sizes, [0.3, 0.5], [0.7, 0.1], minimum_power=0.5, analysis_prior=[1, 50, 1, 1], **settings
    )
    plain = results["expected_sample_size"].best
    assert changed.best.calibration_design.max_subjects == plain.calibration_design.max_subjects
    assert_array_equal(changed.best.cutoff_scales, plain.cutoff_scales)
    assert np.any(changed.best.analysis_oc.success_probability[:3] > alpha)


def test_unattainable_power_is_not_confused_with_invalid_input():
    with pytest.raises(BOP2InfeasibleError, match="no sample size"):
        optimize_bop2_paired_sample_size(
            [1, 2],
            [0.2, 0.4],
            [0.3, 0.5],
            minimum_power=0.99,
            interim_looks=[],
            cutoff_scales=[0.99],
            gammas=[0],
        )
    with pytest.raises(ValueError, match="joint_rates") as error:
        optimize_bop2_efftox_sample_size(
            [12], [0.3, 0.5], [0.7, 0.1], minimum_power=0.5, joint_rates=[0.1]
        )
    assert not isinstance(error.value, BOP2InfeasibleError)
