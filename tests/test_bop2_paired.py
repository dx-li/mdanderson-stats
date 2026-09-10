"""Dirichlet marginal identities, categorical path enumeration and grid optima."""

from fractions import Fraction
from itertools import product
from math import comb

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import bop2_paired_design, optimize_bop2_paired


def beta_sf(a, b, p):
    n = int(a + b - 1)
    return sum(Fraction(comb(n, k)) * p**k * (1 - p) ** (n - k) for k in range(int(a)))


@pytest.mark.parametrize("endpoint", ["ordinal", "multiple"])
def test_dirichlet_marginals_and_both_futility_rule(endpoint):
    ordinal = endpoint == "ordinal"
    increments = np.array([(1, 1), (0, 1), (0, 0)] if ordinal else [(1, 1), (1, 0), (0, 1), (0, 0)])
    prior = np.ones(len(increments))
    design = bop2_paired_design(
        6,
        [0.25, 0.5],
        endpoint=endpoint,
        null_joint_rate=None if ordinal else 0.125,
        prior=prior,
        cutoff_scale=0.75,
        gamma=0,
        looks=[2, 4, 6],
    )
    for x in product(range(3), repeat=len(increments)):
        n = sum(x)
        if n > 6:
            continue
        state = design.monitor(x)
        posterior = prior + x
        a = posterior @ increments
        b = posterior.sum() - a
        exact = [beta_sf(a[j], b[j], p) for j, p in enumerate([Fraction(1, 4), Fraction(1, 2)])]
        assert_allclose(state.marginal_success_probability, [float(p) for p in exact], atol=2e-15)
        bad = all(p < Fraction(3, 4) for p in exact)
        expected = "continue"
        if n in [2, 4]:
            expected = "stop_futility" if bad else "continue"
        elif n == 6:
            expected = "final_negative" if bad else "final_positive"
        assert state.decision == expected
    equal = bop2_paired_design(
        1,
        [0.5, 0.5],
        endpoint=endpoint,
        null_joint_rate=None if ordinal else 0.25,
        prior=prior,
        cutoff_scale=0.5 if ordinal else 0.3125,
        gamma=0,
        looks=[1],
    )
    counts = np.zeros(len(increments), dtype=int)
    counts[-1] = 1
    assert equal.monitor(counts).decision == "final_positive"
    above = bop2_paired_design(
        1,
        [0.5, 0.5],
        endpoint=endpoint,
        null_joint_rate=None if ordinal else 0.25,
        prior=prior,
        cutoff_scale=np.nextafter(0.5 if ordinal else 0.3125, 1),
        gamma=0,
        looks=[1],
    )
    assert above.monitor(counts).decision == "final_negative"


@pytest.mark.parametrize("endpoint", ["ordinal", "multiple"])
def test_exact_correlated_oc_against_every_categorical_path(endpoint):
    ordinal = endpoint == "ordinal"
    increments = np.array([(1, 1), (0, 1), (0, 0)] if ordinal else [(1, 1), (1, 0), (0, 1), (0, 0)])
    design = bop2_paired_design(
        6,
        [0.25, 0.5],
        endpoint=endpoint,
        null_joint_rate=None if ordinal else 0.125,
        cutoff_scale=0.85,
        gamma=0.5,
        looks=[2, 4, 6],
    )
    probabilities = (
        np.array([[0.2, 0.3, 0.5], [1, 0, 0], [0, 0, 1]])
        if ordinal
        else np.array(
            [[0.2, 0.1, 0.1, 0.6], [0.09, 0.21, 0.21, 0.49], [0, 0.5, 0.5, 0], [0, 0, 0, 1]]
        )
    )
    oc = design.operating_characteristics(probabilities)
    paths = np.array(list(product(range(len(increments)), repeat=6)))
    outcomes = increments[paths].cumsum(axis=1)
    weights = np.prod(probabilities[:, paths], axis=-1)
    terminal = np.full(paths.shape[0], 6)
    stopped = np.zeros(paths.shape[0], dtype=bool)
    stop_mass = np.zeros((len(probabilities), 3))
    for j, (look, bounds) in enumerate(zip(design.looks, design.futility_max)):
        bad = (outcomes[:, look - 1] <= bounds).all(axis=-1) & ~stopped
        stop_mass[:, j] = weights[:, bad].sum(axis=-1)
        terminal[bad] = look
        stopped |= bad
    assert_allclose(oc.stop_probability, stop_mass, atol=2e-14)
    assert_allclose(oc.success_probability, weights[:, ~stopped].sum(axis=-1), atol=2e-14)
    assert_allclose(oc.expected_sample_size, weights @ terminal, atol=2e-14)
    assert_allclose(oc.sample_size_probability.sum(axis=-1), 1, atol=2e-14)
    replay = design.monitor_outcomes(paths)
    assert_array_equal(replay.decision[:, -1] == "final_positive", ~stopped)
    if not ordinal:
        assert abs(oc.success_probability[0] - oc.success_probability[1]) > 0.001
    with pytest.raises(ValueError, match="sum to one"):
        design.operating_characteristics(np.ones(len(increments)))


def test_paired_grid_optimum_and_informative_prior():
    settings = dict(looks=[4, 8, 12], cutoff_scales=[0.7, 0.85, 0.95, 0.99], gammas=[0, 0.5, 1])
    result = optimize_bop2_paired(12, [0.15, 0.3], [0.3, 0.6], **settings)
    candidates = []
    for scale, gamma in product(settings["cutoff_scales"], settings["gammas"]):
        design = bop2_paired_design(
            12, [0.15, 0.3], cutoff_scale=scale, gamma=gamma, looks=settings["looks"]
        )
        oc = design.operating_characteristics([[0.15, 0.15, 0.7], [0.3, 0.3, 0.4]])
        if oc.success_probability[0] <= 0.1:
            candidates.append(
                (-oc.success_probability[1], oc.expected_sample_size[0], scale, gamma)
            )
    assert (result.cutoff_scale, result.gamma) == min(candidates)[2:]
    changed = optimize_bop2_paired(
        12, [0.15, 0.3], [0.3, 0.6], analysis_prior=[20, 1, 1], **settings
    )
    assert (changed.cutoff_scale, changed.gamma) == (result.cutoff_scale, result.gamma)
    assert changed.analysis_oc.success_probability[0] > 0.1
    assert_array_equal(
        changed.calibration_design.futility_max, result.calibration_design.futility_max
    )
