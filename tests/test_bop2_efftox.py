"""Joint BOP2: exact posterior identities, all paths, independence and three-null calibration."""

from fractions import Fraction
from itertools import product
from math import comb

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import bop2_efftox_design, optimize_bop2_efftox


def beta_sf(a, b, p):
    n = int(a + b - 1)
    return sum(Fraction(comb(n, k)) * p**k * (1 - p) ** (n - k) for k in range(int(a)))


def test_exact_marginal_rules_and_equality_with_distinct_looks():
    design = bop2_efftox_design(
        6,
        [0.25, 0.5],
        cutoff_scales=[0.5, 0.5],
        gamma=0,
        prior=[1, 1, 1, 1],
        efficacy_looks=[2, 4, 6],
        toxicity_looks=[3, 6],
    )
    assert_array_equal(design.looks, [2, 3, 4, 6])
    for x in product(range(3), repeat=4):
        n = sum(x)
        if n > 6:
            continue
        e, t = x[0] + x[1], x[0] + x[2]
        pe = beta_sf(2 + e, 2 + n - e, Fraction(1, 4))
        pt = 1 - beta_sf(2 + t, 2 + n - t, Fraction(1, 2))
        futile = n in [2, 4, 6] and pe <= Fraction(1, 2)
        unsafe = n in [3, 6] and pt <= Fraction(1, 2)
        expected = "continue"
        if n == 6:
            expected = "final_negative" if futile or unsafe else "final_positive"
        elif futile and unsafe:
            expected = "stop_futility_toxicity"
        elif futile:
            expected = "stop_futility"
        elif unsafe:
            expected = "stop_toxicity"
        state = design.monitor(x)
        assert state.decision == expected
        assert_allclose(state.marginal_success_probability, [float(pe), float(pt)], atol=2e-15)
    for equality_continues in [False, True]:
        equal = bop2_efftox_design(
            2,
            [0.5, 0.5],
            cutoff_scales=[0.5, 0.5],
            gamma=0,
            prior=[1, 1, 1, 1],
            efficacy_looks=[2],
            toxicity_looks=[2],
            equality_continues=equality_continues,
        )
        assert equal.monitor([0, 1, 1, 0]).decision == (
            "final_positive" if equality_continues else "final_negative"
        )


def test_correlated_oc_and_replay_against_every_path():
    design = bop2_efftox_design(
        6,
        [0.3, 0.5],
        cutoff_scales=[0.8, 0.7],
        gamma=0.6,
        efficacy_looks=[2, 6],
        toxicity_looks=[3, 6],
    )
    p = np.array([[0.2, 0.1, 0.3, 0.4], [0.15, 0.15, 0.35, 0.35], [0, 1, 0, 0], [1, 0, 0, 0]])
    paths = np.array(list(product(range(4), repeat=6)))
    events = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])[paths].cumsum(axis=1)
    weight = p[:, paths].prod(axis=-1)
    stopped = np.zeros(paths.shape[0], dtype=bool)
    terminal = np.full(paths.shape[0], 6)
    mass = np.zeros((len(p), len(design.looks)))
    for j, n in enumerate(design.looks):
        bad = (
            (events[:, n - 1, 0] <= design.futility_max[j, 0])
            | (events[:, n - 1, 1] >= design.toxicity_min[j])
        ) & ~stopped
        mass[:, j] = weight[:, bad].sum(axis=-1)
        terminal[bad] = n
        stopped |= bad
    oc = design.operating_characteristics(p)
    assert_allclose(oc.stop_probability, mass, atol=2e-14)
    assert_allclose(oc.success_probability, weight[:, ~stopped].sum(axis=-1), atol=2e-14)
    expected = weight @ terminal
    assert_allclose(oc.expected_sample_size, expected, atol=2e-14)
    assert_allclose(
        oc.sample_size_sd**2,
        np.sum(weight * (terminal - expected[:, None]) ** 2, axis=-1),
        atol=2e-14,
    )
    assert_allclose(oc.sample_size_probability.sum(axis=-1), 1, atol=2e-14)
    assert_array_equal(design.monitor_outcomes(paths).decision[:, -1] == "final_positive", ~stopped)


def test_independent_joint_error_identity_and_marginal_factorization():
    design = bop2_efftox_design(
        24,
        [0.3, 0.4],
        cutoff_scales=[0.8, 0.85],
        gamma=0.6,
        efficacy_looks=[12, 24],
        toxicity_looks=[6, 12, 18, 24],
    )
    e = np.array([0.3, 0.3, 0.6, 0.6])
    t = np.array([0.4, 0.2, 0.4, 0.2])
    p = np.stack([e * t, e * (1 - t), (1 - e) * t, (1 - e) * (1 - t)], axis=-1)
    success = design.operating_characteristics(p).success_probability
    first = design.marginals[0].operating_characteristics(e).positive_conclusion
    second = design.marginals[1].operating_characteristics(t).complete_negative
    assert_allclose(success, first * second, rtol=2e-14)
    assert_allclose(success[0] * success[3], success[1] * success[2], rtol=2e-14)


def test_three_null_grid_optimum_and_separate_analysis_prior():
    settings = dict(
        efficacy_looks=[4, 8, 12],
        toxicity_looks=[3, 6, 9, 12],
        efficacy_scales=[0.7, 0.9, 0.99],
        toxicity_scales=[0.7, 0.9, 0.99],
        gammas=[0, 0.5, 1],
        type1_error=[0.05, 0.15, 0.15],
    )
    result = optimize_bop2_efftox(12, [0.3, 0.5], [0.7, 0.1], **settings)
    candidates = []
    p = result.calibration_oc.category_probability
    for se, st, g in product(
        settings["efficacy_scales"], settings["toxicity_scales"], settings["gammas"]
    ):
        design = bop2_efftox_design(
            12,
            [0.3, 0.5],
            cutoff_scales=[se, st],
            gamma=g,
            efficacy_looks=settings["efficacy_looks"],
            toxicity_looks=settings["toxicity_looks"],
        )
        oc = design.operating_characteristics(p)
        if np.all(oc.success_probability[:3] <= settings["type1_error"]):
            candidates.append((-oc.success_probability[3], oc.expected_sample_size[0], se, st, g))
    best = min(candidates)
    assert (*result.cutoff_scales, result.gamma) == best[2:]
    changed = optimize_bop2_efftox(
        12, [0.3, 0.5], [0.7, 0.1], analysis_prior=[1, 50, 1, 1], **settings
    )
    assert_array_equal(changed.cutoff_scales, result.cutoff_scales)
    assert changed.gamma == result.gamma
    assert np.any(changed.analysis_oc.success_probability[:3] > settings["type1_error"])
