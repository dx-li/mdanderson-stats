"""Independent beta identities, full outcome enumeration and grid calibration."""

from fractions import Fraction
from itertools import product
from math import comb

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import bop2_binary_design, optimize_bop2_binary


def test_boundaries_against_exact_beta_binomial_identity_and_equality():
    design = bop2_binary_design(
        12, 0.25, cutoff_scale=0.75, gamma=0, looks=[3, 6, 9, 12], prior=[1, 1]
    )
    for n, bound in zip(design.looks, design.futility_max):
        go = [
            sum(
                Fraction(comb(int(n) + 1, k) * 3 ** (int(n) + 1 - k), 4 ** (int(n) + 1))
                for k in range(y + 1)
            )
            for y in range(int(n) + 1)
        ]
        assert bound == sum(p < Fraction(3, 4) for p in go) - 1
    with pytest.raises(ArithmeticError, match="cutoff underflows"):
        bop2_binary_design(10, 0.5, cutoff_scale=np.nextafter(0.0, 1.0), gamma=1, looks=[1, 10])
    equal = bop2_binary_design(1, 0.5, cutoff_scale=0.25, gamma=0, looks=[1], prior=[1, 1])
    assert equal.monitor(0, 1).decision == "final_positive"
    above = bop2_binary_design(
        1, 0.5, cutoff_scale=np.nextafter(0.25, 1), gamma=0, looks=[1], prior=[1, 1]
    )
    assert above.monitor(0, 1).decision == "final_negative"


def test_exact_operating_characteristics_against_all_binary_paths():
    design = bop2_binary_design(6, 0.2, cutoff_scale=0.8, gamma=0.5, looks=[2, 4, 6], prior=[1, 1])
    rates = np.array([0, 0.2, 0.6, 1])
    oc = design.operating_characteristics(rates)
    for index, p in enumerate(rates):
        success = expected_n = 0.0
        for path in product((0, 1), repeat=6):
            weight = p ** sum(path) * (1 - p) ** (6 - sum(path))
            last = 6
            for look, bound in zip(design.looks[:-1], design.futility_max[:-1]):
                if sum(path[:look]) <= bound:
                    last = int(look)
                    break
            if last == 6 and sum(path) >= design.final_positive_min:
                success += weight
            expected_n += last * weight
        assert_allclose(oc.positive_conclusion[index], success, atol=1e-14)
        assert_allclose(oc.expected_sample_size[index], expected_n, atol=1e-14)
    assert_allclose(oc.sample_size_probability.sum(axis=-1), 1, atol=1e-14)


def test_grid_optimum_and_separate_informative_analysis_prior():
    settings = dict(looks=[5, 10, 20], cutoff_scales=[0.7, 0.85, 0.95, 0.99], gammas=[0, 0.5, 1])
    result = optimize_bop2_binary(20, 0.2, 0.5, **settings)
    candidates = []
    for scale, gamma in product(settings["cutoff_scales"], settings["gammas"]):
        design = bop2_binary_design(
            20, 0.2, cutoff_scale=scale, gamma=gamma, looks=settings["looks"]
        )
        oc = design.operating_characteristics([0.2, 0.5])
        if oc.positive_conclusion[0] <= 0.1:
            candidates.append(
                (-oc.positive_conclusion[1], oc.expected_sample_size[0], scale, gamma)
            )
    best = min(candidates)
    assert (result.cutoff_scale, result.gamma) == best[2:]
    assert result.calibration_oc.positive_conclusion[0] <= 0.1
    informative = optimize_bop2_binary(20, 0.2, 0.5, analysis_prior=[20, 1], **settings)
    assert (informative.cutoff_scale, informative.gamma) == (result.cutoff_scale, result.gamma)
    assert_array_equal(
        informative.calibration_design.futility_max, result.calibration_design.futility_max
    )
    assert informative.analysis_oc.positive_conclusion[0] > 0.1
