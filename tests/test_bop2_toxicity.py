"""Toxicity orientation checked with exact beta identities and complete trial paths."""

from fractions import Fraction
from itertools import product
from math import comb

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import bop2_binary_design, optimize_bop2_binary


def test_toxicity_boundaries_against_rational_lower_tail_and_equality():
    design = bop2_binary_design(
        12,
        0.25,
        endpoint="toxicity",
        cutoff_scale=0.75,
        gamma=0,
        looks=[3, 6, 9, 12],
        prior=[1, 1],
    )
    for n, bound in zip(design.looks, design.positive_min):
        go = [
            sum(
                Fraction(comb(int(n) + 1, k) * 3 ** (int(n) + 1 - k), 4 ** (int(n) + 1))
                for k in range(y + 1, int(n) + 2)
            )
            for y in range(int(n) + 1)
        ]
        assert bound == sum(p >= Fraction(3, 4) for p in go)
    equal = bop2_binary_design(
        1, 0.5, endpoint="toxicity", cutoff_scale=0.25, gamma=0, looks=[1], prior=[1, 1]
    )
    assert equal.monitor(1, 1).decision == "final_negative"  # acceptable at equality
    above = bop2_binary_design(
        1,
        0.5,
        endpoint="toxicity",
        cutoff_scale=np.nextafter(0.25, 1),
        gamma=0,
        looks=[1],
        prior=[1, 1],
    )
    assert above.monitor(1, 1).decision == "final_positive"  # unsafe
    tiny = bop2_binary_design(
        2, 1e-100, endpoint="toxicity", cutoff_scale=0.5, gamma=0, looks=[1, 2]
    )
    assert np.isfinite(tiny.low_probability[np.tril_indices(3)]).all()
    assert tiny.monitor_outcomes([1, 0]).decision.tolist() == ["stop_toxicity", "stop_toxicity"]


def test_toxicity_oc_against_all_paths_including_event_and_sample_size_moments():
    design = bop2_binary_design(
        6,
        0.5,
        endpoint="toxicity",
        cutoff_scale=0.8,
        gamma=0.5,
        looks=[2, 4, 6],
        prior=[1, 1],
    )
    rates = np.array([0, 0.2, 0.5, 1])
    oc = design.operating_characteristics(rates)
    for index, p in enumerate(rates):
        safe = expected_n = expected_y = 0.0
        pmf = np.zeros(3)
        for path in product((0, 1), repeat=6):
            weight = p ** sum(path) * (1 - p) ** (6 - sum(path))
            for j, (look, bound) in enumerate(zip(design.looks, design.positive_min)):
                events = sum(path[:look])
                if events >= bound or look == 6:
                    break
            safe += weight * (look == 6 and events < bound)
            expected_n += look * weight
            expected_y += events * weight
            pmf[j] += weight
        assert_allclose(oc.complete_negative[index], safe, atol=1e-14)
        assert_allclose(oc.positive_conclusion[index], 1 - safe, atol=1e-14)
        assert_allclose(oc.expected_sample_size[index], expected_n, atol=1e-14)
        assert_allclose(oc.expected_events[index], expected_y, atol=1e-14)
        assert_allclose(oc.sample_size_probability[index], pmf, atol=1e-14)


def test_toxicity_grid_matches_reflected_efficacy_and_separates_analysis_prior():
    settings = dict(looks=[5, 10, 20], cutoff_scales=[0.7, 0.85, 0.95, 0.99], gammas=[0, 0.5, 1])
    tox = optimize_bop2_binary(20, 0.75, 0.5, endpoint="toxicity", **settings)
    eff = optimize_bop2_binary(20, 0.25, 0.5, **settings)
    assert (tox.cutoff_scale, tox.gamma) == (eff.cutoff_scale, eff.gamma)
    assert_allclose(tox.calibration_success_probability, eff.calibration_success_probability)
    assert_allclose(
        tox.calibration_oc.expected_sample_size, eff.calibration_oc.expected_sample_size
    )
    assert_array_equal(
        tox.calibration_design.positive_min,
        eff.calibration_design.looks - eff.calibration_design.futility_max,
    )
    assert tox.calibration_success_probability[0] <= 0.1
    informative = optimize_bop2_binary(
        20, 0.75, 0.5, endpoint="toxicity", analysis_prior=[1, 20], **settings
    )
    assert (informative.cutoff_scale, informative.gamma) == (tox.cutoff_scale, tox.gamma)
    assert informative.analysis_success_probability[0] > 0.1
    with pytest.raises(ValueError, match="alternative<null"):
        optimize_bop2_binary(20, 0.25, 0.5, endpoint="toxicity", **settings)
