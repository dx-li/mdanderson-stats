"""Small exact checks for the Multc Lean marginal design core."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.multc_core import multc_lean_design


def test_multc_monitor_bounds_and_exact_joint_operating_characteristics():
    design = multc_lean_design(
        4,
        (1, 1),
        (1, 1),
        historical_response=0.5,
        historical_toxicity=0.25,
        response_cutoff=0.5,
        toxicity_cutoff=0.5,
        min_subjects=1,
        cohort_size=2,
        pretrial_check=False,
    )
    assert_array_equal(design.looks, [2, 4])
    assert_array_equal(design.stopping_bounds().toxicity_stop_min, [1, 1])
    assert design.monitor(2, 0, 2).decision.item() == "continue"
    assert design.monitor(0, 2, 2).decision.item() == "stop_both"
    assert design.monitor(1, 0, 2).decision.item() == "continue"  # equality continues
    assert design.monitor_outcomes([[1, 0], [0, 1]]).sample_size[-1] == 2

    probabilities = [0.2, 0.3, 0.1, 0.4]
    oc = design.operating_characteristics(probabilities)
    assert_allclose(oc.sample_size_probability.sum(), 1, atol=1e-12)
    assert_allclose(
        oc.stop_response_only + oc.stop_toxicity_only + oc.stop_both + oc.cap_completion,
        1,
        atol=1e-12,
    )
    assert_allclose(oc.expected_responses, 0.5 * oc.expected_sample_size, atol=1e-12)
    assert_allclose(oc.expected_toxicities, 0.3 * oc.expected_sample_size, atol=1e-12)


def test_multc_validates_joint_scenario_and_native_margin_constraint():
    design = multc_lean_design(
        3,
        (1, 1),
        (1, 1),
        historical_response=0.5,
        historical_toxicity=0.5,
        pretrial_check=False,
    )
    try:
        design.operating_characteristics([0.2, 0.2, 0.2, 0.2])
    except ValueError as error:
        assert "summing to one" in str(error)
    else:
        raise AssertionError("invalid scenario probabilities were accepted")

    try:
        multc_lean_design(
            3,
            (1, 1),
            (1, 1),
            historical_response=0.5,
            historical_toxicity=0.5,
            response_margin=-0.1,
            toxicity_margin=0.1,
        )
    except ValueError as error:
        assert "cannot have" in str(error)
    else:
        raise AssertionError("forbidden margin combination was accepted")


def test_multc_pretrial_rejection_truncates_outcomes_and_preflights_broadcast():
    design = multc_lean_design(
        4,
        (1, 1),
        (1, 1),
        historical_response=0.1,
        historical_toxicity=0.5,
        response_cutoff=0.05,
        toxicity_cutoff=0.95,
    )
    trace = design.monitor_outcomes([[1, 0], [1, 0]])
    assert_array_equal(trace.sample_size, [0])
    assert trace.decision[0] == "stop_response"
    assert not trace.decision.flags.writeable

    try:
        design.monitor(np.zeros(10_001), 0, 0)
    except ValueError as error:
        assert "10,000-snapshot" in str(error)
    else:
        raise AssertionError("oversized monitor broadcast was accepted")
