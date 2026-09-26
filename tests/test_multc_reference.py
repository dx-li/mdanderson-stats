"""Published Multc examples and independent base-R numerical references."""

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.multc_core import multc_lean_design


def _rows(name):
    with (Path(__file__).parent / "fixtures" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_shifted_beta_probabilities_against_independent_r_integrals():
    for row in _rows("multc-beta-tails.csv"):
        a, b, ha, hb, delta = (float(row[k]) for k in ("a", "b", "ha", "hb", "delta"))
        n, y = int(row["n"]), int(row["y"])
        nmax = max(3, n)
        design = multc_lean_design(
            nmax,
            (a, b),
            (a, b),
            historical_response=(ha, hb),
            historical_toxicity=(ha, hb),
            response_margin=delta,
            toxicity_margin=delta,
            cohort_size=nmax,
            pretrial_check=False,
        )
        state = design.monitor(y, y, n)
        assert float(state.response_probability) == pytest.approx(float(row["below"]), abs=2e-9)
        assert float(state.toxicity_probability) == pytest.approx(float(row["above"]), abs=2e-9)


def test_official_multc_lean_tutorial_full_and_potential_boundaries():
    design = multc_lean_design(
        30,
        (0.6, 1.4),
        (0.5, 1.5),
        historical_response=(30, 70),
        historical_toxicity=(20, 60),
    )
    bounds = design.stopping_bounds()
    # Full-boundary tables in tutorial pp25-28; cap30 is reported separately.
    expected_r = [-1] * 5 + [0] * 6 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 3
    expected_t = [2, 3] + [3] * 2 + [4] * 2 + [5] * 2 + [6] * 3 + [7] * 3 + [8] * 3
    expected_t += [9] * 2 + [10] * 3 + [11] * 3 + [12] * 3 + [13]
    np.testing.assert_array_equal(bounds.response_stop_max[:-1], expected_r)
    np.testing.assert_array_equal(bounds.toxicity_stop_min[:-1], expected_t)
    potential = design.potential_boundaries()
    rpoints = [
        (lo, int(n))
        for n, (lo, hi) in zip(potential.looks, potential.response_stop)
        if lo <= hi and n < 30
    ]
    tpoints = [
        (lo, int(n))
        for n, (lo, hi) in zip(potential.looks, potential.toxicity_stop)
        if lo <= hi and n < 30
    ]
    assert rpoints == [(0, 6), (1, 12), (2, 17), (3, 22), (4, 27)]
    assert tpoints == [
        (3, 3),
        (3, 4),
        (4, 6),
        (5, 8),
        (6, 10),
        (6, 11),
        (7, 13),
        (7, 14),
        (8, 16),
        (8, 17),
        (9, 19),
        (10, 21),
        (10, 22),
        (11, 24),
        (11, 25),
        (12, 27),
        (12, 28),
    ]


def test_multc99_phaseiia_fixed_historical_rate_example():
    design = multc_lean_design(
        15,
        (1, 1),
        (1, 1),
        historical_response=0.5,
        historical_toxicity=0.5,
    )
    rows = _rows("multc99-phaseiia-bounds.csv")
    bounds = design.stopping_bounds()
    np.testing.assert_array_equal(bounds.response_stop_max, [int(r["response"]) for r in rows])
    np.testing.assert_array_equal(bounds.toxicity_stop_min, [int(r["toxicity"]) for r in rows])
    # Published table 1: n4,y0 and n7,y1. Native efficacy cutoff .05 is
    # the complement of Lean's .95 bad-response probability threshold.
    state = design.monitor([0, 1], [0, 0], [4, 7])
    np.testing.assert_allclose(state.response_probability, [0.96875, 0.96484375], atol=1e-14)
    assert design.monitor(15, 0, 15).decision.item() == "cap_complete"


def test_joint_operating_characteristics_against_full_path_enumeration():
    rows = _rows("multc-path-enumeration.csv")
    moments = _rows("multc-path-moments.csv")
    for cohort in (1, 2):
        design = multc_lean_design(
            6,
            (1, 1),
            (1, 1),
            historical_response=0.45,
            historical_toxicity=0.3,
            response_cutoff=0.8,
            toxicity_cutoff=0.85,
            min_subjects=2,
            cohort_size=cohort,
            pretrial_check=False,
        )
        means = []
        for scenario, probabilities in enumerate(([0.1, 0.45, 0.15, 0.3], [0.25, 0.3, 0, 0.45]), 1):
            oc = design.operating_characteristics(probabilities)
            selected = [
                r for r in rows if int(r["cohort"]) == cohort and int(r["scenario"]) == scenario
            ]
            expected = np.array([float(r["probability"]) for r in selected]).reshape(7, 4)
            np.testing.assert_allclose(oc.sample_size_probability, expected.sum(axis=1), atol=3e-13)
            np.testing.assert_allclose(
                [oc.stop_response_only, oc.stop_toxicity_only, oc.stop_both, oc.cap_completion],
                expected.sum(axis=0),
                atol=3e-13,
            )
            moment = next(
                r for r in moments if int(r["cohort"]) == cohort and int(r["scenario"]) == scenario
            )
            np.testing.assert_allclose(
                [
                    oc.expected_sample_size,
                    oc.sample_size_sd,
                    oc.expected_responses,
                    oc.expected_toxicities,
                ],
                [float(moment[k]) for k in ("expected_n", "sd_n", "responses", "toxicities")],
                atol=3e-13,
            )
            means.append(float(oc.expected_sample_size))
        # Equal marginal event rates do not imply equal joint stopping behavior.
        assert abs(means[0] - means[1]) > 0.1


def test_support_endpoints_underflow_and_strict_prior_ties():
    design = multc_lean_design(
        6,
        (100, 1),
        (1, 1),
        historical_response=1e-8,
        historical_toxicity=(1, 1),
        response_cutoff=0,
        toxicity_cutoff=1,
        pretrial_check=False,
    )
    # The probability is positive mathematically, despite double-precision underflow.
    state = design.monitor(0, 0, 1)
    assert float(state.response_probability) == 0
    assert state.decision.item() == "stop_response"
    trace = design.monitor_outcomes([[0, 0], [1, 0], [1, 0], [1, 0]])
    np.testing.assert_array_equal(trace.sample_size, [0, 1])

    impossible = multc_lean_design(
        6,
        (1, 1),
        (1, 1),
        historical_response=0,
        historical_toxicity=1,
        response_cutoff=0,
        toxicity_cutoff=0,
    )
    assert impossible.monitor(0, 6, 6).decision.item() == "cap_complete"
    oc = impossible.operating_characteristics([0, 0, 1, 0])
    assert float(oc.expected_sample_size) == 6
    assert float(oc.sample_size_sd) == 0
    assert float(oc.cap_completion) == 1

    symmetric = multc_lean_design(
        3,
        (1, 1),
        (1, 1),
        historical_response=(1, 1),
        historical_toxicity=(1, 1),
        response_cutoff=0.5,
        toxicity_cutoff=0.5,
    )
    assert symmetric.monitor(0, 0, 0).decision.item() == "continue"


def test_pretrial_rejection_and_bounded_allocations():
    kwargs = dict(
        response_prior=(1, 1),
        toxicity_prior=(1, 1),
        historical_response=0.5,
        historical_toxicity=0.5,
    )
    rejected = multc_lean_design(6, response_cutoff=0.4, **kwargs)
    oc = rejected.operating_characteristics([0.1, 0.45, 0.15, 0.3])
    assert oc.sample_size_probability[0] == 1
    assert float(oc.expected_sample_size) == float(oc.sample_size_sd) == 0
    assert all(lo > hi for lo, hi in rejected.potential_boundaries().reachable_response)
    # These are cheap views. Rejection must happen before dtype conversion/copies.
    oversized = np.broadcast_to(np.uint8(0), (100_001,))
    with pytest.raises(ValueError, match="10,000-snapshot"):
        rejected.monitor(oversized, 0, 0)
    with pytest.raises(ValueError, match="four entries"):
        rejected.operating_characteristics(oversized)
    with pytest.raises(ValueError, match="must be scalar"):
        rejected.operating_characteristics_independent(oversized, 0.5)
    with pytest.raises(ValueError, match="shape"):
        rejected.monitor_outcomes(np.broadcast_to(np.uint8(0), (100_001, 2)))
    large = multc_lean_design(
        1000, cohort_size=1000, response_cutoff=1, toxicity_cutoff=1, **kwargs
    )
    with pytest.raises(ValueError, match="10-million-state"):
        large.operating_characteristics([0.1, 0.45, 0.15, 0.3])
