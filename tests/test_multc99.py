import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.multc99 import Multc99Event, multc99_design
from mdanderson_stats.multc99_trial import run_multc99_trial, simulate_multc99

_FIXTURES = Path(__file__).parent / "fixtures"
_NATIVE = json.loads((_FIXTURES / "multc99-native-reference.json").read_text())
_PRECISE = json.loads((_FIXTURES / "multc99-high-precision-reference.json").read_text())


@pytest.mark.parametrize("row", _PRECISE["cases"])
def test_probabilities_against_independent_45_digit_historical_density_integrals(row):
    design = multc99_design(
        [row["aE"], row["bE"]],
        [row["aS"], row["bS"]],
        [Multc99Event("event", (1, 0))],
        max_subjects=13,
    )
    result = design.event_probability(
        "event", [row["x"], row["n"] - row["x"]], margin=row["margin"]
    )
    assert abs(result.probability - row["probability"]) <= 3e-9
    assert result.absolute_error <= 1e-9


@pytest.mark.parametrize("row", _NATIVE["boundaries"])
def test_native_single_mixture_conditional_and_minimum_enrollment_boundaries(row):
    conditional = bool(row["conditional"])
    experimental = [2, 3, 1] if conditional else [2, 3]
    historical = [[7, 13], [6, 4]] if row["mixture"] else [[7, 13]]
    if conditional:
        historical = [h + [1] for h in historical]
    event = Multc99Event(
        "event",
        (1, 0, 0) if conditional else (1, 0),
        conditioning_definition=(1, 1, 0) if conditional else None,
        lower_margin=0.05,
        lower_cutoff=0.1,
        upper_margin=0.1,
        upper_cutoff=0.9,
    )
    design = multc99_design(
        experimental,
        historical,
        [event],
        max_subjects=20,
        min_subjects=row["nmin"],
        historical_weights=[0.3, 0.7] if row["mixture"] else None,
    )
    # Normalize unreachable sentinel counts; the native values are +/-99.
    np.testing.assert_array_equal(
        np.maximum(design.lower_bounds[0], -1), np.maximum(row["lower"], -1)
    )
    np.testing.assert_array_equal(
        np.minimum(design.upper_bounds[0], np.arange(21) + 1),
        np.minimum(row["upper"], np.arange(21) + 1),
    )


def _simple_design():
    return multc99_design(
        [1, 1],
        [1, 1],
        [Multc99Event("success", (1, 0), lower_cutoff=0.3)],
        max_subjects=5,
    )


def test_exact_potential_tape_enumeration_and_cap_outcome_retention():
    design = _simple_design()
    trial_sizes = []
    for tape in itertools.product([0, 1], repeat=5):
        result = run_multc99_trial(design, tape)
        trial_sizes.append(result.sample_size)
        expected_size = 2 if tape[:2] == (1, 1) else 5
        assert result.sample_size == expected_size
        np.testing.assert_array_equal(
            result.elementary_counts, np.bincount(tape[:expected_size], minlength=2)
        )
    assert trial_sizes.count(2) == 8
    assert trial_sizes.count(5) == 24
    assert not design.lower_bounds.flags.writeable
    with pytest.raises(ValueError):
        design.lower_bounds.setflags(write=True)


def test_conditional_event_uses_parent_count_and_excludes_outside_data():
    event = Multc99Event(
        "conditional", (1, 0, 0), conditioning_definition=(1, 1, 0), lower_cutoff=0.3
    )
    design = multc99_design([1, 1, 5], [1, 1, 3], [event], max_subjects=20)
    a = design.event_probability("conditional", [2, 3, 0])
    b = design.event_probability("conditional", [2, 3, 10])
    assert a == b
    assert a.event_count == 2
    assert a.conditioning_count == 5
    assert design.monitor_counts([0, 0, 10]).decision == "continue"
    assert design.monitor_counts([0, 2, 10]).decision == "stop"


def test_inclusive_upper_strict_lower_ties_and_conflicting_rules_are_explicit():
    lower = Multc99Event("lower", (1, 0), lower_cutoff=0.5)
    upper = Multc99Event("upper", (1, 0), upper_cutoff=0.5)
    design = multc99_design([1, 1], [1, 1], [lower, upper], max_subjects=5)
    state = design.monitor_counts([1, 1])
    np.testing.assert_array_equal(state.lower_hits, [False, False])
    np.testing.assert_array_equal(state.upper_hits, [False, True])
    both = multc99_design(
        [1, 1],
        [1, 1],
        [Multc99Event("both", (1, 0), lower_cutoff=0.9, upper_cutoff=0.1)],
        max_subjects=5,
    ).monitor_counts([1, 1])
    assert both.lower_hits[0] and both.upper_hits[0]
    assert design.monitor_counts([0, 0]).decision == "continue"
    assert design.monitor_counts([2, 3]).decision == "cap_complete"


def test_cutoff_endpoints_and_unattainable_bounds_terminate_within_cap():
    for cutoff in (0, 1):
        d = multc99_design(
            [1, 1],
            [1, 1],
            [Multc99Event("e", (1, 0), lower_cutoff=cutoff, upper_cutoff=cutoff)],
            max_subjects=5,
        )
        if cutoff == 0:
            assert d.lower_bounds[0, 2] == -1
            assert d.upper_bounds[0, 2] == 0
        else:
            assert d.lower_bounds[0, 2] == 2
            assert d.upper_bounds[0, 2] == 3


def test_simulation_agrees_with_exact_two_failure_stop_probability_and_replays():
    design = _simple_design()
    a = simulate_multc99(design, [0.4, 0.6], trials=2000, seed=123)
    b = simulate_multc99(design, [0.4, 0.6], trials=2000, seed=123)
    np.testing.assert_array_equal(a.sample_sizes, b.sample_sizes)
    np.testing.assert_array_equal(a.elementary_counts, b.elementary_counts)
    assert abs(a.early_stop_probability - 0.36) < 4 * a.early_stop_mcse
    assert abs(a.mean_sample_size - 3.92) < 4 * a.sample_size_mcse
    np.testing.assert_array_equal(a.elementary_counts.sum(axis=1), a.sample_sizes)
    assert sum(n for _, n in a.hit_patterns) == a.trials
    assert a.lower_hits.dtype == bool
    assert not a.elementary_counts.flags.writeable


def test_explicit_looks_and_period_count_monitoring_are_distinct_from_cohorts():
    design = _simple_design()
    assert run_multc99_trial(design, [1] * 5, look_sizes=[]).sample_size == 5
    assert run_multc99_trial(design, [1] * 5, look_sizes=[3]).sample_size == 3
    for window in (0, 1, 2, 2.3):
        a = simulate_multc99(
            design,
            [0, 1],
            trials=10,
            seed=3,
            monitoring_period=1,
            accrual_rate=0.2,
            response_window=window,
        )
        b = simulate_multc99(
            design,
            [0, 1],
            trials=10,
            seed=3,
            monitoring_period=10,
            accrual_rate=0.02,
            response_window=10 * window,
        )
        np.testing.assert_array_equal(a.sample_sizes, b.sample_sizes)


@pytest.mark.parametrize(
    "arguments",
    [
        dict(experimental_prior=[0, 1]),
        dict(historical_prior=[np.inf, 1]),
        dict(historical_prior=[[1, 1], [2, 3]]),
        dict(historical_weights=[0.5]),
        dict(max_subjects=501),
        dict(min_subjects=5),
        dict(cohort_size=True),
    ],
)
def test_invalid_design_inputs(arguments):
    base = dict(
        experimental_prior=[1, 1],
        historical_prior=[1, 1],
        events=[Multc99Event("e", (1, 0))],
        max_subjects=5,
    )
    base.update(arguments)
    with pytest.raises(ValueError):
        multc99_design(**base)


def test_invalid_masks_counts_tapes_and_monitoring_configuration():
    for definition in ((1, 1), (0, 0), (1, 2), (1.0, 0.0)):
        with pytest.raises(ValueError):
            Multc99Event("e", definition)
    d = _simple_design()
    for counts in ([6, 0], [1.0, 2.0], [1, -1], [4, 3]):
        with pytest.raises(ValueError):
            d.monitor_counts(counts)
    with pytest.raises(ValueError):
        run_multc99_trial(d, [1, 1])
    with pytest.raises(ValueError):
        run_multc99_trial(d, [1] * 5, look_sizes=[2, 1])
    with pytest.raises(ValueError):
        simulate_multc99(d, [0, 1], monitoring_period=1)
    with pytest.raises(ValueError):
        simulate_multc99(d, [0, 1], response_window=1)
