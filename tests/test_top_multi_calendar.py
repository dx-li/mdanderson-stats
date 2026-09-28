import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.top_endpoints import TOPMultiEndpointDesign
from mdanderson_stats.top_multi_calendar import run_top_multiendpoint_trial
from mdanderson_stats.top_multi_simulation import simulate_top_multiendpoint

FIXTURES = Path(__file__).parent / "fixtures"


def _design(mode: str, windows: tuple[float, float], *, scale: float = 0.5):
    return TOPMultiEndpointDesign(
        4,
        [0.25] * 4,
        scale,
        0,
        mode=mode,
        windows=windows,
        looks=[2, 4],
    )


def test_hand_calendar_paths_preserve_partial_observation_and_pause_times():
    gaps = np.zeros(4)
    co = _design("coprimary", (1, 2))
    stopped = run_top_multiendpoint_trial(co, gaps, np.full((4, 2), np.inf))
    assert (stopped.decision, stopped.final_time, stopped.pending.tolist()) == (
        "stop_futility",
        2,
        [0, 0],
    )
    assert stopped.observed_outcomes.tolist() == [[0, 0], [0, 0]]

    partial = run_top_multiendpoint_trial(co, gaps, np.tile([0.5, np.inf], (4, 1)))
    assert (partial.decision, partial.final_time, partial.pending.tolist()) == (
        "success",
        1,
        [0, 4],
    )
    assert partial.interim_suspension_time == 0.5
    assert partial.final_followup_time == 0.5
    np.testing.assert_allclose(partial.observed_event_times[:, 0], [0.5, 0.5, 1, 1])
    assert np.all(np.isnan(partial.observed_outcomes[:, 1]))
    assert np.all(np.isinf(partial.observed_event_times[:, 1]))

    early_toxicity = run_top_multiendpoint_trial(
        _design("efficacy_toxicity", (2, 1)),
        gaps,
        np.tile([np.inf, 0.5], (4, 1)),
    )
    assert (early_toxicity.decision, early_toxicity.final_time) == ("stop_toxicity", 0.5)
    assert early_toxicity.pending.tolist() == [2, 0]


def test_calendar_rescales_with_time_units_and_does_not_look_ahead():
    design = _design("coprimary", (1, 2))
    gaps = np.zeros(4)
    delays = np.tile([0.5, np.inf], (4, 1))
    base = run_top_multiendpoint_trial(design, gaps, delays)
    factor = 1e100
    scaled = run_top_multiendpoint_trial(
        _design("coprimary", (factor, 2 * factor)),
        gaps * factor,
        delays * factor,
    )
    assert scaled.decision == base.decision
    np.testing.assert_allclose(scaled.final_time / factor, base.final_time)
    np.testing.assert_allclose(scaled.observed_event_times / factor, base.observed_event_times)


def test_final_only_simulation_matches_independent_multinomial_power():
    with (FIXTURES / "top-multiendpoint-power.csv").open(newline="") as stream:
        reference = next(
            row for row in csv.DictReader(stream) if row["case"] == "coprimary_joint_0.1"
        )
    n = int(reference["maximum"])
    design = TOPMultiEndpointDesign(
        n,
        [float(reference[f"prior{name}"]) for name in ("11", "10", "01", "00")],
        float(reference["cutoff_scale"]),
        float(reference["gamma"]),
        mode=reference["mode"],
        windows=[1, 1],
        looks=[n],
    )
    truth_timing = np.array([[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]])
    original_timing = truth_timing.copy()
    result = simulate_top_multiendpoint(
        design,
        [float(reference[f"truth{name}"]) for name in ("11", "10", "01", "00")],
        1,
        trials=20_000,
        arrival="fixed",
        truth_timing_probabilities=truth_timing,
        rng=71,
    )
    np.testing.assert_array_equal(truth_timing, original_timing)
    expected = float(reference["success"])
    tolerance = 5 * np.sqrt(expected * (1 - expected) / 20_000)
    assert abs(result.success_probability - expected) < tolerance
    np.testing.assert_allclose(
        result.success_probability,
        result.early_success_probability + result.final_success_probability,
    )
    assert np.all(result.events + result.pending <= result.patients[:, None])
