import csv

import numpy as np
import pytest

from mdanderson_stats import (
    Multc99Event,
    multc99_design,
    multc99_probability_curve,
    multc99_study_report,
    multc99_with_boundaries,
    replay_multc99_study,
    run_multc99_trial,
    simulate_multc99,
)


def test_manual_boundaries_change_conduct_and_survive_saved_input_replay(tmp_path):
    original = multc99_design(
        [1, 1],
        [1, 1],
        [Multc99Event("e", (1, 0))],
        max_subjects=6,
        min_subjects=3,
    )
    lower = np.full((1, 7), -1)
    lower[0, 3:] = 1
    manual = multc99_with_boundaries(original, lower, np.arange(7)[None, :] + 1)
    assert manual.boundary_origin == "manual"
    # At n=2, the source run-back limit is 0; two failures stop early.
    assert run_multc99_trial(manual, [1] * 6).sample_size == 2
    assert run_multc99_trial(original, [1] * 6).sample_size == 6
    report = multc99_study_report(simulate_multc99(manual, [0.4, 0.6], trials=12, seed=91))
    replay = replay_multc99_study(report.write_inputs(tmp_path / "manual.json"))
    assert replay.results.design.boundary_origin == "manual"
    np.testing.assert_array_equal(replay.results.design.lower_bounds, manual.lower_bounds)
    np.testing.assert_array_equal(replay.results.sample_sizes, report.results.sample_sizes)


def test_probability_curve_exports_ordered_full_precision_csv(tmp_path):
    design = multc99_design([1, 1], [1, 1], [Multc99Event("e", (1, 0))], max_subjects=6)
    # Independent triangular distribution of the difference of two uniforms.
    curve = multc99_probability_curve(design, "e", [0, 0], [0.5, 0, -0.5, 1, -1])
    np.testing.assert_allclose(curve.probabilities, [0.125, 0.5, 0.875, 0, 1], atol=2e-9, rtol=0)
    path = curve.write_csv(tmp_path / "curve.csv")
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    np.testing.assert_array_equal([float(row["margin"]) for row in rows], curve.margins)
    assert not curve.probabilities.flags.writeable


def test_invalid_manual_suffixes_and_oversized_curves_are_rejected():
    design = multc99_design([1, 1], [1, 1], [Multc99Event("e", (1, 0))], max_subjects=6)
    lower = np.full((1, 7), -1)
    upper = np.arange(7)[None, :] + 1
    bad = lower.copy()
    bad[0, 3] = 7
    with pytest.raises(ValueError, match="nondecreasing"):
        multc99_with_boundaries(design, bad, upper)
    bad = lower.copy()
    bad[0, 3] = 2
    with pytest.raises(ValueError, match="nondecreasing"):
        multc99_with_boundaries(design, bad, upper)
    with pytest.raises(ValueError, match="1,000"):
        multc99_probability_curve(design, "e", [0, 0], [0] * 1001)
