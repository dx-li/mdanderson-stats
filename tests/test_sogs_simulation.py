import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import (
    SOGS_MOUSE_LENGTHS,
    format_sogs,
    sogs_recombine,
    sogs_screen,
    sogs_simulate,
    sogs_summary,
)
from mdanderson_stats.sogs_simulation import _gamete, _missed


def test_native_monte_carlo_all_rules_with_and_without_screening():
    refs = json.loads((Path(__file__).parent / "fixtures/sogs-native-simulation.json").read_text())
    for ref in refs:
        result = sogs_simulate(
            offspring=(3, 2, 4, 3),
            exclude=tuple(range(3, 20)),
            rule=ref["rule"],
            screening_error=ref["screen"],
            avoidance=5 if ref["screen"] else 0,
            replicates=1000,
            seed=9056 + 4 * int(ref["screen"]) + ref["rule"],
        )
        s = sogs_summary(result)
        uncertainty = np.sqrt(s.mcse[1:, :4] ** 2 + np.asarray(ref["mcse"]) ** 2)
        assert np.all(np.abs(s.mean[1:, :4] - ref["mean"]) <= 5 * uncertainty + 1e-5)
        assert np.all(np.diff(result.pure_ipt, axis=1) >= 0)
        assert np.all(np.diff(result.donor_length, axis=1) <= 1e-10)
        assert np.all(result.missed_length <= result.donor_length)
        assert np.all(result.missed + result.pure_ipt <= 2)


def test_random_breeding_halves_donor_material_and_first_cross_purity():
    result = sogs_simulate(
        offspring=(1, 1, 1, 1),
        exclude=tuple(range(2, 20)),
        rule=0,
        screening_error=False,
        replicates=20000,
        seed=1956,
    )
    summary = sogs_summary(result)
    expected = SOGS_MOUSE_LENGTHS[0] / 2.0 ** np.arange(5)
    assert np.all(np.abs(summary.mean[:, 2] - expected) <= 5 * summary.mcse[:, 2] + 1e-9)
    p = 0.5 * np.exp(-SOGS_MOUSE_LENGTHS[0] / 100)
    assert abs(result.pure_ipt[:, 1].mean() - p) <= 5 * np.sqrt(p * (1 - p) / 20000)
    assert_array_equal(result.missed, np.zeros_like(result.missed))


def test_inner_partition_replay_and_generation_aligned_report():
    parent = [(1.0, 4.0), (6.0, 9.0), (12.0, 17.0)]
    breaks = [2.0, 7.0, 13.0, 16.0]
    ref = sogs_recombine(parent, 20, breaks)
    for i, segments in enumerate((ref.retained, ref.transferred)):
        child = _gamete(parent, breaks, i)
        assert_array_equal(child, segments)
        assert _missed(child, [0.0, 5.0, 10.0, 15.0, 20.0]) == sogs_screen(segments, 20).missed
    options = dict(
        offspring=(2, 3), exclude=tuple(range(3, 20)), replicates=30, seed=2040, avoidance=10
    )
    a, b = sogs_simulate(**options), sogs_simulate(**options)
    assert_array_equal(a.donor_length, b.donor_length)
    assert_array_equal(a.missed, b.missed)
    s = sogs_summary(a)
    assert s.stages == ("F1", "BC1", "BC2")
    assert s.mean[0, 0] == 2
    assert s.mean[0, 4] == 100
    assert_allclose(s.apparent_dmt_cdf[:, -1], 1)
    assert np.all(np.diff(s.apparent_dmt_cdf, axis=1) >= 0)
    report = format_sogs(a)
    assert "F1 is the initial state" in report and "BC2" in report and "P(count <= k)" in report
    assert not a.donor_length.flags.writeable
