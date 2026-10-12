"""Whole native wrappers under independently controlled outcome/tie services."""

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_boin_desktop_patients import study

from mdanderson_stats.boin_desktop import BOINDesktopScenario
from mdanderson_stats.boin_desktop_simulation import (
    _contour_trials,
    _single_trials,
    simulate_boin_desktop_combination,
)

ROWS = list(
    csv.DictReader(
        (Path(__file__).parents[1] / "research/fixtures/boin-desktop/simulation.csv")
        .read_text()
        .splitlines()
    )
)


class NativeProbeGenerator(np.random.Generator):
    def random(self, size=None, dtype=np.float64, out=None):
        return 0.5 if size is None else np.full(size, 0.5, dtype=dtype)

    def binomial(self, n, p, size=None):
        return int(n * (0.5 < p)) if size is None else np.full(size, int(n * (0.5 < p)))

    def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
        if high is None:
            low, high = 0, low
        value = low + int(0.5 * (high - low))
        return value if size is None else np.full(size, value, dtype=dtype)


def source_study(row):
    p = {
        "none": np.zeros((2, 3)),
        "all": np.ones((2, 3)),
        "gradient": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
    }[row["truth"]]
    saved = replace(
        replace(
            study(),
            xml=study()
            .xml.replace("<ExtraSafe>false", "<ExtraSafe>" + row.get("extra_safe", "FALSE").lower())
            .replace(
                "<AcceptableToxicity_UpperBound>.35",
                "<AcceptableToxicity_UpperBound>" + row.get("upper", ".35"),
            ),
        ),
        maximum_patients=18,
        repetitions=2,
        mode="contour" if row["mode"] == "contour" else "combination",
        subtrial_maximums=(12, 6) if row["mode"] == "contour" else (),
        run_in=row["runin"] == "TRUE",
        titration=row["titration"] == "TRUE",
        titration_cap=int(row["cap"]),
        scenarios=(BOINDesktopScenario("Reference truth", tuple(tuple(r) for r in p)),),
    )
    return saved, p


@pytest.mark.parametrize("row", ROWS)
def test_complete_original_wrapper(row):
    saved, p = source_study(row)
    generator = NativeProbeGenerator(np.random.PCG64(1))
    run = _contour_trials if row["mode"] == "contour" else _single_trials
    n, y, actual, actual_tox, selected, unsafe, reasons = run(
        saved, p, 2, generator, "desktop_legacy"
    )

    def expected(field):
        return np.fromstring(row[field], sep=";").reshape(2, 3)

    np.testing.assert_allclose(n.mean(axis=0), expected("mean_patients"), atol=0.005)
    np.testing.assert_allclose(y.mean(axis=0), expected("mean_toxicities"), atol=0.005)
    np.testing.assert_allclose(selected.mean(axis=0) * 100, expected("selection"), atol=0.005)
    assert abs(n.sum(axis=(1, 2)).mean() - float(row["total_patients"])) <= 0.051
    assert actual.shape == n.shape and len(reasons) == 2


@pytest.mark.parametrize("mode", ["single", "contour"])
def test_safe_counts_capacity_and_seed_replay(mode):
    row = dict(mode=mode, truth="gradient", runin="TRUE", titration="TRUE", cap="2")
    saved, _ = source_study(row)
    result = simulate_boin_desktop_combination(saved, trials=5, seed=971)
    replay = simulate_boin_desktop_combination(saved, trials=5, seed=971)
    np.testing.assert_array_equal(result.patients, result.actual_patients)
    np.testing.assert_array_equal(result.patients, replay.patients)
    assert np.all(result.patients.sum(axis=(1, 2)) <= 18)
    assert np.all(result.toxicities <= result.actual_patients)
    assert not result.patients.flags.writeable
    assert np.all(result.selection_mcse >= 0)


def test_legacy_phantom_counts_are_visible_and_do_not_affect_default():
    saved, p = source_study(
        dict(mode="single", truth="none", runin="FALSE", titration="TRUE", cap="2")
    )
    result = _single_trials(saved, p, 2, NativeProbeGenerator(np.random.PCG64(1)), "desktop_legacy")
    assert np.any(result[0] != result[2])
    safe = _single_trials(saved, p, 2, NativeProbeGenerator(np.random.PCG64(1)), "safe")
    np.testing.assert_array_equal(safe[0], safe[2])


def test_whole_request_bounds_precede_simulation():
    saved = study()
    for kwargs in (
        {"trials": 0},
        {"trials": 10001},
        {"seed": True},
        {"scenario_index": -1},
        {"profile": "unknown"},
    ):
        with pytest.raises(ValueError):
            simulate_boin_desktop_combination(saved, **kwargs)
