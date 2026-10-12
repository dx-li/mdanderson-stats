"""Executed original current-desktop branch references, supplied posterior services."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.efftox_desktop_decision import _admissible, _select

ROOT = Path(__file__).resolve().parents[1] / "research/fixtures/efftox-native"


@pytest.mark.parametrize("case", json.loads((ROOT / "current-admissibility.json").read_text()))
def test_original_current_admissibility(case):
    actual = _admissible(
        np.array(case["counts"]),
        np.full(5, case["toxicity_tail"]),
        np.full(5, case["efficacy_tail"]),
        case["start"],
        0.1,
        0.1,
    )
    assert bool(actual[case["candidate"]]) == case["admissible"]


@pytest.mark.parametrize("case", json.loads((ROOT / "current-selection.json").read_text()))
def test_original_current_selection(case):
    counts = np.array(case["counts"])
    admissible = _admissible(
        counts,
        np.full(5, case["toxicity_tail"]),
        np.full(5, case["efficacy_tail"]),
        case["start"],
        0.1,
        0.1,
    )
    selected = _select(counts, admissible, np.array(case["utilities"]), case["start"])
    assert (selected is None) == case["stopped"]
    assert selected == (None if case["selected"] == -1 else case["selected"])


def test_current_profile_integrates_binary_posterior_without_changing_generic_default():
    from mdanderson_stats import EffToxContour, EffToxPrior, efftox_decision, fit_efftox
    from mdanderson_stats.efftox_desktop_decision import efftox_desktop_decision

    counts = np.zeros((5, 2, 2), dtype=int)
    counts[0, 1, 0] = counts[2, 1, 0] = 1
    fit = fit_efftox(
        [1, 2, 3, 4, 5],
        counts,
        prior=EffToxPrior(np.array([-10, 0.01, 0, 1, 0, 0]), np.zeros(6)),
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(7),
    )
    contour = EffToxContour.from_points(0.5, 0.65, 0.7, 0.25)
    settings = dict(
        efficacy_limit=0.01, toxicity_limit=0.5, efficacy_probability=0.1, toxicity_probability=0.1
    )
    current = efftox_desktop_decision(fit, contour, starting_dose_index=0, **settings)
    generic = efftox_decision(fit, contour, starting_dose=1, phase="final", **settings)
    assert current.dose_index == 3
    assert generic.dose == 5
    np.testing.assert_array_equal(current.admissible, [True, True, True, True, False])
