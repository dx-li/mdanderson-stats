"""Archived piecewise hazards and independent person-time accounting."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import pehaz

CASES = json.loads((Path(__file__).parent / "fixtures/pehaz.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_archived_s_function(case):
    r = pehaz(case["times"], case["delta"], width=case["width"], bounds=case["bounds"], legacy=True)
    rows = np.column_stack([r.cuts[:-1], r.cuts[1:], r.hazard, r.events, r.at_risk, r.follow_up])
    assert_allclose(rows, np.array(case["rows"], dtype=float), rtol=2e-13, atol=2e-14)


def test_final_event_is_retained_and_final_bin_is_truncated():
    r = pehaz([0.5, 1, 2, 2.5, 3], [1, 1, 1, 1, 1], width=1, bounds=(0, 2.5))
    assert_array_equal(r.cuts, [0, 1, 2, 2.5])
    assert_array_equal(r.events, [1, 1, 2])
    assert_allclose(r.follow_up, [4.5, 3, 1])
    legacy = pehaz([1, 2, 3], width=1, legacy=True)
    ordinary = pehaz([1, 2, 3], width=1)
    assert legacy.events.sum() == 2
    assert ordinary.events.sum() == 3


def test_person_time_and_events_match_direct_bin_calculations():
    rng = np.random.default_rng(787)
    times = rng.uniform(0, 20, 500)
    status = rng.integers(0, 2, 500)
    r = pehaz(times, status, width=0.7, bounds=(3, 16))
    for i, (left, right) in enumerate(zip(r.cuts[:-1], r.cuts[1:], strict=True)):
        expected = np.clip(times - left, 0, right - left).sum()
        assert_allclose(r.follow_up[i], expected)
        assert r.at_risk[i] == np.count_nonzero(times >= left)
        assert r.events[i] == np.sum(status[(times >= left) & (times < right)])
        assert_allclose(r.hazard[i], r.events[i] / expected)
    assert_allclose(r.follow_up.sum(), np.clip(times - 3, 0, 13).sum())
    assert r.events.sum() == status[(times >= 3) & (times <= 16)].sum()


def test_default_width_and_permutation_time_scaling():
    t = np.array([0.2, 0.9, 1.3, 2, 3, 4])
    d = np.array([1, 1, 0, 1, 1, 0])
    r = pehaz(t, d)
    assert_allclose(r.width, 4 / (8 * 4**0.2))
    other = pehaz(t[::-1] * 3, d[::-1], width=r.width * 3)
    assert_allclose(other.cuts, r.cuts * 3)
    assert_array_equal(other.events, r.events)
    assert_allclose(other.follow_up, r.follow_up * 3)
    assert_allclose(other.hazard, r.hazard / 3)


def test_large_time_origin_retains_small_exposures():
    start = 1e12
    r = pehaz([start + 0.25, start + 0.5, start + 1], width=0.25, bounds=(start, start + 1))
    assert_allclose(r.follow_up, [0.75, 0.5, 0.25, 0.25], rtol=0, atol=0)


def test_no_exposure_is_unavailable_and_instantaneous_event_has_infinite_mle():
    r = pehaz([1, 2], [0, 0], width=1, bounds=(0, 4))
    assert_array_equal(r.hazard[:2], [0, 0])
    assert np.all(np.isnan(r.hazard[2:]))
    assert "NA" in r.report()
    instant = pehaz([0], width=1, bounds=(0, 1))
    assert instant.hazard[0] == np.inf
    assert "inf" in instant.report()


def test_report_and_readonly_results(tmp_path):
    r = pehaz([0.2, 0.7, 1.8, 2], width=0.75)
    text = r.report(digits=17)
    parsed = np.loadtxt(io.StringIO(text), skiprows=2)
    assert_array_equal(parsed[:, 2], r.hazard)
    destination = tmp_path / "hazard.tsv"
    r.write_report(destination, digits=17)
    assert destination.read_text() == text
    with pytest.raises(ValueError):
        r.follow_up[0] = 9
    with pytest.raises(FileNotFoundError):
        r.write_report(tmp_path / "absent" / "report")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"times": []},
        {"times": [-1, 2]},
        {"delta": [1]},
        {"delta": [1, 2]},
        {"width": 0},
        {"width": np.nan},
        {"bounds": (2, 1)},
        {"bounds": (0, np.inf)},
        {"times": [0, 0]},
        {"times": [1, np.nan]},
        {"delta": [0, 0], "width": None},
        {"legacy": 1},
    ],
)
def test_invalid_input(kwargs):
    args = dict(times=[1, 2], width=0.5)
    args.update(kwargs)
    with pytest.raises(ValueError):
        pehaz(**args)


@pytest.mark.parametrize("digits", [0, 18, 1.5, True])
def test_invalid_precision(digits):
    with pytest.raises(ValueError):
        pehaz([1, 2], width=1).report(digits=digits)


def test_width_far_larger_than_domain_still_produces_one_bin():
    r = pehaz([1e-308], width=1e308, bounds=(0, 1e-308))
    assert r.events.tolist() == [1]
    assert_allclose(r.follow_up, [1e-308], atol=0)
    assert_allclose(r.hazard, [1e308])


def test_finite_but_unrepresentable_hazard_is_a_numerical_error():
    with pytest.raises(RuntimeError, match="numerical range"):
        pehaz([1e-320], width=1, bounds=(0, 1e-320))
