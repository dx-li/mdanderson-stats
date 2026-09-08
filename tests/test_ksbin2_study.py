"""Study-level scan/report semantics and independent expected sample sizes."""

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import KStageTwoSampleBinomial, ksbin2_boundary_table, ksbin2_study


def design(alternative="greater"):
    return KStageTwoSampleBinomial(
        [[2, 2], [3, 3], [4, 4]], [0, 0, 1], [2, 2], criteria=(1,), alternative=alternative
    )


@pytest.mark.parametrize("alternative", ["less", "greater", "two-sided"])
@pytest.mark.parametrize("null", [0, 0.2, 0.5, 1])
def test_independent_null_decisions_and_report(alternative, null):
    d = design(alternative)
    s = ksbin2_study(d, 0.6, 0.2, null_probability=null)
    rej, qt = np.zeros((2, 3))
    for first in itertools.product([0, 1], repeat=4):
        for second in itertools.product([0, 1], repeat=4):
            k = sum(first) + sum(second)
            weight = null**k * (1 - null) ** (8 - k)
            for stage, n in enumerate([2, 3, 4]):
                point = (sum(first[:n]), sum(second[:n]))
                ordering = d.orderings[stage]
                row = next(i for i, p in enumerate(ordering.events) if tuple(p) == point)
                group = np.searchsorted(ordering.group_end, row)
                if group <= d.reject_group[stage]:
                    rej[stage] += weight
                    break
                if stage == 2 or group >= d.quit_group[stage]:
                    qt[stage] += weight
                    break
    assert_allclose(s.null.rejection, rej, atol=3e-14)
    assert_allclose(s.null.quitting, qt, atol=3e-14)
    expected = (rej + qt) @ np.array(d.cumulative_trials)
    assert_allclose(s.null.expected_sample_size, expected, atol=3e-14)
    lines = s.report(digits=17).splitlines()
    start = lines.index("Null operating characteristics") + 2
    for i in range(3):
        values = list(map(float, lines[start + i].split("\t")))
        assert_allclose(
            values[1:5], [rej[i], sum(rej[: i + 1]), qt[i], sum(qt[: i + 1])], atol=3e-14
        )
    assert_allclose(list(map(float, lines[start + 3].split("\t")[1:])), expected, atol=3e-14)


def test_null_grid_matches_selected_boundary_tables():
    d = design()
    s = ksbin2_study(d, 0.6, 0.2, null_probability=0.1)
    for stage, group in enumerate(d.reject_group, start=1):
        t = ksbin2_boundary_table(d, stage, 0.6, 0.2)
        assert_allclose(s.cumulative_grid_significance[stage - 1], t.significance[group])
    # PRH0 changes actual null characteristics, not nuisance-grid significance.
    changed = s.revise(null_probability=0.5)
    assert_allclose(changed.cumulative_grid_significance, s.cumulative_grid_significance)
    assert not np.allclose(s.null.expected_sample_size, changed.null.expected_sample_size)
    assert not np.isclose(s.null.rejection_probability, s.cumulative_grid_significance[-1])


def test_broadcast_scan_revision_and_files(tmp_path):
    d = design()
    s = ksbin2_study(d, [[0.5], [0.7]], [0.1, 0.2, 0.3])
    assert s.alternative.rejection.shape == (2, 3, 3)
    assert_allclose(s.null_probability, ([[0.5], [0.7]] + np.array([0.1, 0.2, 0.3])) / 2)
    before = s.report(digits=17)
    assert sum(line.startswith("Case ") for line in before.splitlines()) == 6
    changed = s.revise(probability1=0.8)
    assert_allclose(changed.null_probability, s.null_probability)
    fresh = ksbin2_study(d, 0.8, s.probability2, null_probability=s.null_probability)
    assert_allclose(changed.alternative.rejection, fresh.alternative.rejection)
    assert s.report(digits=17) == before
    assert_allclose(
        s.scan(0.5, 0.5).expected_sample_size,
        d.operating_characteristics(0.5, 0.5).expected_sample_size,
    )
    path = tmp_path / "report.tsv"
    path.write_text("old")
    assert s.write_report(path, digits=17) == path
    assert path.read_text() == before
    new = s.revise(design=KStageTwoSampleBinomial([[3, 4]], [-1], []), null_grid=[0.2, 0.5, 0.8])
    assert_allclose(new.cumulative_grid_significance, [0])
    assert new.null_scan.rejection.shape == (3, 1)


@pytest.mark.parametrize(
    "changes",
    [
        {"design": None},
        {"probability1": 1e308},
        {"null_probability": -0.1},
        {"null_grid": []},
        {"null_grid": [0.5, 0.4]},
        {"null_grid": [[0.2, 0.5]]},
    ],
)
def test_invalid_inputs(changes):
    args = dict(design=design(), probability1=0.6, probability2=0.2)
    args.update(changes)
    with pytest.raises(ValueError):
        ksbin2_study(**args)


def test_report_errors_preserve_file(tmp_path):
    s = ksbin2_study(design(), 0.6, 0.2)
    path = tmp_path / "report"
    path.write_text("preserve")
    for digits in [0, 18, True, 1.5]:
        with pytest.raises(ValueError):
            s.write_report(path, digits=digits)
        assert path.read_text() == "preserve"
    with pytest.raises(OSError):
        s.write_report(tmp_path / "absent" / "file")
