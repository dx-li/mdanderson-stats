"""Exact region output, including disconnected and unreachable outcomes."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import KStageTwoSampleBinomial


@pytest.mark.parametrize("alternative", ["less", "greater", "two-sided"])
@pytest.mark.parametrize("criteria", [(1,), (2,), (3,), (4,), (1, 2)])
def test_regions_partition_outcomes_and_reproduce_decision_probabilities(alternative, criteria):
    d = KStageTwoSampleBinomial(
        [[2, 3], [4, 5], [6, 6]], [0, 0, 1], [2, 2], criteria=criteria, alternative=alternative
    )
    r = d.operating_characteristics(0.37, 0.61)
    for stage in range(1, 4):
        grid = d.decision_grid(stage)
        n1, n2 = d.cumulative_trials[stage - 1]
        assert grid.shape == (n1 + 1, n2 + 1)
        rebuilt = np.full(grid.shape, "", dtype="<U11")
        counts = np.zeros(grid.shape, dtype=int)
        for line in d.region_report(stage).splitlines()[2:]:
            first, lo, hi, label = line.split("\t")
            first, lo, hi = int(first), int(lo), int(hi)
            assert 0 <= lo <= hi <= n2
            rebuilt[first, lo : hi + 1] = label
            counts[first, lo : hi + 1] += 1
        assert_array_equal(counts, 1)
        assert_array_equal(grid, rebuilt)
        mass = d.stage_distribution(stage, 0.37, 0.61)
        assert_allclose(mass[grid == "reject"].sum(), r.rejection[stage - 1])
        assert_allclose(mass[grid == "quit"].sum(), r.quitting[stage - 1])
        assert_allclose(mass[grid == "continue"].sum(), r.continuation[stage - 1])
        assert np.all(mass[grid == "unreachable"] == 0)
        # Public reachable ordering independently identifies all non-unreachable cells.
        reachable = set(map(tuple, d.orderings[stage - 1].events))
        assert {
            (i, j) for i in range(n1 + 1) for j in range(n2 + 1) if grid[i, j] != "unreachable"
        } == reachable


def test_returned_grid_cannot_mutate_design():
    d = KStageTwoSampleBinomial([[2, 2], [4, 4]], [0, 1], [2])
    before = d.operating_characteristics(0.6, 0.2).rejection.copy()
    grid = d.decision_grid(1)
    grid[:] = "quit"
    assert_allclose(d.operating_characteristics(0.6, 0.2).rejection, before)
    assert np.any(d.decision_grid(1) == "reject")


def test_disabled_boundaries_and_final_decisions():
    d = KStageTwoSampleBinomial([[1, 2], [2, 3]], [-1, -1], [-1])
    assert np.all(d.decision_grid(1) == "continue")
    assert np.all(d.decision_grid(2) == "quit")
    assert "0\t0\t3\tquit" in d.region_report(2)


def test_export_and_io_failure(tmp_path):
    d = KStageTwoSampleBinomial([[2, 2], [4, 4]], [0, 1], [2])
    path = tmp_path / "regions.tsv"
    path.write_text("old")
    assert d.write_regions(path) == path
    content = path.read_text()
    assert content.startswith("KSBIN2 decision regions\nDirection\tgreater\nCriteria\t1 2\n")
    assert d.region_report(1) + d.region_report(2) in content
    with pytest.raises(OSError):
        d.write_regions(tmp_path / "missing" / "file")


@pytest.mark.parametrize("stage", [0, 3, True, 1.5])
def test_invalid_stage(stage):
    d = KStageTwoSampleBinomial([[2, 2], [4, 4]], [0, 1], [2])
    with pytest.raises(ValueError):
        d.region_report(stage)
