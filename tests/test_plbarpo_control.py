import numpy as np
import pytest

from mdanderson_stats.plbarpo_control import (
    plbarpo_control_counts,
    plbarpo_control_monitor,
)


def test_control_counts_use_half_open_windows_and_as_of():
    counts = plbarpo_control_counts(
        [0, 1, 2, 3, 4],
        [1, 0, 1, 0, 1],
        [0, 1, np.inf, 3, 4],
        [[0, 2], [2, 4], [4, np.inf]],
        as_of=4,
    )
    np.testing.assert_array_equal(counts, [[1, 1], [0, 1], [1, 0]])
    assert not counts.flags.writeable


def test_control_counts_reject_invalid_records_and_windows():
    with pytest.raises(ValueError, match="observation_time"):
        plbarpo_control_counts([1], [1], [0], [[0, 2]], as_of=2)
    with pytest.raises(ValueError, match="open < close"):
        plbarpo_control_counts([1], [1], [1], [[2, 2]], as_of=2)
    counts = plbarpo_control_counts([1, 2], [np.nan, 1], [np.inf, 2], [[0, 3]], as_of=2)
    np.testing.assert_array_equal(counts, [[1, 0]])
    with pytest.raises(ValueError, match="missing outcomes"):
        plbarpo_control_counts([1], [np.nan], [1], [[0, 3]], as_of=2)


def test_entire_control_monitor_returns_treatment_only_ordering():
    result = plbarpo_control_monitor(
        [1, 8],
        [4, 2],
        prior=[[1, 1], [1, 1]],
        control_prior=[1, 1],
        control_counts=[4, 6],
        control_mode="entire",
        pfut=0.8,
        peff=0.8,
        pfinal=0.8,
    )
    assert result.treatment_alpha.shape == (2,)
    assert result.control_alpha.shape == (2,)
    assert result.efficacy_probability[1] > result.efficacy_probability[0]
    assert result.efficacious.dtype == bool
    assert not result.efficacious.flags.writeable


def test_concurrent_control_uses_one_control_posterior_per_window():
    result = plbarpo_control_monitor(
        [3, 3],
        [1, 1],
        prior=[[1, 1], [1, 1]],
        control_prior=[1, 1],
        control_counts=[[3, 1], [1, 3]],
        control_mode="concurrent",
        peff=0.0,
    )
    assert result.control_alpha[0] == 4
    assert result.control_alpha[1] == 2
    np.testing.assert_array_equal(result.efficacious, [True, True])


def test_monitor_validation_happens_before_comparison():
    with pytest.raises(ValueError, match="control_mode"):
        plbarpo_control_monitor(
            [1],
            [1],
            prior=[[1, 1]],
            control_prior=[1, 1],
            control_counts=[1, 1],
            control_mode="bad",
        )
