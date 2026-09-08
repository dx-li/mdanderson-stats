"""Independent life-table geometry checks for every SCAT-BOX branch."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import censored_box


@pytest.mark.parametrize(
    "deaths,quartiles,lower,upper",
    [
        (0, [np.nan, np.nan, np.nan], None, None),
        (1, [np.nan, np.nan, np.nan], 1, 1),
        (2, [8, np.nan, np.nan], 8, 2),
        (4, [2, 8, np.nan], 2, 4),
        (6, [2, 4, 8], 2, 8),
        (8, [2, 4, 6], 2, 6),
    ],
)
def test_every_source_box_branch_including_last_plateau_quartiles(deaths, quartiles, lower, upper):
    box = censored_box(np.arange(1, 9), [1] * deaths + [0] * (8 - deaths))
    assert_allclose(box.quartiles, quartiles, equal_nan=True)
    if deaths == 0:
        assert box.first_failure_time is box.last_failure_time is None
        assert box.first_failure_survival is box.last_failure_survival is None
        assert box.segments.shape == (0, 2, 2)
    else:
        expected = [[[0.667, q], [1.33, q]] for q in quartiles if np.isfinite(q)]
        expected += [[[x, lower], [x, upper]] for x in (0.667, 1.33)]
        assert_allclose(box.segments, expected)
        assert box.first_failure_time == 1 and box.last_failure_time == deaths
        assert box.first_failure_survival == 0.875
        assert box.last_failure_survival == 1 - deaths / 8
    assert not box.quartiles.flags.writeable and not box.segments.flags.writeable


def test_step_option_and_unsorted_patient_alignment():
    box = censored_box([8, 2, 1, 3, 4, 5, 6, 7], [0, 1, 1, 0, 0, 0, 0, 0], quantile_method="step")
    assert_allclose(box.quartiles, [2, np.nan, np.nan], equal_nan=True)
    assert box.last_failure_time == 2
    assert_array_equal(box.segments[-1], [[1.33, 2], [1.33, 2]])


def test_sequential_tie_mode_retains_first_failure_survival():
    grouped = censored_box([1, 1, 2, 3], [1, 1, 0, 0])
    source = censored_box([1, 1, 2, 3], [1, 1, 0, 0], legacy=True)
    assert grouped.first_failure_survival == 0.5
    assert source.first_failure_survival == 0.75
    assert source.last_failure_survival == 0.5


def test_zero_time_failure_and_invalid_quantile_method():
    box = censored_box([0], [1])
    assert_array_equal(box.quartiles, [0, 0, 0])
    assert box.last_failure_survival == 0
    with pytest.raises(ValueError):
        censored_box([1], [1], quantile_method="unknown")
