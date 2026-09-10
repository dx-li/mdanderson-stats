import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats import event_convert


def test_original_factor_order_unused_levels_and_shared_time_column():
    ref = json.loads((Path(__file__).parent / "fixtures/eventchart-codes-native.json").read_text())
    x = [
        [5, "death", "10"],
        [6, "censored", "2"],
        [None, "death", "2"],
        [1, "death", "10"],
        [2, "censored", "2"],
    ]
    result = event_convert(
        x,
        time_columns=[0, 0],
        code_columns=[1, 2],
        names=("time", "cause", "status"),
        code_levels=(("death", "censored", "relapse"), None),
    )
    assert_array_equal(result.times, np.asarray(ref["times"], dtype=float))
    assert result.names == tuple(ref["names"])
    assert result.codes == ("death", "censored", "relapse", "10", "2")
    assert result.source_columns == (0, 0, 0, 0, 0)
    assert not result.times.flags.writeable


def test_missing_literal_labels_numeric_identity_and_invalid_categories():
    result = event_convert([[1, "NA"], [2, ""], [3, None], [4, np.nan], [None, "NA"]])
    assert result.codes == ("", "NA")
    assert_array_equal(
        result.times,
        [[np.nan, 1], [2, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
    )
    result = event_convert([[1, 2**53], [2, 2**53 + 1]])
    assert result.codes == (2**53, 2**53 + 1)
    assert result.names[0] != result.names[1]
    assert_array_equal(result.times, [[1, np.nan], [np.nan, 2]])
    result = event_convert([[1, None]], code_levels=(("death",),))
    assert result.times.shape == (1, 1)
    assert np.isnan(result.times).all()
    for x, levels in [
        ([[1, "a"], [2, 1]], None),
        ([[1, "a"]], (("b",),)),
        ([[1, "a"]], (("a", "a"),)),
        ([[1, np.inf]], None),
    ]:
        with pytest.raises(ValueError):
            event_convert(x, code_levels=levels)
