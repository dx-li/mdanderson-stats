"""Real file round trips and aligned named observations."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats import ExploratoryTable, exploratory_survival


def test_roundtrip_full_precision_and_stable_cosort(tmp_path):
    values = np.array([[2, 0, np.nextafter(1.0, 2.0)], [1, 1, 7], [2, 1, 8]])
    table = ExploratoryTable(["time", "status", "id"], values)
    values[:] = -1
    assert table.column("time").tolist() == [2, 1, 2]
    sorted_table = table.cosort("time")
    assert_array_equal(sorted_table.column("id"), [7, np.nextafter(1.0, 2.0), 8])
    assert not table.values.flags.writeable and not table.column("id").flags.writeable
    path = tmp_path / "data.txt"
    table.write(path)
    restored = ExploratoryTable.read(path, table.names)
    assert_array_equal(restored.values, table.values)
    assert (
        exploratory_survival(restored.column("time"), restored.column("status")).n_observations == 3
    )
    with pytest.raises(KeyError):
        table.column("absent")


def test_whitespace_and_single_row_column(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("\n  1\t2  \n\n3 4\n")
    assert_array_equal(ExploratoryTable.read(path, ["a", "b"]).values, [[1, 2], [3, 4]])
    path.write_text("1\n")
    assert ExploratoryTable.read(path, ["a"]).values.shape == (1, 1)


@pytest.mark.parametrize("text", ["", "1 2\n3\n", "NA 2\n", "nan 2\n", "inf 2\n", "# comment\n"])
def test_invalid_file_is_rejected(tmp_path, text):
    path = tmp_path / "bad.txt"
    path.write_text(text)
    with pytest.raises(ValueError):
        ExploratoryTable.read(path, ["a", "b"])


@pytest.mark.parametrize(
    "names,values",
    [
        ([], [[1]]),
        (["a", "a"], [[1, 2]]),
        ([""], [[1]]),
        (["a"], []),
        (["a"], [1]),
        (["a"], [[np.nan]]),
        (["a"], [[1, 2]]),
    ],
)
def test_invalid_table(names, values):
    with pytest.raises(ValueError):
        ExploratoryTable(names, values)
