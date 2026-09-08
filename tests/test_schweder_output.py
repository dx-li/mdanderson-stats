import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import plot_schweder, schweder_fit, write_schweder_data

REFERENCE = json.loads((Path(__file__).parent / "fixtures/multi.json").read_text())


@pytest.mark.parametrize(
    "case", [case for case in REFERENCE["schweder"] if not case["reference"]["desktop"]["status"]]
)
def test_export_preserves_all_native_coordinates(case, tmp_path):
    fit = schweder_fit(case["pvalues"], alpha=case["alpha"])
    path = write_schweder_data(fit, tmp_path / "coordinates.csv")
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        assert next(reader) == ["1-p", "Np"]
        rows = list(reader)
    reference = case["reference"]["desktop"]
    np.testing.assert_allclose(
        [float(row[0]) for row in rows], reference["one_minus_p"], atol=1e-15
    )
    np.testing.assert_array_equal([int(row[1]) for row in rows], reference["upper_counts"])
    np.testing.assert_array_equal([float(row[0]) for row in rows], fit.one_minus_p)


def test_plot_renders_observations_and_original_line(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fit = schweder_fit([0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 0.8])
    fig, supplied = plt.subplots()
    try:
        ax = plot_schweder(fit, supplied)
        assert ax is supplied
        points, line = ax.lines
        np.testing.assert_array_equal(points.get_xdata(), fit.one_minus_p)
        np.testing.assert_array_equal(points.get_ydata(), fit.upper_counts)
        np.testing.assert_array_equal(line.get_xdata(), [0, 1])
        np.testing.assert_allclose(line.get_ydata(), [0, fit.null_estimate])
        assert ax.get_xlim() == (0, 1)
        assert ax.get_xlabel() == "1-p" and ax.get_ylabel() == "Np"
        path = tmp_path / "schweder.png"
        fig.savefig(path)
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close(fig)
    created = plot_schweder(fit)
    plt.close(created.figure)
