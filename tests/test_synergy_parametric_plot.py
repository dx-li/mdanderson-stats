import csv
import io
from pathlib import Path

import pytest

from mdanderson_stats import fit_synergy_parametric, plot_synergy_parametric


@pytest.mark.parametrize("model", ["greco", "machado", "plummer", "carter"])
def test_source_workflow_produces_curves_and_exportable_contours(model):
    pyplot = pytest.importorskip("matplotlib.pyplot")
    path = Path(__file__).parent / "fixtures/synergy-parametric-inputs.csv"
    with path.open(newline="") as source:
        rows = [
            row
            for row in csv.DictReader(source)
            if row["model"] == model and row["case"] == "synthetic"
        ]
    fit = fit_synergy_parametric(
        [float(r["drug1"]) for r in rows],
        [float(r["drug2"]) for r in rows],
        [float(r["response"]) for r in rows],
        model=model,
    )
    figure = plot_synergy_parametric(fit, dose_ratio=2.0, resolution=20)
    try:
        assert len(figure.axes) == 2
        assert len(figure.axes[0].lines) == 3
        assert len(figure.axes[0].collections) == 3
        output = io.BytesIO()
        figure.savefig(output, format="png")
        assert len(output.getvalue()) > 3000
        before = pyplot.get_fignums()
        for options in [{"dose_ratio": 0}, {"resolution": 151}, {"contour_levels": [0.5, 0.3]}]:
            with pytest.raises(ValueError):
                plot_synergy_parametric(fit, **options)
        assert pyplot.get_fignums() == before
    finally:
        pyplot.close(figure)
