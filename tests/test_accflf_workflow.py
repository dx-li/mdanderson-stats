import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    accflf_marginal_survival,
    accflf_report,
    accflf_survival,
    fit_accflf,
    read_accflf_data,
)


def test_source_table_selection_fit_report_and_marginal_prediction(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text(
        "# source format\ntime status multi x extra\n1 1 2 0 5\n"
        "2 1 1 0 6\n3 0 3 1 4\n4 1 2 2 7\n6 1 1 1 8\n8 0 1 0 3\n"
    )
    data = read_accflf_data(path)
    assert data.covariate_names == ()
    assert data.available_covariates == ("X", "EXTRA")
    selected = data.add(("x",)).add(("extra",)).select(("x",))
    assert selected.covariate_names == ("X",)
    assert data.covariate_names == ()
    assert not selected.values.flags.writeable
    with pytest.raises(ValueError, match="distinct"):
        selected.add(("X",))
    with pytest.raises(ValueError, match="excluding"):
        selected.select(("TIME",))
    fit = fit_accflf(
        selected.time,
        selected.event,
        covariates=selected.covariates,
        weights=selected.weights,
        p=0,
        q=1,
        fixed_sigma=1,
    )
    report = accflf_report(fit, covariate_names=selected.covariate_names)
    assert f"Log likelihood={fit.log_likelihood:.12g}" in report
    assert "X=" in report and "Sigma fixed=True" in report
    output = tmp_path / "report.txt"
    output.write_text(report)
    assert output.read_text() == report
    path.write_text("1 1\n2 0\n")
    bare = read_accflf_data(path, time=0, event=1)
    assert bare.names == ("X1", "X2")
    path.write_text("time status multi\n1 1 1.5\n")
    with pytest.raises(ValueError, match="multiplicity"):
        read_accflf_data(path)

    case = json.loads((Path(__file__).parent / "fixtures/accflf-marginal.json").read_text())
    covariates = np.array([[0.0], [0.0], [1.0], [2.0]])
    kwargs = dict(p=case["p"], q=case["q"], sigma=0.7, coefficients=[1, 0.4])
    marginal = accflf_marginal_survival(case["time"], covariates=covariates, **kwargs)
    assert_allclose(marginal, case["survival"], atol=1e-14, rtol=0)
    curves = [
        accflf_survival(case["time"], covariates=np.tile(row, (8, 1)), **kwargs)
        for row in covariates
    ]
    assert_allclose(marginal, np.mean(curves, axis=0), atol=1e-14)
    # Equal-row averaging retains duplicate covariate rows, not just unique values.
    assert not np.allclose(marginal, np.mean([curves[0], curves[2], curves[3]], axis=0))
    tiny = accflf_marginal_survival([1e-300, 1e200], covariates=covariates, log=True, **kwargs)
    assert np.all(np.isfinite(tiny)) and np.all(tiny <= 0)
    with pytest.raises(ArithmeticError, match="converge"):
        fit_accflf(selected.time, selected.event, p=1, q=0, max_iterations=1)
