from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.ipdfromkm_diagnostics import ipd_reconstruction_diagnostics

_FIXTURES = Path(__file__).parent / "fixtures"


def test_source_report_metrics_and_r_441_ks_defaults() -> None:
    with (_FIXTURES / "ipdfromkm-diagnostics-input.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    with (_FIXTURES / "ipdfromkm-diagnostics-summary.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        summaries = list(csv.DictReader(stream))

    for summary in summaries:
        case_rows = [row for row in rows if row["case"] == summary["case"]]
        observed = np.array(
            [float(row["observed"]) if row["observed"] else np.nan for row in case_rows]
        )
        reconstructed = np.array([float(row["reconstructed"]) for row in case_rows])
        expected_retained = np.array([row["retained"] == "TRUE" for row in case_rows])
        result = ipd_reconstruction_diagnostics(observed, reconstructed)

        assert result.n_points == int(summary["n"])
        assert result.omitted_points == int(summary["omitted"])
        np.testing.assert_array_equal(result.retained_indices, np.flatnonzero(expected_retained))
        expected_fitted = np.array(
            [float(row["fitted_rounded"]) for row in case_rows if row["fitted_rounded"]]
        )
        expected_difference = np.array(
            [float(row["difference_rounded"]) for row in case_rows if row["difference_rounded"]]
        )
        np.testing.assert_array_equal(result.rounded_reconstructed_survival, expected_fitted)
        np.testing.assert_array_equal(result.rounded_difference, expected_difference)
        for field in (
            "rmse",
            "mean_absolute_error",
            "legacy_signed_max_error",
            "max_absolute_error",
        ):
            assert getattr(result, field) == float(summary[field])
        assert result.ks_statistic == pytest.approx(float(summary["ks_statistic"]), abs=3e-15)
        assert math.isclose(
            result.ks_pvalue,
            float(summary["ks_pvalue"]),
            rel_tol=2e-8,
            abs_tol=1e-15,
        )
        if summary["case"] == "exact_separation_99":
            assert result.ks_pvalue == pytest.approx(2 / math.comb(198, 99), rel=1e-12, abs=1e-300)
        assert result.ks_method == ("exact" if summary["ks_exact"] == "TRUE" else "asymptotic")

    signed = next(row for row in summaries if row["case"] == "signed_error")
    assert float(signed["legacy_signed_max_error"]) == -0.1
    separated = next(row for row in summaries if row["case"] == "exact_separation")
    assert float(separated["ks_pvalue"]) == pytest.approx(2 / math.comb(10, 5))
    tied_tail = next(row for row in summaries if row["case"] == "tied_tail_exact")
    assert 0 < float(tied_tail["ks_pvalue"]) < 1


def test_diagnostics_reject_unpaired_or_nonprobability_inputs() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        ipd_reconstruction_diagnostics([0.8, 0.5], [0.7, 0.6, 0.5])
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        ipd_reconstruction_diagnostics([0.8, np.inf], [0.7, 0.6])
    with pytest.raises(ValueError, match="at least two complete"):
        ipd_reconstruction_diagnostics([np.nan, 0.5], [0.4, np.nan])
