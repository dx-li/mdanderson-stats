"""Compare supplied-pair WFMM calculations with independent base-R fixtures."""

from __future__ import annotations

import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats import (
    wfmm_basis,
    wfmm_covariance,
    wfmm_inverse,
    wfmm_summarize,
    wfmm_summarize_covariance,
    wfmm_transform,
)


def main() -> None:
    start = time.monotonic()
    fixture_root = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    tables: dict[str, list[dict[str, str]]] = {}
    for name in ("matrices", "curves", "covariance", "summaries"):
        with (fixture_root / f"wfmm-custom-{name}.csv").open(newline="") as stream:
            tables[name] = list(csv.DictReader(stream))
    cases = sorted({row["case"] for row in tables["matrices"]})
    max_scaled_error = 0.0
    values_checked = 0

    def compare(actual: np.ndarray, expected: np.ndarray) -> None:
        nonlocal max_scaled_error, values_checked
        assert_allclose(actual, expected, rtol=3e-13, atol=2e-14)
        max_scaled_error = max(
            max_scaled_error,
            float(np.max(np.abs(actual - expected) / (1 + np.abs(expected)))),
        )
        values_checked += expected.size

    for case in cases:
        rows = [row for row in tables["matrices"] if row["case"] == case]
        size = 1 + max(int(row["row"]) for row in rows)
        analysis, synthesis = np.empty((size, size)), np.empty((size, size))
        for row in rows:
            index = int(row["row"]), int(row["column"])
            analysis[index], synthesis[index] = float(row["analysis"]), float(row["synthesis"])
        basis = wfmm_basis(
            size, transform="custom", analysis_matrix=analysis, synthesis_matrix=synthesis
        )
        observed = np.empty((2, 4, 2, size))
        coefficients = np.empty_like(observed)
        reconstructed = np.empty_like(observed)
        for row in tables["curves"]:
            if row["case"] != case:
                continue
            index = tuple(int(row[key]) for key in ("chain", "draw", "effect", "coordinate"))
            observed[index] = float(row["observed"])
            coefficients[index] = float(row["coefficient"])
            reconstructed[index] = float(row["reconstructed"])
        transformed = wfmm_transform(observed.reshape(-1, size), basis)
        compare(transformed.coefficients, coefficients.reshape(-1, size))
        compare(
            wfmm_inverse(coefficients.reshape(-1, size), basis), reconstructed.reshape(-1, size)
        )
        summary = wfmm_summarize(
            coefficients, basis, confidence=0.9, quantiles=(0.1, 0.5, 0.9), retain_curves=True
        )
        assert summary.curves is not None
        compare(summary.curves, reconstructed)
        expected_summary = {
            field: np.empty((2, size))
            for field in (
                "mean",
                "sd",
                "lower_quantile",
                "median",
                "upper_quantile",
                "simultaneous_lower",
                "simultaneous_upper",
                "critical",
            )
        }
        for row in tables["summaries"]:
            if row["case"] == case:
                index = int(row["effect"]), int(row["coordinate"])
                for field, target in expected_summary.items():
                    target[index] = float(row[field])
        for field, actual in (
            ("mean", summary.mean),
            ("sd", summary.standard_deviation),
            ("lower_quantile", summary.quantiles[0]),
            ("median", summary.quantiles[1]),
            ("upper_quantile", summary.quantiles[2]),
            ("simultaneous_lower", summary.simultaneous_lower),
            ("simultaneous_upper", summary.simultaneous_upper),
        ):
            compare(actual, expected_summary[field])
        compare(summary.simultaneous_critical_value, expected_summary["critical"][:, 0])
        variances, covariance = np.empty(size), np.empty((size, size))
        for row in tables["covariance"]:
            if row["case"] == case:
                i, j = int(row["row"]), int(row["column"])
                variances[i], covariance[i, j] = float(row["variance"]), float(row["covariance"])
        compare(wfmm_covariance(variances, basis), covariance)
        variance_draws = np.broadcast_to(variances, (2, 4, 1, size)).copy()
        covariance_summary = wfmm_summarize_covariance(
            variance_draws, basis, include_covariance=True, retain_covariance_draws=True
        )
        assert covariance_summary.covariance_mean is not None
        assert covariance_summary.covariance_draws is not None
        compare(covariance_summary.covariance_mean[0], covariance)
        compare(covariance_summary.variance_function_mean[0], np.diag(covariance))
        compare(
            covariance_summary.covariance_draws, np.broadcast_to(covariance, (2, 4, 1, size, size))
        )
        assert not covariance_summary.covariance_draws.flags.writeable
        assert not summary.curves.flags.writeable

    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "values_checked": values_checked,
                "maximum_scaled_error": max_scaled_error,
                "elapsed_seconds": round(time.monotonic() - start, 3),
                "peak_mib": round(usage.ru_maxrss / 1024**2, 2),
                "swaps": usage.ru_nswap,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
