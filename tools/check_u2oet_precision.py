"""Check U2OET precision diagnostics against independent base-R moments."""

from __future__ import annotations

import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats import summarize_chains
from mdanderson_stats.u2oet_adaptive_precision import _diagnostics


def main() -> None:
    start = time.monotonic()
    fixture_root = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    tables: dict[str, list[dict[str, str]]] = {}
    for name in ("traces", "reference", "rhat"):
        with (fixture_root / f"u2oet-precision-{name}.csv").open(newline="") as stream:
            tables[name] = list(csv.DictReader(stream))
    checked = 0
    max_error = 0.0

    def parse(value: str) -> float:
        return np.nan if value == "NA" else float(value)

    def compare(actual: np.ndarray, expected: np.ndarray) -> None:
        nonlocal checked, max_error
        assert_allclose(actual, expected, rtol=3e-13, atol=2e-14, equal_nan=True)
        finite = np.isfinite(expected)
        if np.any(finite):
            max_error = max(max_error, float(np.max(np.abs(actual[finite] - expected[finite]))))
        checked += expected.size

    for case in sorted({row["case"] for row in tables["traces"]}):
        rows = [row for row in tables["traces"] if row["case"] == case]
        draws = 1 + max(int(row["draw"]) for row in rows)
        trace = np.empty((2, draws, 4))
        for row in rows:
            trace[int(row["chain"]), int(row["draw"])] = [
                float(row[f"corner{index}"]) for index in range(4)
            ]
        expected_sd, expected_mcse, expected_ratio = (np.empty((2, 4)) for _ in range(3))
        for row in tables["reference"]:
            if row["case"] == case:
                index = int(row["chain"]), int(row["corner"])
                expected_sd[index] = parse(row["posterior_sd"])
                expected_mcse[index] = parse(row["mcse"])
                expected_ratio[index] = parse(row["ratio"])
        expected_rhat = np.empty(4)
        for row in tables["rhat"]:
            if row["case"] == case:
                expected_rhat[int(row["corner"])] = parse(row["split_rhat"])
        for scale in (1e-200, 1.0, 1e200):
            scaled = trace * scale
            sd, mcse, ratio = _diagnostics(scaled)
            compare(sd / scale, expected_sd)
            compare(mcse / scale, expected_mcse)
            compare(ratio, expected_ratio)
            # Match the controller's bounded one-corner-at-a-time diagnostic.
            actual_rhat = np.empty(4)
            for corner in range(4):
                normalized = scaled[:, :, corner] / np.max(scaled[:, :, corner])
                actual_rhat[corner] = summarize_chains(normalized).split_rhat
            compare(actual_rhat, expected_rhat)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(
        json.dumps(
            {
                "trace_cases": 3,
                "scales": 3,
                "diagnostics_checked": checked,
                "maximum_absolute_error_after_scale_normalization": max_error,
                "elapsed_seconds": round(time.monotonic() - start, 3),
                "peak_mib": round(usage.ru_maxrss / 1024**2, 2),
                "swaps": usage.ru_nswap,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
