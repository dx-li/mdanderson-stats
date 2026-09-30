"""Compare the original GAO model with independent base-R probability references."""

from __future__ import annotations

import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats.u2oet_gao2010 import (
    U2OETGAO2010Marginal,
    u2oet_gao2010_probabilities,
)


def main() -> None:
    start = time.monotonic()
    fixture_root = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    tables: dict[str, list[dict[str, str]]] = {}
    for name in ("coefficients", "doses", "probabilities", "partial", "utilities", "likelihood"):
        with (fixture_root / f"u2oet-gao2010-{name}.csv").open(newline="") as stream:
            tables[name] = list(csv.DictReader(stream))
    cells = likelihoods = utilities = 0
    largest_error = 0.0

    def compare(actual: np.ndarray | float, expected: np.ndarray | float) -> None:
        nonlocal largest_error
        assert_allclose(actual, expected, atol=3e-12, rtol=2e-12)
        a, b = np.asarray(actual), np.asarray(expected)
        finite = np.isfinite(b)
        if np.any(finite):
            largest_error = max(largest_error, float(np.max(np.abs(a[finite] - b[finite]))))

    for case in sorted({row["scenario"] for row in tables["coefficients"]}):
        doses = []
        for agent in (1, 2):
            rows = sorted(
                (
                    row
                    for row in tables["doses"]
                    if row["scenario"] == case and int(row["agent"]) == agent
                ),
                key=lambda row: int(row["index"]),
            )
            doses.append(np.array([float(row["value"]) for row in rows]))
        margins = []
        for endpoint in ("e", "t"):
            rows = sorted(
                (
                    row
                    for row in tables["coefficients"]
                    if row["scenario"] == case and row["endpoint"] == endpoint
                ),
                key=lambda row: int(row["threshold"]),
            )
            margins.append(
                U2OETGAO2010Marginal(
                    [[float(row[f"intercept{j}"]) for j in (1, 2)] for row in rows],
                    [[float(row[f"slope{j}"]) for j in (1, 2)] for row in rows],
                    float(rows[0]["lambda"]),
                    float(rows[0]["gamma"]),
                )
            )
        for rho in (-1.0, -0.65, 0.0, 0.55, 1.0):
            result = u2oet_gao2010_probabilities(
                *doses, efficacy=margins[0], toxicity=margins[1], association=rho
            )
            expected = np.empty_like(result.joint)
            expected_e = np.empty_like(result.log_efficacy)
            expected_t = np.empty_like(result.log_toxicity)
            counts = np.zeros_like(expected)
            partial = np.zeros_like(expected_t)
            for row in tables["probabilities"]:
                if row["scenario"] == case and float(row["association"]) == rho:
                    i, j, e, t = (
                        int(row[key]) for key in ("dose1", "dose2", "efficacy", "toxicity")
                    )
                    expected[i, j, e, t] = float(row["joint_probability"])
                    expected_e[i, j, e] = float(row["efficacy_probability"])
                    expected_t[i, j, t] = float(row["toxicity_probability"])
                    counts[i, j, e, t] = int(row["count"])
            for row in tables["partial"]:
                if row["scenario"] == case and float(row["association"]) == rho:
                    partial[tuple(int(row[key]) for key in ("dose1", "dose2", "toxicity"))] = int(
                        row["count"]
                    )
            compare(result.joint, expected)
            compare(np.exp(result.log_efficacy), expected_e)
            compare(np.exp(result.log_toxicity), expected_t)
            cells += expected.size
            row = next(
                row
                for row in tables["likelihood"]
                if row["scenario"] == case and float(row["association"]) == rho
            )
            for actual, key in (
                (result.loglikelihood(counts), "complete"),
                (
                    result.loglikelihood(np.zeros_like(counts), toxicity_only=partial),
                    "toxicity_only",
                ),
                (result.loglikelihood(counts, toxicity_only=partial), "combined"),
            ):
                compare(actual, float(row[key]))
                likelihoods += 1
            utility = (
                20 + 30 * np.arange(expected.shape[-2])[:, None] - 5 * np.arange(expected.shape[-1])
            )
            expected_utility = np.empty(expected.shape[:2])
            for row in tables["utilities"]:
                if row["scenario"] == case and float(row["association"]) == rho:
                    expected_utility[int(row["dose1"]), int(row["dose2"])] = float(
                        row["expected_utility"]
                    )
            compare(result.expected_utility(utility), expected_utility)
            utilities += expected_utility.size
            if rho == 0.55:
                translated = u2oet_gao2010_probabilities(
                    doses[0] + 16,
                    doses[1] + 32,
                    efficacy=margins[0],
                    toxicity=margins[1],
                    association=rho,
                )
                compare(translated.joint, result.joint)
                for scale in (1e-150, 1e150):
                    scaled_margins = [
                        U2OETGAO2010Marginal(m.intercepts, m.slopes / scale, m.lambda_, m.gamma)
                        for m in margins
                    ]
                    rescaled = u2oet_gao2010_probabilities(
                        doses[0] * scale,
                        doses[1] * scale,
                        efficacy=scaled_margins[0],
                        toxicity=scaled_margins[1],
                        association=rho,
                    )
                    compare(rescaled.joint, result.joint)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(
        json.dumps(
            {
                "cases": 4,
                "joint_cells": cells,
                "likelihood_checks": likelihoods,
                "utility_checks": utilities,
                "maximum_absolute_error": largest_error,
                "translation_checks": 4,
                "unit_rescaling_checks": 8,
                "elapsed_seconds": round(time.monotonic() - start, 3),
                "peak_mib": round(usage.ru_maxrss / 1024**2, 2),
                "swaps": usage.ru_nswap,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
