"""Check predictive propagation against independent R finite-mixture moments."""

from __future__ import annotations

import csv
import json
import resource
import time
from pathlib import Path

import numpy as np

from mdanderson_stats.hierarchical_binomial import summarize_chains
from mdanderson_stats.wfmm_basis import wfmm_basis
from mdanderson_stats.wfmm_model import WFMMCoefficientFit
from mdanderson_stats.wfmm_posterior import wfmm_summarize
from mdanderson_stats.wfmm_prediction import wfmm_predict_coefficients


def main() -> None:
    start = time.monotonic()
    root = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

    def table(name: str) -> list[dict[str, str]]:
        with (root / f"wfmm-prediction-{name}.csv").open(newline="") as stream:
            return list(csv.DictReader(stream))

    supplied = {key: np.empty((8, 2, 2)) for key in ("b", "u", "q", "s")}
    for row in table("draws"):
        supplied[row["kind"]][int(row["draw"]), int(row["row"]), int(row["coefficient"])] = float(
            row["value"]
        )
    design = table("design")
    x, old, new = (
        np.array([[float(row[f"{prefix}{k}"]) for k in (1, 2)] for row in design])
        for prefix in ("x", "existing", "new")
    )
    residual = np.array([int(row["residual_group"]) for row in design])
    synthesis = np.array([[1, 0.5], [-0.25, 2]])
    basis = wfmm_basis(
        2,
        transform="custom",
        analysis_matrix=np.linalg.solve(synthesis, np.eye(2)),
        synthesis_matrix=synthesis,
    )

    def fit(repeats: int) -> WFMMCoefficientFit:
        b, u, q, s = (
            np.tile(supplied[key][None], (2, repeats, 1, 1)) for key in ("b", "u", "q", "s")
        )
        return WFMMCoefficientFit(
            coefficients=b,
            inclusion_indicators=np.ones(b.shape, dtype=bool),
            random_variances=q,
            residual_variances=s,
            random_effects=u,
            log_likelihood=np.zeros(b.shape[:2]),
            summary=summarize_chains(b),
            random_strata=np.array([0, 1]),
            residual_strata=np.array([0, 1, 0]),
            coefficient_partition=np.zeros(2, dtype=int),
            coefficient_scale=np.zeros(2, dtype=int),
            warmup=0,
            estimate_variances=True,
            variance_acceptance=np.zeros((2, 4, 2)),
            likelihood_evaluations=0,
        )

    conditional = table("conditional")
    references = table("moments")
    exact_checks = moment_checks = 0
    largest_z = 0.0
    cases = []
    for target in ("population", "existing", "new_latent", "replicate"):
        stochastic = target in ("new_latent", "replicate")
        fitted = fit(512 if stochastic else 1)
        coefficients = wfmm_predict_coefficients(
            fitted,
            x,
            existing_random_design=None if target == "population" else old,
            new_random_design=new if stochastic else None,
            new_random_strata=[1, 0] if stochastic else None,
            include_residual=target == "replicate",
            residual_strata=residual if target == "replicate" else None,
            rng=np.random.default_rng(47021 if target == "replicate" else 47020),
        )
        assert not coefficients.flags.writeable
        summary = wfmm_summarize(coefficients, basis, retain_curves=True)
        assert summary.curves is not None
        sample = summary.curves.reshape(-1, 6)
        if not stochastic:
            expected = np.empty((8, 3, 2))
            for row in conditional:
                if row["target"] == target:
                    expected[int(row["draw"]), int(row["row"]), int(row["time"])] = float(
                        row["value"]
                    )
            np.testing.assert_allclose(
                summary.curves, np.tile(expected[None], (2, 1, 1, 1)), atol=2e-14
            )
            exact_checks += summary.curves.size
        selected = [row for row in references if row["target"] == target]
        mean = np.array([float(row["value"]) for row in selected if row["metric"] == "mean"])
        centered = sample - mean
        case_z = 0.0
        for row in selected:
            i, j = int(row["i"]), int(row["j"])
            observations = (
                sample[:, i] if row["metric"] == "mean" else centered[:, i] * centered[:, j]
            )
            expected_value = float(row["value"])
            actual = float(observations.mean())
            if stochastic:
                se = float(observations.std(ddof=1) / np.sqrt(sample.shape[0]))
                z = abs(actual - expected_value) / se
                assert z < 6, (target, row, actual, se, z)
                case_z = max(case_z, z)
            else:
                np.testing.assert_allclose(actual, expected_value, atol=2e-14, rtol=2e-13)
            moment_checks += 1
        largest_z = max(largest_z, case_z)
        cases.append({"target": target, "draws": sample.shape[0], "maximum_mcse_multiple": case_z})
        del fitted, coefficients, summary, sample
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(
        json.dumps(
            {
                "conditional_values": exact_checks,
                "predictive_moments": moment_checks,
                "maximum_mcse_multiple": largest_z,
                "cases": cases,
                "elapsed_seconds": round(time.monotonic() - start, 3),
                "peak_mib": round(usage.ru_maxrss / 1024**2, 2),
                "swaps": usage.ru_nswap,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
