"""Compare constrained GAO posterior sampling with independent base-R quadrature."""

from __future__ import annotations

import csv
import json
import resource
import time
from pathlib import Path

import numpy as np

from mdanderson_stats.hierarchical_binomial import summarize_chains
from mdanderson_stats.u2oet_gao2010_fit import (
    fit_u2oet_gao2010,
    u2oet_gao2010_parameter_names,
)


def main() -> None:
    start = time.monotonic()
    path = Path(__file__).resolve().parents[1] / "tests/fixtures/u2oet-gao2010-posterior.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    references = {
        case: {row["metric"]: float(row["value"]) for row in rows if row["case"] == case}
        for case in {row["case"] for row in rows}
    }
    names = u2oet_gao2010_parameter_names(2, 2)
    results = []
    checks = 0
    for index, case in enumerate(
        ("restricted_prior", "restricted_posterior", "toxicity_only", "uniform_rho_posterior")
    ):
        mu = np.zeros(len(names) - 1)
        sd = np.zeros_like(mu)
        complete = np.zeros((2, 2, 2, 2), dtype=int)
        partial = np.zeros((2, 2, 2), dtype=int)
        if case == "uniform_rho_posterior":
            mu[[0, 1, 6, 7]] = -np.log(2)
            complete[:] = [[2, 1], [1, 3]]
            reference = references[case]
            fixed_association = None
        else:
            offset = 6 if case == "toxicity_only" else 0
            mu[offset : offset + 6] = [0.2, 0.4, 0.3, 0.2, 0, -1]
            sd[offset] = 0.6
            sd[offset + 5] = 0.65
            successes = [1, 3, 2, 4]
            failures = [3, 1, 2, 1]
            for pair, (i, j) in enumerate(np.ndindex((2, 2))):
                if case == "restricted_posterior":
                    complete[i, j, 1, 1] = successes[pair]
                    complete[i, j, 0, 0] = failures[pair]
                elif case == "toxicity_only":
                    partial[i, j] = [failures[pair], successes[pair]]
            reference = references["restricted_posterior" if case == "toxicity_only" else case]
            fixed_association = 0.0
        fitted = fit_u2oet_gao2010(
            [0, 2],
            [0, 2],
            complete,
            toxicity_only=partial,
            prior_mean=mu,
            prior_sd=sd,
            fixed_association=fixed_association,
            draws=4096,
            warmup=512,
            chains=2,
            rng=np.random.default_rng(78401 + index),
        )
        if case == "uniform_rho_posterior":
            rho = fitted.parameters[..., -1]
            features = np.stack(
                (
                    rho,
                    (rho - reference["association_mean"]) ** 2,
                    fitted.joint[..., 0, 0, 0, 0],
                    fitted.joint[..., 0, 0, 0, 1],
                ),
                axis=-1,
            )
            metrics = ["association_mean", "association_variance", "joint_same", "joint_different"]
        else:
            alpha, gamma = fitted.parameters[..., offset], fitted.parameters[..., offset + 5]
            a = alpha - reference["alpha_mean"]
            g = gamma - reference["gamma_mean"]
            probability = (
                fitted.joint[..., 1].sum(axis=-1)
                if case == "toxicity_only"
                else fitted.joint[..., 1, :].sum(axis=-1)
            )
            features = np.concatenate(
                (
                    np.stack((alpha, gamma, a * a, g * g, a * g), axis=-1),
                    probability.reshape(2, 4096, 4),
                ),
                axis=-1,
            )
            metrics = [
                "alpha_mean",
                "gamma_mean",
                "alpha_variance",
                "gamma_variance",
                "alpha_gamma_covariance",
                *[f"efficacy.{k}" for k in range(4)],
            ]
        expected = np.array([reference[name] for name in metrics])
        summary = summarize_chains(features)
        z = np.abs(summary.mean - expected) / summary.batch_mean_mcse
        assert np.all(np.isfinite(z)) and np.max(z) < 6, (case, metrics, z)
        assert np.max(summary.split_rhat) < 1.05, (case, summary.split_rhat)
        checks += len(metrics)
        results.append(
            {
                "case": case,
                "summary_checks": len(metrics),
                "maximum_mcse_multiple": float(np.max(z)),
                "maximum_split_rhat": float(np.max(summary.split_rhat)),
                "likelihood_evaluations": fitted.likelihood_evaluations,
            }
        )
        del fitted, features
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(
        json.dumps(
            {
                "posterior_summary_checks": checks,
                "cases": results,
                "elapsed_seconds": round(time.monotonic() - start, 3),
                "peak_mib": round(usage.ru_maxrss / 1024**2, 2),
                "swaps": usage.ru_nswap,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
