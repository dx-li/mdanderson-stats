"""Compare bounded GAO posterior draws with independent base-R references."""

import csv
import json
import resource
import time
from pathlib import Path

import numpy as np

from mdanderson_stats.hierarchical_binomial import summarize_chains
from mdanderson_stats.u2oet_gao_fit import fit_u2oet_gao

root = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((root / "tests/fixtures/u2oet-gao-posterior.csv").open()))
counts = np.zeros((2, 2, 2, 2))
partial = np.zeros((2, 2, 2))
for i, j, e, t in np.ndindex(counts.shape):
    counts[i, j, e, t] = ((i + 1) + 2 * (j + 1) + (e + 1) + (t + 1)) % 3
for i, j in np.ndindex((2, 2)):
    partial[i, j] = [(i + 1) % 2, (j + 1) % 2]
start = time.monotonic()
checks = []
for case, seed in [("efficacy_intercept", 77031), ("fisher_z", 77032)]:
    reference = {r["metric"]: float(r["value"]) for r in rows if r["case"] == case}
    mu = np.array(
        [-1, 0.3, 0.2, -0.1, 0, -0.4, -1.2, -0.05, 0.15, 0, np.log(0.4), np.arctanh(0.55)]
    )
    sd = np.zeros(12)
    free = 0 if case == "efficacy_intercept" else 11
    sd[free] = 0.5 if free == 0 else 0.2
    if free == 11:
        mu[11] = np.arctanh(0.3)
    initial = np.tile(mu, (2, 1))
    initial[:, free] += [-sd[free], sd[free]]
    fitted = fit_u2oet_gao(
        [1, 3],
        [2, 5],
        counts,
        toxicity_only=partial,
        prior_mean=mu,
        prior_sd=sd,
        draws=1500,
        warmup=300,
        chains=2,
        initial=initial,
        rng=np.random.default_rng(seed),
    )
    theta = fitted.parameters[:, :, free]
    metrics = np.concatenate(
        (
            theta[:, :, None],
            ((theta - reference["mean"]) ** 2)[:, :, None],
            np.tanh(fitted.parameters[:, :, -1:]),
            fitted.joint.reshape(2, 1500, 16),
        ),
        axis=-1,
    )
    expected = np.array(
        [
            reference["mean"],
            reference["variance"],
            reference["mean_association"],
            *[reference[f"joint.{i}.{j}.{e}.{t}"] for i, j, e, t in np.ndindex(counts.shape)],
        ]
    )
    summary = summarize_chains(metrics)
    error = np.abs(summary.mean - expected)
    assert np.all(error <= 5 * summary.batch_mean_mcse + 1e-9), (
        case,
        error,
        summary.batch_mean_mcse,
    )
    variable = summary.batch_mean_mcse > 1e-10
    assert np.nanmax(summary.split_rhat[variable]) < 1.05
    check = dict(
        case=case,
        metrics=len(expected),
        maximum_error_mcse=float(np.max(error[variable] / summary.batch_mean_mcse[variable])),
        maximum_split_rhat=float(np.nanmax(summary.split_rhat[variable])),
        parameter_mean=float(summary.mean[0]),
        reference_mean=reference["mean"],
        evaluations=fitted.likelihood_evaluations,
        work=fitted.likelihood_work_units,
    )
    checks.append(check)
    print(json.dumps(check), flush=True)
usage = resource.getrusage(resource.RUSAGE_SELF)
print(
    json.dumps(
        dict(
            checks=checks,
            elapsed_seconds=time.monotonic() - start,
            peak_mib=usage.ru_maxrss / 1024**2,
            swaps=usage.ru_nswap,
        ),
        indent=2,
    )
)
