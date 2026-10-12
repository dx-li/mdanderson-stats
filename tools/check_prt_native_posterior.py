"""Independent tensor quadrature and Python/native posterior comparison for two-dose pilots."""

import json
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, logsumexp, roots_hermitenorm

from mdanderson_stats import fit_prt_model

cases = json.loads(
    (Path(__file__).resolve().parents[1] / "tests/fixtures/prt-desktop-posterior.json").read_text()
)["cases"]
results = []
for index, case in enumerate(cases[:2]):
    C, D = case["intervals"], case["doses"]
    survived, events = np.zeros((C, D)), np.zeros((C, D))
    for dose, interval, censored in case["rows"]:
        survived[: interval - (1 - censored), dose - 1] += 1
        if not censored:
            events[interval - 1, dose - 1] += 1

    def integrate(nodes):
        z, w = roots_hermitenorm(nodes)
        w /= np.sqrt(2 * np.pi)
        b1 = case["prior_mean"] + np.sqrt(case["prior_variance"]) * np.repeat(z, nodes)
        b2 = b1 + np.sqrt(case["prior_variance"]) * np.tile(z, nodes)
        logits = np.column_stack((b1, b2))
        prior = np.repeat(w, nodes) * np.tile(w, nodes)
        positive = prior > 0
        logits, prior = logits[positive], prior[positive]
        log_survival = log_ndtr(-logits)
        log_event = log_ndtr(logits)
        means = []
        for s, e in zip(survived, events, strict=True):
            weights = np.log(prior) + log_survival @ s + log_event @ e
            weights = np.exp(weights - logsumexp(weights))
            means.append(weights @ np.exp(log_survival))
        return 1 - np.prod(means, axis=0)

    a, reference = integrate(100), integrate(160)
    assert np.max(np.abs(a - reference)) < 1e-10
    fit = fit_prt_model(
        survived,
        events,
        prior_mean=case["prior_mean"],
        prior_variance=case["prior_variance"],
        draws=4000,
        warmup=2000,
        chains=4,
        rng=np.random.default_rng(69092 + index),
        max_likelihood_evaluations=2_000_000,
    )
    mean = fit.conditional_toxicity[:, :, 0].mean((0, 1))
    mcse = fit.risk_summary.batch_mean_mcse[0]
    native_mean, native_mcse = np.array(case["mean"]), np.array(case["batch_mcse"])
    assert np.all(abs(mean - reference) <= 6 * mcse)
    assert np.all(abs(native_mean - reference) <= 6 * native_mcse)
    assert np.max(fit.risk_summary.split_rhat) < 1.02
    results.append(
        dict(
            case=index,
            reference=reference.tolist(),
            python_mean=mean.tolist(),
            python_mcse=mcse.tolist(),
            native_mean=native_mean.tolist(),
            native_mcse=native_mcse.tolist(),
            maximum_python_rhat=float(np.max(fit.risk_summary.split_rhat)),
            quadrature_node_difference=float(np.max(abs(a - reference))),
        )
    )
print(
    json.dumps(
        dict(
            scope="two-dose three-interval synthetic pre-projection posterior pilots",
            results=results,
            clinical_validation=False,
        ),
        indent=2,
    )
)
