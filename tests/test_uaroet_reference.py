"""Independent uniform-quantile copula and reduced posterior references."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.hierarchical_binomial import summarize_chains
from mdanderson_stats.uaroet import uaroet_probabilities
from mdanderson_stats.uaroet_fit import fit_uaroet


def _rows(name):
    with (Path(__file__).parent / "fixtures" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_probability_model_against_independent_r():
    rows = _rows("uaroet-probabilities.csv")
    logits = _rows("uaroet-logits.csv")
    for scenario in dict.fromkeys(row["scenario"] for row in rows):
        selected = [row for row in rows if row["scenario"] == scenario]
        doses = 1 + max(int(row["dose"]) for row in selected)
        matrices = []
        for endpoint in ("efficacy", "toxicity"):
            group = [
                row for row in logits if row["scenario"] == scenario and row["endpoint"] == endpoint
            ]
            matrices.append(np.array([float(row["logit"]) for row in group]).reshape(doses, -1))
        fit = uaroet_probabilities(*matrices, association=float(selected[0]["rho"]))
        expected = np.array([float(row["joint"]) for row in selected]).reshape(fit.joint.shape)
        np.testing.assert_allclose(fit.joint, expected, rtol=2e-9, atol=3e-12, err_msg=scenario)
        for actual, key, axis in (
            (fit.efficacy, "efficacy_marginal", 2),
            (fit.toxicity, "toxicity_marginal", 1),
        ):
            reference = np.array([float(row[key]) for row in selected]).reshape(fit.joint.shape)
            np.testing.assert_allclose(actual, reference.take(0, axis=axis), atol=2e-14, rtol=2e-13)
        if scenario == "published":
            utility = [[50, 25, 10, 0], [85, 50, 15, 5], [92, 60, 20, 7], [100, 75, 25, 10]]
            expected_utility = np.array([float(row["mean_utility"]) for row in selected])
            expected_utility = expected_utility.reshape(fit.joint.shape)[:, 0, 0]
            np.testing.assert_allclose(
                fit.expected_utility(utility), expected_utility, atol=2e-9, rtol=0
            )


def test_independence_posterior_against_direct_quadrature():
    reference = _rows("uaroet-posterior.csv")
    fit = fit_uaroet(
        [[[6, 1], [2, 3]]],
        prior_mean=[float(row["prior_mean"]) for row in reference],
        prior_sd=[float(row["prior_sd"]) for row in reference],
        association=0,
        draws=1200,
        warmup=400,
        chains=2,
        max_evaluations=200_000,
        rng=np.random.default_rng(92692),
    )
    summary = fit.parameter_summary
    assert np.max(summary.split_rhat) < 1.05
    for index, row in enumerate(reference):
        assert abs(summary.mean[index] - float(row["theta_mean"])) < (
            6 * summary.batch_mean_mcse[index] + 0.003
        )
        assert abs(summary.standard_deviation[index] - float(row["theta_sd"])) < 0.08
    for endpoint, axis in (("efficacy", -1), ("toxicity", -2)):
        probability = fit.joint.sum(axis=axis)[..., 0, 1]
        summary = summarize_chains(probability)
        row = next(row for row in reference if row["endpoint"] == endpoint)
        assert abs(float(summary.mean) - float(row["response_mean"])) < (
            6 * float(summary.batch_mean_mcse) + 0.002
        )
