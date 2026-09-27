"""Independent R quadrature references for CiBolus, including difficult limits."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from mdanderson_stats.cibolus import (
    CiBolusObservation,
    CiBolusPrior,
    cibolus_loglikelihood,
    cibolus_predict,
    cibolus_response,
    cibolus_toxicity,
)
from mdanderson_stats.cibolus_fit import fit_cibolus
from mdanderson_stats.hierarchical_binomial import summarize_chains


def _rows(filename):
    with (Path(__file__).parent / "fixtures" / filename).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _vector(value):
    return np.array(value.split("|"), dtype=float)


def _observation(row):
    kind = "failure" if row["kind"] == "no_response" else row["kind"]
    timing = {}
    if kind == "exact":
        timing["time"] = float(row["upper"])
    elif kind == "interval":
        timing = {key: float(row[key]) for key in ("lower", "upper")}
    return CiBolusObservation(
        float(row["concentration"]),
        float(row["bolus"]),
        kind,
        bool(int(row["toxicity"])),
        **timing,
    )


def test_response_and_toxicity_against_independent_r_quadrature():
    for row in _rows("cibolus-probabilities.csv"):
        log_parameters = np.log(_vector(row["parameters"]))
        time, concentration, bolus = (float(row[key]) for key in ("time", "concentration", "bolus"))
        response = cibolus_response(time, concentration, bolus, log_parameters)
        actual = [response.bolus_probability, response.cdf, response.cumulative_hazard]
        expected = [
            float(row[key])
            for key in ("bolus_probability", "response_probability", "continuous_cumulative")
        ]
        if row["continuous_hazard"] != "NA":
            actual.append(response.continuous_hazard)
            expected.append(float(row["continuous_hazard"]))
        actual += [
            cibolus_toxicity(time, concentration, bolus, log_parameters),
            cibolus_toxicity(1, concentration, bolus, log_parameters, failure=True),
        ]
        expected += [float(row[key]) for key in ("toxicity", "failure_toxicity")]
        np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-12, err_msg=row["case"])


def test_joint_categories_and_mixed_observations_against_r():
    cases = defaultdict(list)
    for row in _rows("cibolus-joint.csv"):
        cases[row["case"]].append(row)
    for name, rows in cases.items():
        row = rows[0]
        prediction = cibolus_predict(
            np.log(_vector(row["parameters"])),
            [float(row["concentration"])],
            [float(row["bolus"])],
            _vector(row["endpoints"]),
            utility=np.zeros((len(rows), 2)),
        )
        expected = np.array(
            [[float(item[key]) for key in ("no_toxicity", "toxicity")] for item in rows]
        )
        np.testing.assert_allclose(
            prediction.joint[0, 0], expected, rtol=2e-9, atol=2e-12, err_msg=name
        )
        np.testing.assert_allclose(prediction.joint.sum(), 1, atol=2e-12)

    observations = []
    expected_total = 0.0
    for row in _rows("cibolus-likelihood.csv"):
        observation = _observation(row)
        log_parameters = np.log(_vector(row["parameters"]))
        expected = float(row["log_likelihood"])
        np.testing.assert_allclose(
            cibolus_loglikelihood([observation], log_parameters), expected, rtol=2e-9, atol=2e-12
        )
        observations.append(observation)
        expected_total += expected
    np.testing.assert_allclose(
        cibolus_loglikelihood(observations, log_parameters), expected_total, rtol=2e-9, atol=2e-12
    )


def test_reduced_posterior_against_direct_r_integration():
    row = _rows("cibolus-posterior.csv")[0]
    patients = _rows("cibolus-likelihood.csv")
    mean = np.log(_vector(patients[0]["parameters"]))
    mean[6] = float(row["log_beta0_prior_mean"])
    sd = np.zeros(11)
    sd[6] = float(row["log_beta0_prior_sd"])
    fit = fit_cibolus(
        [_observation(patient) for patient in patients],
        CiBolusPrior(mean, sd),
        [float(row["concentration"])],
        [float(row["bolus"])],
        [0.25, 0.5, 0.75, 1],
        utility=np.zeros((6, 2)),
        draws=1200,
        warmup=400,
        chains=2,
        rng=np.random.default_rng(92886),
    )
    quantities = (
        (fit.log_parameters[..., 6], "log_beta0_mean", "log_beta0_sd"),
        (np.exp(fit.log_parameters[..., 6]), "beta0_mean", "beta0_sd"),
        (fit.toxicity_at_one_response[..., 0, 0], "response_at_one_toxicity_mean", None),
        (
            (fit.toxicity_at_one_response[..., 0, 0] > float(row["toxicity_limit"])).astype(float),
            "overdose_probability",
            None,
        ),
    )
    for samples, mean_key, sd_key in quantities:
        summary = summarize_chains(samples)
        assert float(summary.split_rhat) < 1.05, mean_key
        assert abs(float(summary.mean) - float(row[mean_key])) < (
            6 * float(summary.batch_mean_mcse) + 0.003
        ), mean_key
        if sd_key:
            assert abs(float(summary.standard_deviation) - float(row[sd_key])) < 0.06, sd_key
