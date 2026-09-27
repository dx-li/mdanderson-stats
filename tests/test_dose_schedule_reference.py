"""Independent integrated-hazard, prior and reduced-posterior references."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.dose_schedule import (
    DoseSchedulePatient,
    dose_schedule_cumulative_hazard,
    dose_schedule_hazard,
    dose_schedule_patient_loglikelihood,
)
from mdanderson_stats.dose_schedule_fit import fit_dose_schedule
from mdanderson_stats.dose_schedule_prior import DoseSchedulePrior, dose_schedule_moment_prior
from mdanderson_stats.hierarchical_binomial import summarize_chains


def _rows(filename):
    with (Path(__file__).parent / "fixtures" / filename).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _vector(value, dtype=float):
    return np.array(value.split("|"), dtype=dtype)


def test_hazard_and_actual_history_against_r_integration():
    for row in _rows("dose-schedule-hazards.csv"):
        args = [float(row[key]) for key in ("elapsed", "area", "peak", "tail")]
        actual = [dose_schedule_hazard(*args), dose_schedule_cumulative_hazard(*args)]
        expected = [float(row[key]) for key in ("hazard", "cumulative")]
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-15)

    for row in _rows("dose-schedule-histories.csv"):
        patient = DoseSchedulePatient(
            float(row["time"]),
            bool(int(row["event"])),
            _vector(row["administration_times"]),
            _vector(row["dose_indices"], dtype=int),
        )
        actual = dose_schedule_patient_loglikelihood(
            patient, *[_vector(row[key]) for key in ("area", "peak", "tail")]
        )
        np.testing.assert_allclose(
            actual,
            float(row["log_likelihood"]),
            rtol=2e-12,
            atol=1e-14,
            err_msg=row["case"],
        )


def test_moment_prior_against_published_example_and_r():
    rows = _rows("dose-schedule-prior.csv")
    prior = dose_schedule_moment_prior(
        [float(row["probability"]) for row in rows],
        int(rows[0]["administrations"]),
        [float(row["peak_mean"]) for row in rows],
        [float(row["tail_mean"]) for row in rows],
        lambda1=float(rows[0]["lambda_area"]),
        lambda2=float(rows[0]["lambda_time"]),
    )
    expected = np.array(
        [
            [float(row[key]) for key in ("log_area_mean", "log_peak_mean", "log_tail_mean")]
            for row in rows
        ]
    )
    np.testing.assert_allclose(prior.mean.reshape(-1, 3), expected, rtol=0, atol=2e-14)
    np.testing.assert_allclose(prior.sd**2, float(rows[0]["log_variance"]), atol=2e-14)


def test_reduced_posterior_against_direct_r_integration():
    row = _rows("dose-schedule-posterior.csv")[0]
    patients = [
        DoseSchedulePatient(time, event, times, np.zeros(len(times), dtype=int))
        for time, event, times in (
            (1, True, [0]),
            (3, True, [0, 1]),
            (4, False, [0, 2]),
            (7, False, [0, 1, 4]),
        )
    ]
    prior = DoseSchedulePrior(
        [float(row["log_area_prior_mean"]), np.log(2), np.log(3)],
        [float(row["log_area_prior_sd"]), 0, 0],
        dose_count=1,
    )
    fit = fit_dose_schedule(
        patients,
        prior,
        [[0], [0, 1]],
        horizon=10,
        draws=1200,
        warmup=400,
        chains=2,
        rng=np.random.default_rng(92775),
    )
    quantities = (
        (fit.log_parameters[..., 0], "log_area_mean", "log_area_sd"),
        (fit.areas[..., 0], "area_mean", "area_sd"),
        (fit.regimen_risk[..., 0, 1], "two_administration_risk", None),
        (
            (fit.regimen_risk[..., 0, 1] > float(row["toxicity_limit"])).astype(float),
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
