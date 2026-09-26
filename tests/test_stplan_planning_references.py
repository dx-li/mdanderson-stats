"""Independent R inverse solutions across STPLAN's 25 forward procedures."""

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.stplan_planning import STPLAN_METHODS, stplan_solve

# Each configuration corresponds to the same named case in the independent R
# harness. Formula evaluation and root finding are separate implementations.
CASES = {
    "normal_n": (
        "normal_one_sample_power",
        "sample_size",
        (2, 1000),
        {"difference": 0.5, "sd": 1},
        {},
    ),
    "normal_difference": (
        "normal_one_sample_power",
        "difference",
        (0, 3),
        {"sd": 1, "sample_size": 10},
        {},
    ),
    "normal_sd": (
        "normal_one_sample_power",
        "sd",
        (0.1, 10),
        {"difference": 0.5, "sample_size": 10},
        {},
    ),
    "normal_alpha": (
        "normal_one_sample_power",
        "alpha",
        (0.0001, 0.49),
        {"difference": 0.5, "sd": 1, "sample_size": 20},
        {},
    ),
    "normal_equal_n": (
        "normal_two_sample_power",
        ("n1", "n2"),
        (2, 1000),
        {"difference": 0.5, "sd": 1},
        {},
    ),
    "welch_n1": (
        "welch_two_sample_power",
        "n1",
        (2, 1000),
        {"difference": 0.5, "sd1": 1, "sd2": 2, "n2": 200},
        {},
    ),
    "lognormal_cv": (
        "lognormal_two_sample_power",
        "cv",
        (0.01, 2),
        {"mean1": 100, "mean2": 80, "n1": 100, "n2": 100},
        {},
    ),
    "exponential_ratio": ("exponential_one_sample_power", "ratio", (1, 5), {"sample_size": 30}, {}),
    "exponential_equal_n": (
        "exponential_two_sample_power",
        ("n1", "n2"),
        (2, 1000),
        {"mean1": 10, "mean2": 15},
        {},
    ),
    "correlation_n": (
        "correlation_one_sample_power",
        "sample_size",
        (4, 1000),
        {"null_correlation": 0, "alternative_correlation": 0.4},
        {},
    ),
    "correlation_alternative": (
        "correlation_one_sample_power",
        "alternative_correlation",
        (0.1, 0.95),
        {"null_correlation": 0.1, "sample_size": 50},
        {},
    ),
    "correlation_two_n": (
        "correlation_two_sample_power",
        "n1",
        (4, 1000),
        {"correlation1": 0.1, "correlation2": 0.5, "n2": 100},
        {},
    ),
    "arcsine_equal_n": (
        "arcsine_binomial_two_sample_power",
        ("n1", "n2"),
        (2, 1000),
        {"probability1": 0.2, "probability2": 0.4},
        {},
    ),
    "median_n": (
        "median_split_power",
        "total_size",
        (4, 2000),
        {"overall_probability": 0.3, "delta": 0.1},
        {},
    ),
    "historical_binomial_n": (
        "historical_binomial_power",
        "experimental_size",
        (2, 1000),
        {"control_probability": 0.2, "experimental_probability": 0.4, "control_size": 100},
        {},
    ),
    "responder_margin": (
        "responder_normal_approximation_power",
        "margin",
        (0, 0.9),
        {
            "conservative_probability": 0.3,
            "expensive_probability": 0.4,
            "conservative_size": 100,
            "expensive_size": 80,
        },
        {},
    ),
    "fisher_n": (
        "fisher_exact_approx_power",
        "sample_size",
        (6, 1000),
        {"probability1": 0.2, "probability2": 0.4},
        {},
    ),
    "exact_binomial_probability": (
        "exact_binomial_power",
        "alternative_probability",
        (0.2, 0.9),
        {"null_probability": 0.2, "sample_size": 20},
        {},
    ),
    "exact_binomial_n": (
        "exact_binomial_power",
        "sample_size",
        (1, 100),
        {"null_probability": 0.2, "alternative_probability": 0.4},
        {},
    ),
    "exact_poisson_rate": (
        "exact_poisson_power",
        "alternative_rate",
        (1, 4),
        {"null_rate": 1, "exposure": 10},
        {},
    ),
    "retention_n": (
        "retention_probability",
        "initial_size",
        (80, 140),
        {"minimum_remaining": 80, "dropout_rate": 0.05, "duration": 3},
        {},
    ),
    "retention_threshold": (
        "retention_probability",
        "minimum_remaining",
        (0, 100),
        {"initial_size": 100, "dropout_rate": 0.05, "duration": 3},
        {"integer_goal": "largest"},
    ),
    "unmatched_n": (
        "case_control_power",
        "n_cases",
        (2, 1000),
        {
            "exposure_frequency": 0.3,
            "disease_risk_exposed": 0.1,
            "disease_risk_unexposed": 0.05,
            "n_controls": 120,
        },
        {},
    ),
    "matched_exposed_risk": (
        "matched_case_control_power",
        "disease_risk_exposed",
        (0.05, 0.5),
        {"exposure_frequency": 0.3, "disease_risk_unexposed": 0.05, "n_pairs": 60},
        {},
    ),
    "matched_n": (
        "matched_case_control_power",
        "n_pairs",
        (2, 200),
        {"exposure_frequency": 0.3, "disease_risk_exposed": 0.1, "disease_risk_unexposed": 0.05},
        {},
    ),
    "poisson_two_rate": (
        "poisson_two_sample_power",
        "rate2",
        (1, 4),
        {"rate1": 1, "exposure1": 10, "exposure2": 10},
        {},
    ),
    "censored_one_accrual": (
        "censored_exponential_one_sample_power",
        "accrual_rate",
        (1, 30),
        {"null_mean": 10, "alternative_mean": 15, "accrual_duration": 12, "followup_duration": 6},
        {},
    ),
    "george_desu_accrual": (
        "george_desu_survival_power",
        "accrual_rate",
        (1, 100),
        {"mean1": 10, "mean2": 15, "accrual_duration": 12, "followup_duration": 6},
        {},
    ),
    "information_followup": (
        "information_survival_power",
        "followup_duration",
        (0, 100),
        {"mean1": 10, "mean2": 15, "accrual_rate": 20, "accrual_duration": 12},
        {},
    ),
    "historical_survival_hazard": (
        "historical_survival_power",
        "experimental_hazard",
        (0.01, 0.1),
        {
            "control_hazard": 0.1,
            "accrual_rate": 5,
            "accrual_duration": 12,
            "followup_duration": 6,
            "historical_deaths": 40,
            "historical_alive": 20,
            "control_allocation": 0.2,
        },
        {},
    ),
    "piecewise_accrual": (
        "piecewise_survival_power",
        "accrual_rate",
        (1, 100),
        {
            "hazard_before": 0.2,
            "hazard_after": 0.05,
            "change_time": 8,
            "hazard_ratio": 1.5,
            "accrual_duration": 10,
            "followup_duration": 5,
        },
        {},
    ),
    "binomial_k_total": (
        "binomial_k_sample_power",
        "sample_sizes",
        (1, 2000),
        {"probabilities": [0.1, 0.2, 0.3]},
        {"allocation_weights": [1, 2, 1]},
    ),
    "binomial_k_probability": (
        "binomial_k_sample_power",
        "probabilities",
        (0.2, 0.9),
        {"probabilities": [0.1, 0.2, 0.3], "sample_sizes": [20, 40, 20]},
        {"index": 2},
    ),
}


def _solve(case_id, target):
    method, compute, bounds, parameters, options = CASES[case_id]
    return stplan_solve(
        "stplan_" + method,
        compute=compute,
        target_power=target,
        bounds=bounds,
        parameters=parameters,
        **options,
    )


def test_inverse_designs_against_independent_r():
    path = Path(__file__).parent / "fixtures" / "stplan-planning-r.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        result = _solve(row["case_id"], float(row["target"]))
        np.testing.assert_allclose(result.value, float(row["value"]), rtol=2e-7, atol=2e-9)
        np.testing.assert_allclose(result.achieved_power, float(row["achieved"]), rtol=0, atol=1e-8)
        completed_power = STPLAN_METHODS[result.method].function(**result.inputs)
        np.testing.assert_allclose(completed_power, result.achieved_power, rtol=0, atol=1e-13)
        if row["previous_power"] != "NA":
            np.testing.assert_allclose(
                result.previous_power, float(row["previous_power"]), rtol=0, atol=1e-10
            )
            assert result.previous_power < result.target_power <= result.achieved_power


def test_inverse_designs_against_native_numeric_roots():
    path = Path(__file__).parent / "fixtures" / "stplan-inverse-native.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["case_id"].endswith("native_approx"):
            # Native effect/SD formulas use central-t quantile addition, not a
            # true inverse of the noncentral-t power. Retain as source evidence.
            continue
        assert row["status"] == "0" and row["ok"] == "T"
        result = _solve(row["case_id"], float(row["target"]))
        np.testing.assert_allclose(result.value, float(row["value"]), rtol=1e-6, atol=1e-8)


def test_discrete_significance_jump_is_not_reported_as_an_equation_solution():
    # For n=10, null p=.2 and alternative p=.4, neighboring nonrandomized
    # rejection regions have powers about .6177 and .8327. No alpha yields .8.
    with pytest.raises(ArithmeticError, match="residual"):
        stplan_solve(
            "stplan_exact_binomial_power",
            compute="alpha",
            target_power=0.8,
            bounds=(0.001, 0.49),
            parameters={"null_probability": 0.2, "alternative_probability": 0.4, "sample_size": 10},
        )
