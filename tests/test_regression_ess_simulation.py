import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.parameter_distribution import ParameterDistribution
from mdanderson_stats.regression_ess_simulation import simulate_regression_ess


def _read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_regression_information_paths_and_crossings_match_original_r():
    fixture_dir = Path(__file__).parent / "fixtures"
    by_case = defaultdict(list)
    for row in _read_rows(fixture_dir / "regression-ess-simulation-priors.csv"):
        by_case[row["case"]].append(row)
    covariate_rows = _read_rows(fixture_dir / "regression-ess-simulation-covariates.csv")
    covariates = np.array([[float(row[f"x{i}"]) for i in (1, 2)] for row in covariate_rows])
    covariates = covariates.reshape(16, 32, 2)
    expected_paths = defaultdict(list)
    for row in _read_rows(fixture_dir / "regression-ess-simulation-paths.csv"):
        expected_paths[(row["case"], row["scope"])].append(row)
    expected_ess = {
        (row["case"], row["scope"]): row["ess"]
        for row in _read_rows(fixture_dir / "regression-ess-simulation-ess.csv")
    }

    for case, rows in by_case.items():
        model = rows[0]["model"]
        distributions = []
        for row in rows:
            p1, p2 = float(row["parameter1"]), float(row["parameter2"])
            if row["family"] == "gamma":
                p2 = 1.0 / p2  # the reference table records native R rate
            distributions.append(ParameterDistribution(row["family"], p1, p2))
        precision_prior = distributions.pop() if model == "normal" else None
        result = simulate_regression_ess(
            model,
            distributions,
            precision_prior=precision_prior,
            max_patients=32,
            replicates=16,
            covariate_draws=covariates,
        )
        for scope, indices in (
            ("whole", None),
            ("coefficients", np.arange(3)),
            ("precision_or_slopes", np.array([3]) if model == "normal" else np.array([1, 2])),
        ):
            path = expected_paths[(case, scope)]
            if indices is None:
                selected_prior = float(path[0]["prior_information"])
                selected_epsilon = float(path[0]["epsilon_information"])
                selected_curve = result.mean_cumulative_information.sum(axis=1)
                crossing = result.whole_model
            else:
                selected_prior = float(path[0]["prior_information"])
                selected_epsilon = float(path[0]["epsilon_information"])
                selected_curve = result.mean_cumulative_information[:, indices].sum(axis=1)
                crossing = result.crossing(indices)
            np.testing.assert_allclose(
                selected_curve + selected_epsilon,
                [float(row["mean_posterior_information"]) for row in path],
                rtol=3e-13,
                atol=3e-13,
            )
            selected_indices = (
                np.arange(len(result.parameter_names)) if indices is None else indices
            )
            assert np.sum(result.prior_information[selected_indices]) == pytest.approx(
                selected_prior, rel=2e-13
            )
            assert np.sum(result.epsilon_prior_information[selected_indices]) == pytest.approx(
                selected_epsilon, rel=2e-13
            )
            expected = expected_ess[(case, scope)]
            if expected == "NA":
                assert crossing.status == "not_reached"
                assert crossing.estimate is None
            else:
                assert crossing.estimate == pytest.approx(float(expected), abs=2e-12)
        assert not result.mean_cumulative_information.flags.writeable
        assert not result.covariate_draws.flags.writeable


def test_exact_and_flat_crossings_and_covariate_preflight():
    priors = [ParameterDistribution("normal", 0, 1)]
    result = simulate_regression_ess(
        "logistic",
        priors,
        max_patients=3,
        replicates=2,
        variance_inflation=2,
        covariate_draws=np.empty((2, 3, 0)),
    )
    assert result.whole_model.status == "exact"
    assert result.whole_model.estimate == pytest.approx(2.0)
    with pytest.raises(ValueError, match="match"):
        simulate_regression_ess(
            "logistic",
            priors,
            max_patients=3,
            replicates=2,
            covariate_draws=np.zeros((2, 2, 0)),
        )
    with pytest.raises(ValueError, match=r"\[-1,1\]"):
        simulate_regression_ess(
            "logistic",
            [ParameterDistribution("normal", 0, 1), ParameterDistribution("normal", 0, 1)],
            max_patients=1,
            replicates=1,
            covariate_draws=np.array([[[1.1]]]),
        )


def test_log_domain_crossing_survives_unrepresentable_prior_curvature():
    prior = ParameterDistribution("normal", 0, 1e-320)
    rng = np.random.default_rng(74)
    result = simulate_regression_ess(
        "logistic",
        [prior],
        max_patients=4,
        replicates=2,
        rng=rng,
        covariate_draws=np.empty((2, 4, 0)),
    )
    assert result.whole_model.status == "not_reached"
    assert result.whole_model.estimate is None
    assert result.log_prior_information_gain[0] > np.log(np.finfo(float).max)
    with pytest.raises(ArithmeticError, match="exceeds float64"):
        _ = result.prior_information
    # Invalid model-specific priors are rejected before a supplied RNG is used.
    second_rng = np.random.default_rng(31)
    second_state = second_rng.bit_generator.state
    with pytest.raises(ValueError, match="precision_prior"):
        simulate_regression_ess(
            "normal",
            [ParameterDistribution("normal", 0, 1)],
            max_patients=4,
            replicates=2,
            rng=second_rng,
        )
    assert second_rng.bit_generator.state == second_state
