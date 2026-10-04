"""Compare the 2010 GAO decision routine with an independent base-R fixture."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.u2oet_gao2010_decision import u2oet_gao2010_decision

FIXTURES = Path(__file__).parent / "fixtures"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _tensor(case: str) -> np.ndarray:
    draws = np.zeros((2, 8, 3, 3, 2, 2), dtype=float)
    for row in _rows(FIXTURES / "u2oet-gao2010-conduct-draws.csv"):
        if row["case"] != case:
            continue
        draws[
            int(row["chain"]) - 1,
            int(row["draw"]) - 1,
            int(row["dose1"]) - 1,
            int(row["dose2"]) - 1,
            int(row["efficacy"]),
            int(row["toxicity"]),
        ] = float(row["probability"])
    return draws


def _config(case: str) -> dict[str, str]:
    return next(
        row
        for row in _rows(FIXTURES / "u2oet-gao2010-conduct-config.csv")
        if row["case"] == case
    )


def test_decisions_and_posterior_summaries_match_independent_r_reference() -> None:
    utility = np.array([[0.0, -100.0], [80.0, 60.0]])
    summaries = _rows(FIXTURES / "u2oet-gao2010-conduct-summaries.csv")
    decisions = {
        row["case"]: row
        for row in _rows(FIXTURES / "u2oet-gao2010-conduct-decisions.csv")
    }

    for case in ("interim", "stop_boundary", "global_stop", "final"):
        config = _config(case)
        if config["current_dose1"] != "NA":
            current_pair = (
                int(config["current_dose1"]) - 1,
                int(config["current_dose2"]) - 1,
            )
            treated = np.zeros((3, 3), dtype=int)
            for pair in config["treated_pairs"].split(";"):
                dose1, dose2 = (int(value) - 1 for value in pair.split(":"))
                treated[dose1, dose2] = 1
        else:
            current_pair = None
            treated = None

        result = u2oet_gao2010_decision(
            _tensor(case),
            utility,
            toxicity_limit=float(config["toxicity_limit"]),
            stopping_probability=float(config["stopping_probability"]),
            current_pair=current_pair,
            treated=treated,
        )
        expected_decision = decisions[case]
        assert result.stopped_for_global_toxicity is (
            expected_decision["stopped"] == "TRUE"
        )
        assert result.action == expected_decision["action"]
        assert result.minimum_exceedance_probability == float(
            expected_decision["minimum_exceedance_probability"]
        )
        assert result.selected_pair == (
            None
            if not expected_decision["selected_dose1"]
            else (
                int(expected_decision["selected_dose1"]) - 1,
                int(expected_decision["selected_dose2"]) - 1,
            )
        )
        expected_mask = np.array(
            [int(value) for value in expected_decision["eligible_mask_dose1_major"]],
            dtype=bool,
        ).reshape((3, 3))
        np.testing.assert_array_equal(result.eligible, expected_mask)

        for row in summaries:
            if row["case"] != case:
                continue
            i, j = int(row["dose1"]) - 1, int(row["dose2"]) - 1
            for actual, key in (
                (result.mean_utility[i, j], "mean_utility"),
                (result.utility_sd[i, j], "utility_sd"),
                (result.utility_mcse[i, j], "utility_mcse"),
                (result.mean_severe_toxicity[i, j], "mean_severe_toxicity"),
                (result.severe_toxicity_mcse[i, j], "severe_toxicity_mcse"),
                (
                    result.severe_toxicity_exceedance_probability[i, j],
                    "severe_exceedance_probability",
                ),
                (result.exceedance_probability_mcse[i, j], "exceedance_probability_mcse"),
            ):
                assert np.isclose(actual, float(row[key]), rtol=2e-13, atol=2e-13), (
                    case,
                    i,
                    j,
                    key,
                    actual,
                    row[key],
                )

    # Equality at both strict boundaries is the core safety-rule regression.
    assert decisions["interim"]["minimum_exceedance_probability"] == "0"
    assert decisions["stop_boundary"]["minimum_exceedance_probability"] == "0.75"
    assert decisions["stop_boundary"]["stopped"] == "FALSE"
    assert decisions["global_stop"]["stopped"] == "TRUE"
