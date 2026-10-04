"""Independent base-R checks for CiBolus truth interpolation and utility."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from mdanderson_stats.cibolus_scenarios import cibolus_interpolated_truth

_FIXTURES = Path(__file__).parent / "fixtures"
_ENDPOINTS = np.array([0.13, 0.38, 0.50, 0.79, 1.00])
_CONCENTRATIONS = np.array([0.15, 0.55])
_BOLUS = np.array([0.0, 0.35, 1.0])


def _rows(name: str) -> list[dict[str, str]]:
    with (_FIXTURES / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _matrix(rows: list[dict[str, str]], case: str, field: str) -> np.ndarray:
    selected = [row for row in rows if row["case"] == case]
    values = np.empty((len(_CONCENTRATIONS), len(_BOLUS)), dtype=np.float64)
    for row in selected:
        values[int(row["concentration_index"]), int(row["bolus_index"])] = float(row[field])
    return values


def test_all_interpolation_pairs_match_independent_base_r_joint_cells() -> None:
    settings = _rows("cibolus-scenario-settings.csv")
    reference = _rows("cibolus-scenario-cells.csv")
    for case in dict.fromkeys(row["case"] for row in settings):
        setting = next(row for row in settings if row["case"] == case)
        truth = cibolus_interpolated_truth(
            _CONCENTRATIONS,
            _BOLUS,
            _ENDPOINTS,
            response_zero=_matrix(settings, case, "response_zero"),
            response_one=_matrix(settings, case, "response_one"),
            toxicity_zero=_matrix(settings, case, "toxicity_zero"),
            toxicity_one=_matrix(settings, case, "toxicity_one"),
            toxicity_failure=_matrix(settings, case, "toxicity_failure"),
            response_curve=setting["response_curve"],
            toxicity_curve=setting["toxicity_curve"],
        )
        assert truth.shape == (2, 3, len(_ENDPOINTS) + 2, 2)
        assert not truth.flags.writeable
        expected_rows = [row for row in reference if row["case"] == case]
        for row in expected_rows:
            i = int(row["concentration_index"])
            j = int(row["bolus_index"])
            k = int(row["category_index"])
            np.testing.assert_allclose(
                truth[i, j, k],
                [float(row["no_toxicity"]), float(row["toxicity"])],
                rtol=0.0,
                atol=2e-15,
            )


def test_bolus_intervals_failure_marginals_and_regimen_utility() -> None:
    settings = _rows("cibolus-scenario-settings.csv")
    cells = _rows("cibolus-scenario-cells.csv")
    utility_rows = _rows("cibolus-scenario-utility.csv")
    expected_utility = _rows("cibolus-scenario-expected-utility.csv")
    marginals = _rows("cibolus-scenario-marginals.csv")

    for case in dict.fromkeys(row["case"] for row in settings):
        setting = next(row for row in settings if row["case"] == case)
        truth = cibolus_interpolated_truth(
            _CONCENTRATIONS,
            _BOLUS,
            _ENDPOINTS,
            response_zero=_matrix(settings, case, "response_zero"),
            response_one=_matrix(settings, case, "response_one"),
            toxicity_zero=_matrix(settings, case, "toxicity_zero"),
            toxicity_one=_matrix(settings, case, "toxicity_one"),
            toxicity_failure=_matrix(settings, case, "toxicity_failure"),
            response_curve=setting["response_curve"],
            toxicity_curve=setting["toxicity_curve"],
        )
        for i in range(2):
            for j in range(3):
                regimen = next(
                    row
                    for row in settings
                    if row["case"] == case
                    and int(row["concentration_index"]) == i
                    and int(row["bolus_index"]) == j
                )
                selected = [
                    row
                    for row in cells
                    if row["case"] == case
                    and int(row["concentration_index"]) == i
                    and int(row["bolus_index"]) == j
                ]
                expected_mass = np.array([float(row["response_probability"]) for row in selected])
                np.testing.assert_allclose(
                    truth[i, j].sum(axis=1), expected_mass, rtol=0.0, atol=2e-15
                )
                assert truth[i, j, 0, 1] == pytest.approx(
                    float(regimen["response_zero"]) * float(regimen["toxicity_zero"]),
                    rel=0.0,
                    abs=2e-15,
                )
                assert truth[i, j, -1, 1] == pytest.approx(
                    (1.0 - float(regimen["response_one"])) * float(regimen["toxicity_failure"]),
                    rel=0.0,
                    abs=2e-15,
                )
                utility_by_category = np.array(
                    [
                        [float(row["no_toxicity_utility"]), float(row["toxicity_utility"])]
                        for row in utility_rows
                    ]
                )
                score = float(np.sum(truth[i, j] * utility_by_category))
                expected = next(
                    row
                    for row in expected_utility
                    if row["case"] == case
                    and int(row["concentration_index"]) == i
                    and int(row["bolus_index"]) == j
                )
                assert score == pytest.approx(
                    float(expected["expected_utility"]), rel=0.0, abs=3e-13
                )
                for k, row in enumerate(selected):
                    marginal = next(
                        value
                        for value in marginals
                        if value["case"] == case
                        and int(value["concentration_index"]) == i
                        and int(value["bolus_index"]) == j
                        and int(value["category_index"]) == k
                    )
                    assert truth[i, j, k].sum() == pytest.approx(
                        float(marginal["response_probability"]), rel=0.0, abs=2e-15
                    )
