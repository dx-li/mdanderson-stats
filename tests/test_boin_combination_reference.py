"""Focused comparisons with the native BOIN 2.7.2 combination implementation."""

import csv
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.boin_combination import BOINCombDesign, _biviso

FIXTURES = Path(__file__).parent / "fixtures"


def _matrix(row: dict[str, str], field: str) -> np.ndarray:
    shape = (int(row["nrow"]), int(row["ncol"]))
    return np.asarray([int(v) for v in row[field].split(";")]).reshape(shape)


def test_native_boundary_table_matches_all_patient_counts():
    rows = list(csv.DictReader((FIXTURES / "boin-combination-boundaries.csv").open()))
    design = BOINCombDesign(extra_safe=True)
    table = design.boundary_table(max_patients=int(rows[-1]["n"]))
    assert [int(v["n"]) for v in rows] == table.patients.tolist()
    assert [int(v["escalate_if_dlt_le"]) for v in rows] == table.escalate_max.tolist()
    assert [int(v["deescalate_if_dlt_ge"]) for v in rows] == table.deescalate_min.tolist()
    expected_elim = [
        int(v["eliminate_if_dlt_ge"]) if v["eliminate_if_dlt_ge"] != "NA" else 0 for v in rows
    ]
    # The public boundary object uses n+1 for an impossible elimination cutoff.
    assert table.eliminate_min.tolist() == [
        x or n + 1 for x, n in zip(expected_elim, table.patients)
    ]


def test_native_movement_cases_preserve_axis_neighbors_and_safety():
    rows = list(csv.DictReader((FIXTURES / "boin-combination-movements.csv").open()))
    for row in rows:
        design = BOINCombDesign(
            extra_safe="extra_safe" in row["case"],
            early_stop_patients=9 if "precision" in row["case"] else None,
        )
        result = design.next_dose(
            _matrix(row, "patients"),
            _matrix(row, "toxicities"),
            (int(row["current_a"]), int(row["current_b"])),
            rng=1,
        )
        native = None if row["next_a"] == "NA" else (int(row["next_a"]), int(row["next_b"]))
        # R and NumPy use different RNG streams. The source's posterior tie
        # cases are checked for the same candidate set; unique-score cases are
        # compared exactly.
        if row["case"] in {
            "escalate_equal_neighbor_tie",
            "deescalate_to_higher_posterior_neighbor",
        }:
            assert result.next_dose in {(1, 2), (2, 1)}
        else:
            assert result.next_dose == native, row["case"]
        if "elimination" in row["case"] or "extra_safe" in row["case"]:
            assert result.action == "stop_safety"
            assert result.eliminated.all()


def test_native_unrounded_biviso_fits_match():
    groups: dict[str, list[dict[str, str]]] = {}
    with (FIXTURES / "boin-combination-biviso.csv").open() as source:
        for row in csv.DictReader(source):
            groups.setdefault(row["case"], []).append(row)
    for case, rows in groups.items():
        shape = (int(rows[0]["nrow"]), int(rows[0]["ncol"]))
        values = np.asarray([float(r["raw_estimate"]) for r in rows]).reshape(shape)
        weights = np.asarray([float(r["weight"]) for r in rows]).reshape(shape)
        expected = np.asarray([float(r["biviso_fit"]) for r in rows]).reshape(shape)
        assert_allclose(_biviso(values, weights), expected, rtol=0, atol=5e-8, err_msg=case)


def test_selection_uses_native_rounded_fit_and_cross_closure():
    rows = {
        row["case"]: row
        for row in csv.DictReader((FIXTURES / "boin-combination-selection.csv").open())
    }
    design = BOINCombDesign()

    row = rows["cross_elimination_closure"]
    result = design.select_mtd(_matrix(row, "patients"), _matrix(row, "toxicities"))
    assert result.dose == (3, 3)
    assert_array_equal(
        result.eliminated,
        np.asarray(
            [
                [False, True, True, True],
                [False, True, False, False],
                [False, True, False, False],
            ]
        ),
    )

    row = rows["bound_mtd_rejects_over_target_fit"]
    result = design.select_mtd(_matrix(row, "patients"), _matrix(row, "toxicities"), bound_mtd=True)
    assert result.dose == (1, 2)


def test_contour_matches_all_native_rows():
    rows = list(csv.DictReader((FIXTURES / "boin-combination-selection.csv").open()))
    row = next(r for r in rows if r["case"] == "package_documented_contour")
    result = BOINCombDesign().select_mtd(
        _matrix(row, "patients"), _matrix(row, "toxicities"), mtd_contour=True
    )
    expected = tuple(
        tuple(int(value) for value in pair.strip("()").split(","))
        for pair in row["contour"].split(";")
    )
    assert result.contour == expected


def test_all_toxic_lowest_dose_is_a_clean_no_mtd_result():
    row = next(
        r
        for r in csv.DictReader((FIXTURES / "boin-combination-selection.csv").open())
        if r["case"] == "all_doses_overly_toxic"
    )
    result = BOINCombDesign().select_mtd(_matrix(row, "patients"), _matrix(row, "toxicities"))
    assert result.dose is None
    assert np.isnan(result.isotonic_mean).all()
