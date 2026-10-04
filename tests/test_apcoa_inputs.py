import csv

import numpy as np
import pytest

from mdanderson_stats.apcoa import adjusted_pcoa
from mdanderson_stats.apcoa_inputs import (
    APCoADistanceTable,
    APCoAMetadataTable,
    CategoricalEncoding,
    CovariateEncoding,
    prepare_apcoa_input,
    read_apcoa_distance_csv,
    read_apcoa_metadata_csv,
)


def _write_distance(path, row_ids, column_ids, values, delimiter=","):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter=delimiter)
        writer.writerow(["", *column_ids])
        for sample_id, row in zip(row_ids, values, strict=True):
            writer.writerow([sample_id, *row])


def _write_metadata(path, ids, rows, delimiter=","):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter=delimiter)
        writer.writerow(["sample_id", "age", "batch", "cohort"])
        for sample_id, age, batch, cohort in zip(ids, *rows, strict=True):
            writer.writerow([sample_id, age, batch, cohort])


def test_csv_inputs_align_ids_and_encode_explicit_reference(tmp_path):
    distance = tmp_path / "distance.csv"
    metadata_file = tmp_path / "metadata.csv"
    ids = ["s1", "s2", "s3", "s4"]
    values = np.array([[0, 2, 4, 6], [2, 0, 3, 5], [4, 3, 0, 1], [6, 5, 1, 0]], dtype=float)
    row_order = ["s3", "s1", "s4", "s2"]
    _write_distance(
        distance,
        row_order,
        ids,
        values[[2, 0, 3, 1]],
    )
    metadata_order = ["s4", "s2", "s1", "s3"]
    metadata_rows = {
        "s1": (20, "B", "case"),
        "s2": (30, "A", "control"),
        "s3": (40, "B", "control"),
        "s4": (50, "C", "case"),
    }
    _write_metadata(
        metadata_file,
        metadata_order,
        tuple([metadata_rows[i][j] for i in metadata_order] for j in range(3)),
    )

    dtable = read_apcoa_distance_csv(distance)
    mtable = read_apcoa_metadata_csv(metadata_file)
    prepared = prepare_apcoa_input(
        dtable,
        mtable,
        numeric_covariates=("age",),
        categorical_covariates={"batch": CategoricalEncoding(("A", "B", "C"), reference="B")},
        main_group="cohort",
    )

    assert prepared.sample_ids == tuple(ids)
    np.testing.assert_array_equal(prepared.distances, values)
    np.testing.assert_array_equal(
        prepared.covariates,
        [[20, 0, 0], [30, 1, 0], [40, 0, 0], [50, 0, 1]],
    )
    assert prepared.covariate_columns == ("age", "batch[A]", "batch[C]")
    assert prepared.group_labels == ("case", "control", "control", "case")
    assert prepared.encodings[1] == CovariateEncoding(
        "batch", "categorical", ("batch[A]", "batch[C]"), ("A", "B", "C"), "B"
    )
    assert not prepared.distances.flags.writeable
    assert not prepared.covariates.flags.writeable


def test_prepared_fit_reuses_numeric_core_exactly_and_keeps_group_separate():
    distances = np.array([[0, 1, 3, 5], [1, 0, 2, 4], [3, 2, 0, 2], [5, 4, 2, 0]], dtype=float)
    dtable = APCoADistanceTable(("a", "b", "c", "d"), distances)
    mtable = APCoAMetadataTable(
        ("d", "b", "a", "c"),
        {
            "x": (4.0, 2.0, 1.0, 3.0),
            "constant": (1.0, 1.0, 1.0, 1.0),
            "group": ("g2", "g1", "g1", "g2"),
        },
    )
    prepared = prepare_apcoa_input(
        dtable, mtable, numeric_covariates=("x", "constant"), main_group="group"
    )
    expected = adjusted_pcoa(
        distances, [[1, 1], [2, 1], [3, 1], [4, 1]], intercept=True, components=None
    )
    fit = prepared.fit(intercept=True, components=None)
    np.testing.assert_allclose(fit.adjusted.gram_matrix, expected.adjusted.gram_matrix, atol=0)
    assert fit.covariate_rank == expected.covariate_rank
    assert fit.intercept is True
    assert prepared.group_labels == ("g1", "g1", "g2", "g2")
    assert prepared.covariate_columns == ("x", "constant")


def test_alignment_encoding_and_explicit_group_adjustment_are_explicit():
    dtable = APCoADistanceTable(("a", "b", "c"), np.zeros((3, 3)))
    metadata = APCoAMetadataTable(("a", "b", "c"), {"cat": ("x", "y", "z"), "group": ("g",) * 3})
    with pytest.raises(ValueError, match="match exactly"):
        prepare_apcoa_input(
            dtable,
            APCoAMetadataTable(("a", "b", "other"), metadata.columns),
        )
    with pytest.raises(ValueError, match="undeclared levels"):
        prepare_apcoa_input(
            dtable,
            metadata,
            categorical_covariates={"cat": CategoricalEncoding(("x", "y"), "x")},
        )
    grouped_as_nuisance = prepare_apcoa_input(
        dtable,
        metadata,
        categorical_covariates={"cat": CategoricalEncoding(("x", "y", "z"), "x")},
        main_group="cat",
    )
    assert grouped_as_nuisance.group_labels == ("x", "y", "z")
    assert grouped_as_nuisance.covariate_columns == ("cat[y]", "cat[z]")


def test_csv_readers_reject_duplicate_or_misaligned_distance_ids(tmp_path):
    duplicate = tmp_path / "duplicate.csv"
    _write_distance(duplicate, ["a", "b"], ["a", "a"], [[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="unique"):
        read_apcoa_distance_csv(duplicate)

    mismatch = tmp_path / "mismatch.csv"
    _write_distance(mismatch, ["a", "other"], ["a", "b"], [[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="match exactly"):
        read_apcoa_distance_csv(mismatch)


def test_categorical_column_bound_and_delimiters_are_checked_before_encoding():
    ids = ("a", "b", "c")
    dtable = APCoADistanceTable(ids, np.zeros((3, 3)))
    mtable = APCoAMetadataTable(ids, {"cat": ("a", "b", "c"), "x": (1, 2, 3)})
    with pytest.raises(ValueError, match="encoded covariate count"):
        prepare_apcoa_input(
            dtable,
            mtable,
            numeric_covariates=("x",),
            categorical_covariates={"cat": CategoricalEncoding(("a", "b", "c", "d"), "a")},
        )
    with pytest.raises(ValueError, match="delimiter"):
        read_apcoa_metadata_csv("unused.csv", delimiter=";")
