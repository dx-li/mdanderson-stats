import csv
import gzip
import hashlib
from dataclasses import FrozenInstanceError

import pytest

from mdanderson_stats.easycelltype import easycelltype_fisher
from mdanderson_stats.easycelltype_gsea import easycelltype_gsea_es
from mdanderson_stats.easycelltype_reference import easycelltype_reference

_COLUMNS = ["celltype", "spe", "organ", "entrezid"]
_ROWS = [
    ("Type A", "Human", "Blood", "1"),
    ("Type A", "Human", "Blood", "1"),
    ("Type A", "Human", "Brain", "2"),
    ("Type B", "Human", "Blood", "2"),
    ("Type C", "Human", "", "3"),
    ("Mouse A", "Mouse", "Blood", "9"),
]


def _write_table(path, rows=_ROWS):
    if path.name.endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(_COLUMNS)
            writer.writerows(rows)
    else:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(_COLUMNS)
            writer.writerows(rows)
    return path


def test_reference_streams_gzip_filters_species_and_tissue_without_deduplicating(tmp_path):
    path = _write_table(tmp_path / "markers.csv.gz")
    reference = easycelltype_reference(
        path,
        database="cellmarker",
        species="Human",
        tissues=("Brain", "Blood", "Brain"),
        source_version="test-snapshot",
        source_provenance="synthetic fixture",
    )

    assert reference.requested_tissues == ("Brain", "Blood")
    assert reference.selected_tissues == ("Blood", "Brain")
    assert reference.genes == ("1", "1", "2", "2")
    assert reference.cell_types == ("Type A", "Type A", "Type A", "Type B")
    assert reference.gene_namespace == "EntrezID"
    assert reference.source_rows == 6
    assert reference.selected_rows == 4
    assert reference.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert reference.source_version == "test-snapshot"
    assert reference.source_provenance == "synthetic fixture"
    with pytest.raises(FrozenInstanceError):
        setattr(reference, "species", "Mouse")


def test_reference_vectors_feed_existing_fisher_and_gsea_apis(tmp_path):
    reference = easycelltype_reference(
        _write_table(tmp_path / "markers.csv"),
        database="cellmarker",
        species="Human",
        tissues=["Blood", "Brain"],
    )
    fisher = easycelltype_fisher(["1"], ["cluster"], [1.0], reference.genes, reference.cell_types)
    assert fisher.clusters[0].tests[0].overlap_genes == ("1",)

    gsea = easycelltype_gsea_es(
        ["1", "2", "x"],
        ["cluster"] * 3,
        [3.0, 2.0, 1.0],
        reference.genes,
        reference.cell_types,
    )
    assert {row.cell_type for row in gsea.clusters[0].sets} == {"Type A", "Type B"}


def test_empty_tissue_filter_means_all_species_organs_and_unknown_is_rejected(tmp_path):
    path = _write_table(tmp_path / "markers.csv.gz")
    all_organs = easycelltype_reference(path, database="panglao", species="Human", tissues=())
    assert all_organs.requested_tissues is None
    assert all_organs.selected_tissues == ("Blood", "Brain", "")
    assert all_organs.source_rows == 6
    assert all_organs.selected_rows == 5

    blank_organ = easycelltype_reference(path, database="panglao", species="Human", tissues=("",))
    assert blank_organ.genes == ("3",)
    assert blank_organ.selected_tissues == ("",)

    with pytest.raises(ValueError, match="unknown Human tissue/organ"):
        easycelltype_reference(
            path,
            database="panglao",
            species="Human",
            tissues=("Blood", "Not a tissue"),
        )


def test_mouse_tissue_choices_are_validated_after_species_filter(tmp_path):
    reference = easycelltype_reference(
        _write_table(tmp_path / "markers.csv"),
        database="clustermole",
        species="Mouse",
        tissues=("Blood",),
    )
    assert reference.genes == ("9",)
    assert reference.cell_types == ("Mouse A",)
    assert reference.selected_tissues == ("Blood",)
