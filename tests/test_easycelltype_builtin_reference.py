from __future__ import annotations

from mdanderson_stats.easycelltype import easycelltype_fisher
from mdanderson_stats.easycelltype_builtin_reference import (
    easycelltype_builtin_reference,
)


def test_bundled_cellmarker_reference_is_pinned_and_usable_for_annotation() -> None:
    reference = easycelltype_builtin_reference("cellmarker", "Human", tissues=("Kidney",))

    assert reference.source_rows == 49_149
    assert reference.source_name == "cellmarker.csv.gz"
    assert reference.source_sha256 == (
        "c6c1892ddeca843eac137a1b8152cfa00cb806fb1dd647aa6ebc730a03fc6835"
    )
    assert reference.selected_tissues == ("Kidney",)
    assert reference.gene_namespace == "EntrezID"
    assert reference.source_version is not None

    result = easycelltype_fisher(
        query_genes=("915", "916", "917"),
        clusters=("cluster-1", "cluster-1", "cluster-1"),
        scores=(1.0, 0.9, 0.8),
        reference_genes=reference.genes,
        reference_cell_types=reference.cell_types,
    )
    tests = result.clusters[0].tests
    t_cell = next(test for test in tests if test.cell_type == "T cell")
    assert t_cell.overlap_genes == ("915", "916", "917")
    assert t_cell.overlap_count == 3


def test_bundled_tables_keep_full_source_counts_and_stable_names() -> None:
    expected = {
        "cellmarker": (49_149, "cellmarker.csv.gz"),
        "clustermole": (172_003, "clustermole.csv.gz"),
        "panglao": (15_067, "panglao.csv.gz"),
    }
    for database, (rows, filename) in expected.items():
        reference = easycelltype_builtin_reference(database, "Mouse")
        assert reference.source_rows == rows
        assert reference.source_name == filename
        assert reference.selected_rows > 0
        assert len(reference.genes) == len(reference.cell_types)
