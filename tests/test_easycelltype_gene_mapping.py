from __future__ import annotations

from mdanderson_stats.easycelltype_gene_mapping import easycelltype_gene_mapping


def test_human_mapping_keeps_duplicate_rows_and_drops_unmapped_in_alignment() -> None:
    result = easycelltype_gene_mapping(
        ["TP53", "TP53", "DEL11P13", "not-a-real-symbol"],
        "Human",
        unmapped="drop",
    )

    assert result.mapped_ids == ("7157", "7157", "100528024", None)
    assert result.candidate_ids[0] == ("7157",)
    assert result.candidate_ids[2] == ("100528024", "107648861")
    assert result.retained_indices == (0, 1, 2)
    assert result.retained_ids == ("7157", "7157", "100528024")
    assert result.unmapped_indices == (3,)
    assert result.ambiguous_indices == (2,)
    assert result.source_version == "Bioconductor 3.18; org.Hs.eg.db/org.Mm.eg.db 3.18.0"


def test_keep_policy_stays_aligned_and_reverse_conversion_is_available() -> None:
    kept = easycelltype_gene_mapping(["TP53", "not-a-real-symbol"], "Human", unmapped="keep")
    reverse = easycelltype_gene_mapping(
        ["7157", "not-an-entrez-id"],
        "Human",
        direction="entrez_to_symbol",
        unmapped="keep",
    )

    assert kept.retained_indices == (0, 1)
    assert kept.retained_ids == ("7157", None)
    assert reverse.mapped_ids == ("TP53", None)
    assert reverse.candidate_ids[0] == ("TP53",)


def test_error_policy_rejects_unmapped_ids_and_mouse_namespace_is_explicit() -> None:
    mouse = easycelltype_gene_mapping(["Trp53"], "Mouse")

    assert mouse.mapped_ids == ("22059",)
    assert mouse.species == "Mouse"
    try:
        easycelltype_gene_mapping(["missing"], "Human", unmapped="error")
    except ValueError as error:
        assert "1 identifiers are unmapped" in str(error)
    else:
        raise AssertionError("unmapped='error' must reject missing identifiers")
