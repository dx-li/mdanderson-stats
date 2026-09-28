"""Focused checks for the observed ranked EasyCellType GSEA core."""

import numpy as np
import pytest

from mdanderson_stats.easycelltype_gsea import easycelltype_gsea_es


def test_weighted_score_and_both_leading_edge_conventions():
    result = easycelltype_gsea_es(
        list("abcdefgh"),
        ["c"] * 8,
        [5, 3, 1, 0, 0, -1, -3, -5],
        ["a", "b", "g", "h", "a", "h", "d", "e"],
        ["Top", "Top", "Bottom", "Bottom", "Cross", "Cross", "Zero", "Zero"],
    )
    sets = {entry.cell_type: entry for entry in result.clusters[0].sets}
    assert sets["Top"].enrichment_score == 1.0
    assert sets["Top"].fgsea_leading_edge == ("a", "b")
    assert sets["Top"].core_enrichment_score == 1.0
    assert sets["Top"].core_enrichment == ("a", "b")
    assert sets["Bottom"].enrichment_score == -1.0
    assert sets["Bottom"].fgsea_leading_edge == ("h", "g")
    assert sets["Bottom"].core_enrichment == ("h",)
    # The two pinned APIs choose different edges for a balanced path.
    assert sets["Cross"].fgsea_leading_edge is None
    assert sets["Cross"].core_enrichment_score == 0.5
    assert sets["Cross"].core_enrichment == ("a",)
    assert sets["Zero"].enrichment_score == 0.0
    assert sets["Zero"].core_enrichment is None
    assert sets["Zero"].core_enrichment_reason is not None


def test_duplicate_query_and_reference_rows_follow_ranked_mapping_rules():
    result = easycelltype_gsea_es(
        ["a", "b", "a", "c", "d", "e"],
        ["cluster"] * 6,
        [3, 3, 2, 0, -1, -1],
        ["a", "a", "d", "outside", "a", "b"],
        ["Repeated", "Repeated", "Repeated", "Repeated", "Tied", "Tied"],
    )
    sets = {entry.cell_type: entry for entry in result.clusters[0].sets}
    repeated = sets["Repeated"]
    assert repeated.set_size == 2  # Duplicate reference associations are unique set members.
    assert repeated.fgsea_leading_edge == ("a",)
    assert repeated.core_enrichment == ("a", "a")  # DOSE marks both duplicate ranks.
    assert repeated.core_rank == 3
    assert sets["Tied"].fgsea_leading_edge == ("a", "b")


def test_score_type_exponent_and_inclusive_size_filter():
    genes = list("abcdefgh")
    scores = [5, 3, 1, 0, 0, -1, -3, -5]
    refs = ["a", "b", "g", "h", "a", "h", "d", "e", "a", "b", "c"]
    types = ["Top", "Top", "Bottom", "Bottom", "Cross", "Cross", "Zero", "Zero"]
    types += ["All", "All", "All"]
    positive = easycelltype_gsea_es(genes, ["c"] * 8, scores, refs, types, score_type="pos")
    scored = {entry.cell_type: entry for entry in positive.clusters[0].sets}
    assert scored["Bottom"].enrichment_score == 0.0
    filtered = easycelltype_gsea_es(
        genes,
        ["c"] * 8,
        scores,
        refs,
        types,
        exponent=2,
        min_size=2,
        max_size=2,
    )
    assert [entry.cell_type for entry in filtered.clusters[0].sets] == [
        "Top",
        "Bottom",
        "Cross",
        "Zero",
    ]
    with pytest.raises(ValueError, match="score_type"):
        easycelltype_gsea_es(genes, ["c"] * 8, scores, refs, types, score_type="two-sided")


def test_invalid_rank_inputs_and_zero_weight_do_not_fabricate_core():
    with pytest.raises(ValueError, match="finite numeric"):
        easycelltype_gsea_es(["a", "b"], ["c", "c"], [1.0, np.inf], ["a"], ["T"])
    zero = (
        easycelltype_gsea_es(["a", "b", "c"], ["c"] * 3, [0.0, 0.0, 0.0], ["a"], ["T"])
        .clusters[0]
        .sets[0]
    )
    assert zero.enrichment_score == 1.0  # fgsea spreads all-zero hit mass uniformly.
    assert zero.core_enrichment_score is None
    assert zero.core_enrichment is None
    assert "zero" in (zero.core_enrichment_reason or "")
