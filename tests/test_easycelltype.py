"""Focused behavior checks for the bounded EasyCellType Fisher branch."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import fisher_exact

from mdanderson_stats.easycelltype import easycelltype_fisher, easycelltype_labels


def test_source_table_duplicate_query_rows_gate_and_ordered_overlap():
    result = easycelltype_fisher(
        ["g1", "g2", "g1", "g2"],
        ["cluster-a", "cluster-b", "cluster-b", "cluster-b"],
        [8.0, 2.0, 4.0, -1.0],
        ["g1", "g2", "g2", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11"],
        ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C", "C"],
    )
    assert [group.cluster for group in result.clusters] == ["cluster-a", "cluster-b"]
    group = result.clusters[1]
    a = group.tests[0]
    assert (a.query_rows, a.reference_rows, a.reference_type_rows, a.overlap_count) == (3, 11, 3, 2)
    assert a.overlap_genes == ("g2", "g1")
    assert a.mean_score == pytest.approx(5 / 3)
    assert a.tested
    expected = fisher_exact([[2, 3], [1, 7]], alternative="greater").pvalue
    assert_allclose(a.pvalue, expected, rtol=1e-14)
    assert not group.tests[1].tested  # k/L does not exceed T/N for B.
    assert np.isnan(group.tests[1].adjusted_pvalue)
    assert len(easycelltype_labels(result)) >= 1


def test_labels_total_top_five_and_stable_tie_order():
    genes = [f"g{i}" for i in range(12)]
    refs: list[str] = []
    types: list[str] = []
    for i, gene in enumerate(genes):
        refs.append(gene)
        types.append(f"type-{i}")
    result = easycelltype_fisher(genes[:6], ["c"] * 6, [1.0] * 6, refs, types)
    labels = easycelltype_labels(result, top_n=5)
    assert len(labels) == 5  # Hard is included in the five total labels.
    assert labels[0].method == "hard_fisher"
    assert labels[0].cell_type == "type-0"
    assert all(item.method == "soft_fisher" for item in labels[1:])
    assert [item.cell_type for item in labels] == [f"type-{i}" for i in range(5)]


def test_mixed_ids_and_extreme_score_mean_are_handled_explicitly():
    with pytest.raises(ValueError, match="identifiers"):
        easycelltype_fisher(["g", np.nan], ["c", "c"], [1.0, 2.0], ["g"], ["A"])
    fit = easycelltype_fisher(
        ["g1", "g2"], ["c", "c"], [-1e308, 1e308], ["g1", "g2", "g3"], ["A", "A", "B"]
    )
    a = fit.clusters[0].tests[0]
    assert a.tested
    assert a.mean_score == 0.0
    outlier = easycelltype_fisher(
        ["a", "a", "unmatched"],
        ["c"] * 3,
        [1e-100, 1e-100, 1e308],
        ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        ["A"] + ["B"] * 9,
    )
    assert_allclose(outlier.clusters[0].tests[0].mean_score, 1e-100, rtol=1e-14, atol=0)
    with pytest.raises(ValueError, match="finite numeric"):
        easycelltype_fisher(["a"], ["c"], [np.nan], ["a"], ["A"])
