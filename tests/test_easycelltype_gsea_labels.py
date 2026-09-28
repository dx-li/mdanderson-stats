import pytest

from mdanderson_stats.easycelltype_gsea import EasyCellTypeGSEASet
from mdanderson_stats.easycelltype_gsea_labels import easycelltype_gsea_labels
from mdanderson_stats.easycelltype_gsea_multilevel import (
    EasyCellTypeGSEAInference,
    EasyCellTypeGSEAInferenceCluster,
    EasyCellTypeGSEAInferenceSet,
)


def _result_row(
    cell_type: str,
    raw_p: float | None,
    adjusted_p: float | None,
    nes: float | None,
    *,
    reported: bool = True,
    core: tuple[str, ...] | None = ("g1", "g2"),
) -> EasyCellTypeGSEAInferenceSet:
    observed = EasyCellTypeGSEASet(
        cell_type=cell_type,
        ranked_rows=10,
        set_size=3,
        enrichment_score=0.5,
        fgsea_leading_edge=("g1",),
        core_enrichment_score=0.4,
        core_enrichment=core,
        core_rank=1,
        core_enrichment_reason=None,
    )
    return EasyCellTypeGSEAInferenceSet(
        observed=observed,
        normalized_enrichment_score=nes,
        p_value=raw_p,
        adjusted_p_value=adjusted_p,
        log2_error=0.1,
        inference_method="simple",
        status=None,
        reported=reported,
        pilot_mode_count=20,
        pilot_extreme_count=2,
    )


def _inference(
    clusters: tuple[EasyCellTypeGSEAInferenceCluster, ...],
) -> EasyCellTypeGSEAInference:
    return EasyCellTypeGSEAInference(
        clusters=clusters,
        score_type="std",
        exponent=1.0,
        min_size=1,
        max_size=500,
        p_cut=0.5,
        sample_size=101,
        n_perm_simple=1000,
        eps=1e-10,
    )


def test_labels_keep_fifth_rank_ties_hard_precedence_and_global_hard_order():
    first = EasyCellTypeGSEAInferenceCluster(
        "cluster-a",
        (
            _result_row("a0", 0.01, 0.5, 0.4, core=None),
            _result_row("a1", 0.02, 0.2, 0.8),
            _result_row("a2", 0.02, 0.2, -0.9),
            _result_row("a3", 0.03, 0.1, 0.5),
            _result_row("a4", 0.04, 0.2, 0.1),
            _result_row("a5", 0.04, 0.2, 0.9),
            _result_row("a6", 0.05, 0.3, 0.1),
            _result_row("not-reported", 0.001, 0.001, 10.0, reported=False),
            _result_row("unavailable", None, None, None, reported=False),
        ),
    )
    second = EasyCellTypeGSEAInferenceCluster(
        "cluster-b",
        (
            _result_row("b0", 0.01, 0.1, 0.2),
            _result_row("b1", 0.01, 0.1, -0.7),
        ),
    )

    labels = easycelltype_gsea_labels(_inference((first, second)))
    assert [(row.cluster, row.cell_type, row.method) for row in labels] == [
        ("cluster-a", "a0", "hard_enrich"),
        ("cluster-b", "b1", "hard_enrich"),
        ("cluster-b", "b0", "hard_enrich"),
        ("cluster-a", "a2", "soft_enrich"),
        ("cluster-a", "a1", "soft_enrich"),
        ("cluster-a", "a3", "soft_enrich"),
        ("cluster-a", "a5", "soft_enrich"),
        ("cluster-a", "a4", "soft_enrich"),
    ]
    assert labels[0].core_enrichment is None
    assert labels[1].p_value == pytest.approx(0.01)
    assert all(row.cell_type not in {"not-reported", "unavailable"} for row in labels)


def test_labels_allow_empty_result_and_reject_wrong_input_type():
    assert easycelltype_gsea_labels(_inference(())) == ()
    with pytest.raises(TypeError, match="EasyCellTypeGSEAInference"):
        easycelltype_gsea_labels(object())  # type: ignore[arg-type]


def test_reported_row_requires_available_finite_evidence():
    malformed = EasyCellTypeGSEAInferenceCluster(
        "cluster",
        (_result_row("cell", None, 0.1, 1.0, reported=True),),
    )
    with pytest.raises(ValueError, match="finite p-values and NES"):
        easycelltype_gsea_labels(_inference((malformed,)))
