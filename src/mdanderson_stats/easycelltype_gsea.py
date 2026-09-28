"""Deterministic ranked enrichment scores for caller-supplied EasyCellType sets.

This module implements the observed weighted running-sum statistic and the two
leading-edge conventions used by the pinned clusterProfiler/DOSE/fgsea path.
It does not calculate normalized enrichment, permutation p-values, or labels
selected by the EasyCellType p-value cutoff.

The source contract was checked against EasyCellType 1.5.4 and Bioconductor
3.18 snapshots: clusterProfiler 4.10.1, DOSE 3.28.2, fgsea 1.28.0. The
upstream GSEA formulas and source quirks are independently implemented here;
this file does not copy package code. See the project source audit for pins.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mdanderson_stats.easycelltype import _labels, _scores

_MAX_ROWS = 200_000
_MAX_RESULT_CELLS = 200_000
_MAX_RUNNING_SUM_WORK = 40_000_000
_MAX_RETAINED_GENES = 1_000_000


@dataclass(frozen=True)
class EasyCellTypeGSEASet:
    """Observed enrichment details for one cluster and reference cell type."""

    cell_type: str
    ranked_rows: int
    set_size: int
    enrichment_score: float
    fgsea_leading_edge: tuple[str, ...] | None
    core_enrichment_score: float | None
    core_enrichment: tuple[str, ...] | None
    core_rank: int | None
    core_enrichment_reason: str | None


@dataclass(frozen=True)
class EasyCellTypeGSEACluster:
    """Reference-set results for one marker cluster, in reference order."""

    cluster: str
    sets: tuple[EasyCellTypeGSEASet, ...]


@dataclass(frozen=True)
class EasyCellTypeGSEAResult:
    """Observed ES/core results only; no normalized scores or p-values."""

    clusters: tuple[EasyCellTypeGSEACluster, ...]
    score_type: str
    exponent: float
    min_size: int
    max_size: int


def easycelltype_gsea_es(
    query_genes: ArrayLike,
    clusters: ArrayLike,
    scores: ArrayLike,
    reference_genes: ArrayLike,
    reference_cell_types: ArrayLike,
    *,
    score_type: str = "std",
    exponent: float = 1.0,
    min_size: int = 1,
    max_size: int = 500,
) -> EasyCellTypeGSEAResult:
    """Calculate observed weighted GSEA scores for each cluster and set.

    Marker rows are stably sorted by decreasing score within cluster. The
    deterministic stable tie order is a Python convention; the pinned fgsea
    source warns that tied ranking order is arbitrary. Duplicate query gene
    IDs are retained as ranked rows, matching the wrapper. fgsea membership
    uses one first-ranked occurrence per distinct set gene, while the DOSE
    ``core_enrichment`` calculation marks every row with a matching name as a
    hit. Both values are returned separately because they differ with
    duplicate IDs.

    The set size is the number of unique reference genes present in the
    ranked list. It is filtered inclusively by ``min_size`` and ``max_size``;
    ``max_size`` is additionally capped at ranked length minus one. Its default
    of 500 matches the EasyCellType call's clusterProfiler default; callers may
    explicitly choose another cap. Sets with no
    overlap or outside that size range are omitted. Exponent zero uses the
    fgsea equal-hit fallback when all weighted hit scores are zero; in that
    case DOSE's separate core score is undefined and its fields are ``None``.

    This is the observed-score kernel only. It does not estimate NES,
    permutation probabilities, adjusted probabilities, or apply ``p_cut``.
    Those require the historical default fgsea multilevel backend and are not
    implied by these results.
    """
    gene_values = _labels(query_genes, "query_genes")
    cluster_values = _labels(clusters, "clusters")
    score_values = _scores(scores)
    ref_gene_values = _labels(reference_genes, "reference_genes")
    ref_type_values = _labels(reference_cell_types, "reference_cell_types")
    if not (len(gene_values) == len(cluster_values) == len(score_values)):
        raise ValueError("query_genes, clusters and scores must have equal lengths")
    if not ref_gene_values or len(ref_gene_values) != len(ref_type_values):
        raise ValueError("reference associations must be nonempty parallel vectors")
    _check_integer(min_size, "min_size", 1, _MAX_ROWS)
    _check_integer(max_size, "max_size", 1, _MAX_ROWS)
    if isinstance(exponent, (bool, np.bool_)) or not isinstance(exponent, Real):
        raise ValueError("exponent must be a finite nonnegative number")
    exponent_value = float(exponent)
    if not isfinite(exponent_value) or exponent_value < 0:
        raise ValueError("exponent must be a finite nonnegative number")
    if not isinstance(score_type, str) or score_type not in {"std", "pos", "neg"}:
        raise ValueError("score_type must be 'std', 'pos', or 'neg'")

    cluster_order = tuple(dict.fromkeys(cluster_values))
    type_order = tuple(dict.fromkeys(ref_type_values))
    if len(cluster_order) * len(type_order) > _MAX_RESULT_CELLS:
        raise ValueError("cluster-by-cell-type results exceed the 200,000-cell limit")

    refs_by_type: dict[str, set[str]] = {cell_type: set() for cell_type in type_order}
    for gene, cell_type in zip(ref_gene_values, ref_type_values, strict=True):
        refs_by_type[cell_type].add(gene)

    work = 0
    retained_genes = 0
    output: list[EasyCellTypeGSEACluster] = []
    rows_by_cluster: dict[str, list[int]] = {cluster: [] for cluster in cluster_order}
    for row, cluster in enumerate(cluster_values):
        rows_by_cluster[cluster].append(row)
    for cluster in cluster_order:
        rows = rows_by_cluster[cluster]
        if not rows:
            output.append(EasyCellTypeGSEACluster(cluster, ()))
            continue
        # Python's stable sort makes ties deterministic while preserving input
        # order; native R documents no guaranteed tie ordering for fgsea.
        order = sorted(rows, key=lambda i: -score_values[i])
        ranked_genes = [gene_values[i] for i in order]
        ranked_scores = np.asarray([score_values[i] for i in order], dtype=np.float64)
        if ranked_scores.size > _MAX_ROWS:
            raise ValueError("each ranked cluster is limited to 200,000 rows")
        first_rank: dict[str, int] = {}
        for rank, gene in enumerate(ranked_genes):
            first_rank.setdefault(gene, rank)
        cluster_sets: list[EasyCellTypeGSEASet] = []
        for cell_type in type_order:
            members = refs_by_type[cell_type].intersection(first_rank)
            set_size = len(members)
            effective_max = min(int(max_size), ranked_scores.size - 1)
            if set_size < int(min_size) or set_size > effective_max:
                continue
            work += ranked_scores.size
            if work > _MAX_RUNNING_SUM_WORK:
                raise ValueError("ranked GSEA work exceeds the 40,000,000-step limit")
            selected = np.fromiter(
                sorted(first_rank[gene] for gene in members), dtype=np.intp, count=set_size
            )
            fgsea_es, fgsea_edge = _fgsea_stat(
                ranked_scores, selected, exponent=exponent_value, score_type=score_type
            )
            core_es, core_genes, core_rank, core_reason = _dose_core(
                ranked_genes, ranked_scores, members, exponent_value
            )
            retained_genes += (len(fgsea_edge) if fgsea_edge is not None else 0) + (
                len(core_genes) if core_genes is not None else 0
            )
            if retained_genes > _MAX_RETAINED_GENES:
                raise ValueError("retained leading-edge output exceeds the 1,000,000-ID limit")
            cluster_sets.append(
                EasyCellTypeGSEASet(
                    cell_type=cell_type,
                    ranked_rows=int(ranked_scores.size),
                    set_size=set_size,
                    enrichment_score=fgsea_es,
                    fgsea_leading_edge=(
                        tuple(ranked_genes[int(i)] for i in fgsea_edge)
                        if fgsea_edge is not None
                        else None
                    ),
                    core_enrichment_score=core_es,
                    core_enrichment=core_genes,
                    core_rank=core_rank,
                    core_enrichment_reason=core_reason,
                )
            )
        output.append(EasyCellTypeGSEACluster(cluster, tuple(cluster_sets)))
    return EasyCellTypeGSEAResult(
        tuple(output), score_type, exponent_value, int(min_size), int(max_size)
    )


def _fgsea_stat(
    scores: NDArray[np.float64],
    selected: NDArray[np.intp],
    *,
    exponent: float,
    score_type: str,
) -> tuple[float, NDArray[np.intp] | None]:
    """Pinned fgsea calcGseaStat observed score and its own leading edge."""
    count = int(scores.size)
    set_size = int(selected.size)
    if not 0 < set_size < count:
        raise ValueError("GSEA requires at least one set gene and one miss")
    hit_scores = np.abs(scores[selected])
    if exponent == 0:
        hit_weights = np.ones(set_size, dtype=np.float64)
    else:
        hit_scale = float(np.max(hit_scores, initial=0.0))
        if hit_scale == 0:
            hit_weights = np.zeros(set_size, dtype=np.float64)
        else:
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                hit_weights = (hit_scores / hit_scale) ** exponent
    total = float(np.sum(hit_weights))
    if total == 0:
        hit_cumulative = np.arange(1, set_size + 1, dtype=np.float64) / set_size
        hit_increments = np.full(set_size, 1.0 / set_size, dtype=np.float64)
    else:
        hit_increments = hit_weights / total
        hit_cumulative = np.cumsum(hit_increments)
    misses_before = selected - np.arange(set_size, dtype=np.intp)
    tops = hit_cumulative - misses_before / (count - set_size)
    bottoms = tops - hit_increments
    max_peak = float(np.max(tops))
    min_peak = float(np.min(bottoms))
    if score_type == "std":
        if max_peak == -min_peak:
            score = 0.0
        else:
            score = max_peak if max_peak > -min_peak else min_peak
    elif score_type == "pos":
        score = max_peak
    else:
        score = min_peak

    if max_peak > -min_peak:
        edge_end = int(np.argmax(bottoms)) + 1
        edge = selected[:edge_end]
    elif max_peak < -min_peak:
        edge_start = int(np.argmin(bottoms))
        edge = selected[edge_start:][::-1]
    else:
        edge = None
    return float(score), edge


def _dose_core(
    ranked_genes: list[str],
    scores: NDArray[np.float64],
    members: set[str],
    exponent: float,
) -> tuple[float | None, tuple[str, ...] | None, int | None, str | None]:
    """Pinned DOSE gseaScores + leading_edge core_enrichment convention."""
    hits = np.fromiter((gene in members for gene in ranked_genes), dtype=bool)
    absolute = np.abs(scores[hits])
    if absolute.size == 0:
        return None, None, None, "no ranked hit rows"
    if exponent == 0:
        hit_weights = np.ones(absolute.size, dtype=np.float64)
    else:
        scale = float(np.max(absolute, initial=0.0))
        if scale == 0:
            return None, None, None, "DOSE core is undefined when all hit weights are zero"
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            hit_weights = (absolute / scale) ** exponent
    total = float(np.sum(hit_weights))
    if total == 0 or not isfinite(total):
        return None, None, None, "DOSE core has undefined hit-weight normalization"
    n = int(scores.size)
    set_size = len(members)
    if set_size >= n:
        return None, None, None, "DOSE miss denominator is zero for a full ranked set"
    hit_steps = np.zeros(n, dtype=np.float64)
    hit_ranks = np.flatnonzero(hits)
    hit_steps[hit_ranks] = hit_weights / total
    miss_steps = np.zeros(n, dtype=np.float64)
    miss_steps[~hits] = 1.0 / (n - set_size)
    running_all = np.cumsum(hit_steps) - np.cumsum(miss_steps)
    # DOSE sums all duplicate named rows as hits, whereas fgsea set membership
    # keeps a single first-match index per distinct gene name.
    max_index = int(np.argmax(running_all))
    min_index = int(np.argmin(running_all))
    max_peak = float(running_all[max_index])
    min_peak = float(running_all[min_index])
    core_es = max_peak if abs(max_peak) > abs(min_peak) else min_peak
    hit_running = running_all[hit_ranks]
    if core_es >= 0:
        hit_peak_index = int(np.argmax(hit_running))
        core = tuple(ranked_genes[int(i)] for i in hit_ranks[: hit_peak_index + 1])
        core_rank = max_index + 1
    else:
        hit_peak_index = int(np.argmin(hit_running))
        # Preserve the pinned source's negative-index corner at i == 0: in R,
        # -c(1:0) drops the first hit. The historical fixture checks this quirk.
        start = hit_peak_index + 1 if hit_peak_index == 0 else hit_peak_index
        core = tuple(ranked_genes[int(i)] for i in hit_ranks[start:])
        core_rank = n - min_index
    return float(core_es), core, core_rank, None


def _check_integer(value: object, name: str, minimum: int, maximum: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer from {minimum} through {maximum}")
    if not minimum <= int(value) <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} through {maximum}")
