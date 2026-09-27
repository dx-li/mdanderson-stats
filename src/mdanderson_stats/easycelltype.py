"""Bounded EasyCellType Fisher annotation for caller-supplied marker tables.

This ports the source-compatible Fisher branch only. It counts reference
association rows (including repeated gene/type associations), while overlap
size is the number of unique query IDs shared with a cell type. The tested
table is the unusual source table ``[[k, T], [L-k, N-T-L+k]]``; it is not
silently replaced by a conventional gene-universe enrichment table. A type is
tested only when ``k/L > T/N``. Missing/nonfinite marker data are rejected
rather than silently dropped as in R's ``na.omit`` wrapper.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isfinite
from numbers import Real
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import fisher_exact

type FloatArray = NDArray[np.float64]
_MAX_ROWS = 200_000
_MAX_RESULT_CELLS = 200_000
_MAX_OVERLAP_OUTPUT = 1_000_000


@dataclass(frozen=True)
class EasyCellTypeTest:
    """One cell-type test result, including explicit untested placeholders."""

    cell_type: str
    reference_rows: int
    reference_type_rows: int
    query_rows: int
    overlap_count: int
    overlap_genes: tuple[str, ...]
    tested: bool
    pvalue: float
    adjusted_pvalue: float
    mean_score: float


@dataclass(frozen=True)
class EasyCellTypeCluster:
    """Ordered test results for one cluster."""

    cluster: str
    tests: tuple[EasyCellTypeTest, ...]


@dataclass(frozen=True)
class EasyCellTypeFisherResult:
    """Fisher annotation results in first-seen cluster/type order."""

    clusters: tuple[EasyCellTypeCluster, ...]


@dataclass(frozen=True)
class EasyCellTypeLabel:
    """A hard or soft label and its contributing query genes."""

    cluster: str
    cell_type: str
    method: str
    rank: int
    pvalue: float
    adjusted_pvalue: float
    mean_score: float
    overlap_genes: tuple[str, ...]


def easycelltype_fisher(
    query_genes: ArrayLike,
    clusters: ArrayLike,
    scores: ArrayLike,
    reference_genes: ArrayLike,
    reference_cell_types: ArrayLike,
) -> EasyCellTypeFisherResult:
    """Run the modified Fisher test on marker rows and reference associations.

    Query rows determine ``L`` and the mean score; overlap count ``k`` is
    unique by gene ID. Duplicate query rows therefore increase ``L`` and can
    contribute repeatedly to the mean, but never increase ``k``. Reference
    rows are never deduplicated: ``N`` and ``T`` are row counts. The scipy
    greater-tail exact test is applied to the source table. Negative table
    cells raise ``ValueError`` with cluster/type context.

    An empty query returns an empty result; nonempty clusters with no
    qualifying cell type retain NaN placeholders. Cluster order follows first
    occurrence rather than the R wrapper's lexical split order.
    The row and cluster-by-type work is bounded before conversion/allocation.
    """
    genes = _labels(query_genes, "query_genes")
    group_values = _labels(clusters, "clusters")
    reference = _labels(reference_genes, "reference_genes")
    cell_types = _labels(reference_cell_types, "reference_cell_types")
    score_values = _scores(scores)
    if not (len(genes) == len(group_values) == len(score_values)):
        raise ValueError("query_genes, clusters and scores must have equal lengths")
    if not reference or not cell_types:
        raise ValueError("the reference must contain at least one association row")
    if len(reference) != len(cell_types):
        raise ValueError("reference_genes and reference_cell_types must have equal lengths")

    # Python preserves first appearance; the R wrapper sorts split names
    # lexicographically, so this is a deliberate result-order convention.
    cluster_order = tuple(dict.fromkeys(group_values))
    type_order = tuple(dict.fromkeys(cell_types))
    if len(cluster_order) * len(type_order) > _MAX_RESULT_CELLS:
        raise ValueError("cluster-by-cell-type result exceeds the 200,000-cell limit")
    rows_by_cluster: dict[str, list[int]] = {key: [] for key in cluster_order}
    for i, key in enumerate(group_values):
        rows_by_cluster[key].append(i)
    refs_by_type: dict[str, set[str]] = {key: set() for key in type_order}
    type_counts = {key: 0 for key in type_order}
    for gene, cell_type in zip(reference, cell_types, strict=True):
        refs_by_type[cell_type].add(gene)
        type_counts[cell_type] += 1
    cluster_results: list[EasyCellTypeCluster] = []
    overlap_output = 0
    for cluster in cluster_order:
        row_indices = rows_by_cluster[cluster]
        query_count = len(row_indices)
        local_gene_rows: dict[str, list[int]] = {}
        for i in row_indices:
            local_gene_rows.setdefault(genes[i], []).append(i)
        query_set = set(local_gene_rows)
        local_gene_means: dict[str, float] = {}
        for gene, indices in local_gene_rows.items():
            gene_scale = max(abs(score_values[i]) for i in indices)
            local_gene_means[gene] = (
                gene_scale
                * min(
                    1.0,
                    max(-1.0, fsum(score_values[i] / gene_scale for i in indices) / len(indices)),
                )
                if gene_scale
                else 0.0
            )
        unadjusted: list[float] = []
        provisional: list[EasyCellTypeTest] = []
        for cell_type in type_order:
            common = query_set.intersection(refs_by_type[cell_type])
            overlap_output += len(common)
            if overlap_output > _MAX_OVERLAP_OUTPUT:
                raise ValueError("retained overlap-gene output exceeds the 1,000,000-ID limit")
            # Preserve query first-occurrence order for source-like core IDs.
            common_ordered = tuple(sorted(common, key=lambda gene: local_gene_rows[gene][0]))
            overlap = len(common_ordered)
            type_count = type_counts[cell_type]
            if query_count == 0 or overlap * len(reference) <= type_count * query_count:
                provisional.append(
                    EasyCellTypeTest(
                        cell_type,
                        len(reference),
                        type_count,
                        query_count,
                        overlap,
                        common_ordered,
                        False,
                        float("nan"),
                        float("nan"),
                        float("nan"),
                    )
                )
                continue

            table = (
                overlap,
                type_count,
                query_count - overlap,
                len(reference) - type_count - query_count + overlap,
            )
            if any(value < 0 for value in table):
                raise ValueError(
                    f"source Fisher table has a negative cell for cluster {cluster!r}, "
                    f"cell type {cell_type!r}: {table}"
                )
            pvalue = float(
                fisher_exact(
                    [[table[0], table[1]], [table[2], table[3]]], alternative="greater"
                ).pvalue
            )
            if not isfinite(pvalue) or not 0 <= pvalue <= 1:
                raise ArithmeticError("Fisher exact test returned an invalid probability")
            matching_count = sum(len(local_gene_rows[gene]) for gene in common_ordered)
            matched_scale = max(abs(local_gene_means[gene]) for gene in common_ordered)
            mean_score = (
                matched_scale
                * min(
                    1.0,
                    max(
                        -1.0,
                        fsum(
                            local_gene_means[gene] / matched_scale * len(local_gene_rows[gene])
                            for gene in common_ordered
                        )
                        / matching_count,
                    ),
                )
                if matched_scale
                else 0.0
            )
            if not isfinite(mean_score):
                raise ArithmeticError("overlap mean score is not finite")
            unadjusted.append(pvalue)
            provisional.append(
                EasyCellTypeTest(
                    cell_type,
                    len(reference),
                    type_count,
                    query_count,
                    overlap,
                    common_ordered,
                    True,
                    pvalue,
                    float("nan"),
                    mean_score,
                )
            )

        adjusted = _bh(unadjusted)
        adjusted_index = 0
        completed: list[EasyCellTypeTest] = []
        for item in provisional:
            if item.tested:
                completed.append(
                    EasyCellTypeTest(
                        item.cell_type,
                        item.reference_rows,
                        item.reference_type_rows,
                        item.query_rows,
                        item.overlap_count,
                        item.overlap_genes,
                        True,
                        item.pvalue,
                        adjusted[adjusted_index],
                        item.mean_score,
                    )
                )
                adjusted_index += 1
            else:
                completed.append(item)
        cluster_results.append(EasyCellTypeCluster(cluster, tuple(completed)))
    return EasyCellTypeFisherResult(tuple(cluster_results))


def easycelltype_labels(
    result: EasyCellTypeFisherResult, *, top_n: int = 5
) -> tuple[EasyCellTypeLabel, ...]:
    """Select up to ``top_n`` labels per cluster, including one hard label.

    ``top_n`` is the total number of rows, including the hard label, and is
    bounded by five as in the source. Ordering is adjusted p-value ascending,
    then absolute mean score descending, then original reference type order.
    The hard label is included once, matching the source merge/deduplication.
    """
    if not isinstance(result, EasyCellTypeFisherResult):
        raise ValueError("result must be an EasyCellTypeFisherResult")
    if isinstance(top_n, (bool, np.bool_)) or not isinstance(top_n, (int, np.integer)):
        raise ValueError("top_n must be an integer from 1 through 5")
    if not 1 <= top_n <= 5:
        raise ValueError("top_n must be an integer from 1 through 5")
    labels: list[EasyCellTypeLabel] = []
    for group in result.clusters:
        tested = [entry for entry in group.tests if entry.tested]
        ordered = sorted(
            enumerate(tested),
            key=lambda pair: (pair[1].adjusted_pvalue, -abs(pair[1].mean_score), pair[0]),
        )
        if not ordered:
            continue
        selected = [entry for _, entry in ordered[: int(top_n)]]
        hard = selected[0]
        labels.append(_label(group.cluster, hard, "hard_fisher", 1))
        for rank, entry in enumerate(selected, 1):
            if entry.cell_type == hard.cell_type:
                continue
            labels.append(_label(group.cluster, entry, "soft_fisher", rank))
    return tuple(labels)


def _label(cluster: str, entry: EasyCellTypeTest, method: str, rank: int) -> EasyCellTypeLabel:
    return EasyCellTypeLabel(
        cluster,
        entry.cell_type,
        method,
        rank,
        entry.pvalue,
        entry.adjusted_pvalue,
        entry.mean_score,
        entry.overlap_genes,
    )


def _bounded_array(value: ArrayLike, name: str) -> NDArray:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a one-dimensional sequence, not a scalar string")
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.size > _MAX_ROWS:
            raise ValueError(f"{name} must be one-dimensional with at most {_MAX_ROWS} rows")
        if value.dtype.kind == "O" and any(
            isinstance(item, (list, tuple, np.ndarray, dict, set)) for item in value
        ):
            raise ValueError(f"{name} must contain scalar values, not nested data")
        raw = value
    else:
        try:
            size = len(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(f"{name} must be a one-dimensional sequence") from exc
        if size > _MAX_ROWS:
            raise ValueError(f"{name} must contain at most {_MAX_ROWS} rows")
        items = list(cast(Sequence[object], value))
        if any(isinstance(item, (list, tuple, np.ndarray, dict, set)) for item in items):
            raise ValueError(f"{name} must contain scalar values, not nested data")
        raw = np.asarray(items, dtype=object)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return raw


def _labels(value: ArrayLike, name: str) -> list[str]:
    raw = _bounded_array(value, name)
    result: list[str] = []
    for item in raw:
        if isinstance(item, (bool, np.bool_)) or not isinstance(
            item, (str, np.str_, int, np.integer)
        ):
            raise ValueError(f"{name} must contain nonempty string or integer identifiers")
        if not str(item):
            raise ValueError(f"{name} must contain nonempty identifiers")
        result.append(str(item))
    return result


def _scores(value: ArrayLike) -> list[float]:
    raw = _bounded_array(value, "scores")
    if raw.dtype.kind == "b" or (
        raw.dtype.kind not in "iuf" and any(not isinstance(item, Real) for item in raw)
    ):
        raise ValueError("scores must be finite numeric values")
    try:
        scores = raw.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scores must be finite numeric values") from exc
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite numeric values")
    return scores.tolist()


def _bh(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg adjustment over tested finite hypotheses only."""
    count = len(pvalues)
    if count == 0:
        return []
    order = np.argsort(pvalues, kind="stable")
    ranked = np.asarray(pvalues, dtype=float)[order]
    adjusted_sorted = np.minimum.accumulate((ranked * count / np.arange(1, count + 1))[::-1])[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()
