"""Select EasyCellType hard and soft GSEA labels from ranked inference."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from mdanderson_stats.easycelltype_gsea_multilevel import EasyCellTypeGSEAInference

_MAX_LABELS = 1_000_000


@dataclass(frozen=True)
class EasyCellTypeGSEALabel:
    """One selected cell-type label and its ranked-GSEA evidence."""

    cluster: str
    cell_type: str
    p_value: float
    adjusted_p_value: float
    normalized_enrichment_score: float
    method: str
    core_enrichment: tuple[str, ...] | None


def easycelltype_gsea_labels(
    inference: EasyCellTypeGSEAInference,
) -> tuple[EasyCellTypeGSEALabel, ...]:
    """Select hard and soft annotations from a completed GSEA inference.

    Only rows already marked ``reported`` by the inference's raw and adjusted
    p-value cutoff are considered. Within each cluster, source result order is
    reconstructed by adjusted p-value and decreasing absolute NES, with ties
    stable in the inference's reference order. Hard labels include every raw-p
    minimum tie; soft labels include every tie at the fifth raw-p rank. All
    hard rows precede all soft rows, and a cell type selected as hard is not
    repeated as soft. Cluster blocks retain the Python inference order.

    ``core_enrichment`` is passed through as returned by DOSE; undefined cores
    remain ``None``. An empty tuple is returned when no reported row is
    eligible. This function does not apply a second p-value cutoff.
    """
    if not isinstance(inference, EasyCellTypeGSEAInference):
        raise TypeError("inference must be an EasyCellTypeGSEAInference")

    hard_rows: list[EasyCellTypeGSEALabel] = []
    soft_rows: list[EasyCellTypeGSEALabel] = []
    seen_hard: set[tuple[str, str]] = set()
    seen_all: set[tuple[str, str]] = set()

    for cluster_result in inference.clusters:
        candidates = []
        for item in cluster_result.sets:
            if not item.reported:
                continue
            p_value = item.p_value
            adjusted = item.adjusted_p_value
            nes = item.normalized_enrichment_score
            if (
                p_value is None
                or adjusted is None
                or nes is None
                or not isfinite(p_value)
                or not isfinite(adjusted)
                or not isfinite(nes)
                or not 0.0 <= p_value <= 1.0
                or not 0.0 <= adjusted <= 1.0
            ):
                raise ValueError("reported GSEA rows require finite p-values and NES")
            candidates.append((item, float(p_value), float(adjusted), float(nes)))

        # Python sorting is stable, so the original reference order resolves
        # ties in both source ordering keys.
        candidates.sort(key=lambda row: (row[2], -abs(row[3])))
        if not candidates:
            continue
        ordered_raw = sorted(candidates, key=lambda row: row[1])
        hard_cut = ordered_raw[0][1]
        soft_cut = ordered_raw[min(4, len(ordered_raw) - 1)][1]

        cluster_hard = [row for row in ordered_raw if row[1] == hard_cut]
        cluster_soft = [row for row in ordered_raw if row[1] <= soft_cut]
        for item, p_value, adjusted, nes in cluster_hard:
            cell_type = item.observed.cell_type
            if not isinstance(cell_type, str):
                raise ValueError("reported GSEA rows require a string cell type")
            core = item.observed.core_enrichment
            if core is not None and not isinstance(core, tuple):
                raise ValueError("core_enrichment must be a tuple or None")
            key = (cluster_result.cluster, cell_type)
            if key in seen_hard:
                continue
            if len(hard_rows) + len(soft_rows) >= _MAX_LABELS:
                raise ValueError("selected GSEA labels exceed the 1,000,000-row limit")
            seen_hard.add(key)
            seen_all.add(key)
            hard_rows.append(
                EasyCellTypeGSEALabel(
                    cluster_result.cluster,
                    cell_type,
                    p_value,
                    adjusted,
                    nes,
                    "hard_enrich",
                    core,
                )
            )
        for item, p_value, adjusted, nes in cluster_soft:
            cell_type = item.observed.cell_type
            if not isinstance(cell_type, str):
                raise ValueError("reported GSEA rows require a string cell type")
            core = item.observed.core_enrichment
            if core is not None and not isinstance(core, tuple):
                raise ValueError("core_enrichment must be a tuple or None")
            key = (cluster_result.cluster, cell_type)
            if key in seen_all:
                continue
            if len(hard_rows) + len(soft_rows) >= _MAX_LABELS:
                raise ValueError("selected GSEA labels exceed the 1,000,000-row limit")
            seen_all.add(key)
            soft_rows.append(
                EasyCellTypeGSEALabel(
                    cluster_result.cluster,
                    cell_type,
                    p_value,
                    adjusted,
                    nes,
                    "soft_enrich",
                    core,
                )
            )

    return tuple((*hard_rows, *soft_rows))
