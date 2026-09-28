# EasyCellType observed ranked-enrichment scores

`easycelltype_gsea_es` calculates the weighted running-sum enrichment statistic
for marker genes ranked within each cluster and caller-supplied cell-type gene
sets. It complements the [Fisher annotation branch](easycelltype.md).

This API returns observed scores and contributing genes. It does not yet
calculate normalized enrichment scores, p-values, adjusted p-values or final
cell-type labels. The historical default adaptive multilevel probability
calculation remains a separate part of the GSEA port.

```python
from mdanderson_stats import easycelltype_gsea_es

result = easycelltype_gsea_es(
    query_genes=["a", "b", "c", "d", "e", "f"],
    clusters=["cluster1"] * 6,
    scores=[5, 3, 1, -1, -3, -5],
    reference_genes=["a", "b", "e", "f"],
    reference_cell_types=["Type A", "Type A", "Type B", "Type B"],
)
for row in result.clusters[0].sets:
    print(row.cell_type, row.enrichment_score, row.core_enrichment)
```

The selected reproducibility profile is EasyCellType 1.5.4 with Bioconductor
3.18's clusterProfiler 4.10.1, DOSE 3.28.2 and fgsea 1.28.0. EasyCellType does
not pin these dependencies, so this identifies the source versions checked by
this project rather than asserting the online app's exact backend version.
See the [source audit and references](../research/easycelltype-gsea-audit.md).

## Ranking and scores

Scores must be finite. Each cluster is sorted in descending order, preserving
input order for exact ties. The native source warns that tied ordering may be
arbitrary; deterministic stable ordering is this Python interface's convention.
String and integer identifiers are accepted and normalized to strings, as in
the Fisher interface; missing identifiers are rejected.

Set membership maps each reference gene to its first occurrence in the ranked
query. Duplicate reference associations are deduplicated; duplicate query rows
remain in the ranking. `set_size` is the number of distinct matched positions.
Sets outside the inclusive `min_size`/`max_size` range are omitted. Defaults are
1 and 500, and the maximum is additionally limited to the ranked row count minus
one: an all-query set has no miss distribution and is excluded.

The default exponent is one. Exponent zero gives an unweighted score; positive
exponents weight hits by absolute ranking score to that power. Hit weights are
scaled for numerical stability. If every hit has zero score and the exponent
is positive, fgsea uses equal hit increments. The `"std"` score mode picks the
largest absolute excursion and returns zero for exact opposite-sign ties.
`"pos"` and `"neg"` select the positive peak and negative trough respectively.

## Two contributing-gene conventions

`fgsea_leading_edge` follows fgsea's statistic kernel. The separate
`core_enrichment` and `core_enrichment_score` follow DOSE's recomputation, which
EasyCellType uses for its reported core genes. These can differ: DOSE includes
every matching duplicate query row and retains its historical negative-edge
indexing convention. Even in positive/negative mode, these source routines
determine contributing genes using their own standard signed extrema.

DOSE's core calculation is undefined when all hit weights are zero at positive
exponent. The Python result keeps the valid fgsea score while returning `None`
for the undefined core fields and an explanatory `core_enrichment_reason`.
This is explicit partial output; it is not a fabricated native core result.

Inputs and work are bounded, and results are immutable. Native base-R fixtures
exercise eight small scenarios and thirty set scores, including separate DOSE
core results. No marker databases, gene-ID conversion, full GSEA probability
backend or native plots are bundled by this extension. Entry 159 remains partial.
