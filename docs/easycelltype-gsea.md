# EasyCellType ranked enrichment

`easycelltype_gsea_es` calculates the weighted running-sum enrichment statistic
for marker genes ranked within each cluster and caller-supplied cell-type gene
sets. It complements the [Fisher annotation branch](easycelltype.md).

Use `easycelltype_gsea_es` for observed scores and contributing genes, or
`easycelltype_gsea` for normalized scores, tail probabilities, BH adjustment
and the source cutoff rule. The inference API includes the historical
fgsea adaptive multilevel calculation. `easycelltype_gsea_labels` converts the
inference result into hard and soft cell-type annotations.

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

## Normalization and tail probabilities

```python
from mdanderson_stats import easycelltype_gsea

genes = [f"g{i}" for i in range(12)]
inference = easycelltype_gsea(
    query_genes=genes,
    clusters=["cluster1"] * 12,
    scores=[6.3, 5.9, 4.2, 2.8, 1.1, 0.7, -0.25, -0.8, -1.6, -2.4, -4.3, -7.7],
    reference_genes=["g0", "g1", "g2", "g5", "g6", "g7", "g9", "g10"],
    reference_cell_types=["A"] * 3 + ["B"] * 3 + ["C"] * 2,
    rng=2026,
)
for row in inference.clusters[0].sets:
    print(row.observed.cell_type, row.normalized_enrichment_score, row.p_value)
assert all(row.p_value is not None for row in inference.clusters[0].sets)
```

The default pilot uses 1,000 randomly selected gene sets to estimate the
directional null mean for normalized enrichment (NES). The source's estimated
error comparison selects either the simple tail estimate or adaptive splitting,
which uses 101 samples at each level. NES always comes from the pilot. The
result records each set's `inference_method`, directional pilot count and
extreme count. The default `p_cut=0.5` follows EasyCellType: `reported` requires
both raw and BH-adjusted probabilities to meet that cutoff. All tested sets
remain available in reference order for inspection.

The pilot preserves two source paths. Exactly one retained set uses R's
statistic convention, including its treatment of zero weights. Multiple sets
share an ordered random subset across their sizes and use the C++ cumulative
kernel's small positive weight floor. Inclusive tail comparisons and the
source's distinct zero-score sign rules are retained.

Fewer than ten pilot draws in the relevant direction leaves NES, probabilities
and uncertainty unavailable (`None`), with status
`insufficient_directional_pilot`. A zero directional null mean is likewise
unavailable. Increasing `n_perm_simple` can improve pilot precision. BH uses
the available probabilities, matching R's missing-value behavior.

`log2_error` is the source uncertainty estimate on the log2 probability scale.
A conditional probability estimate below one half leaves this uncertainty
unavailable and records `conditional_probability_below_half`. In the
mixed/multilevel path, probabilities below `eps` (default `1e-10`) are clamped
to that value, with status `below_eps` and no uncertainty estimate. Multiple
diagnostics can be separated by semicolons. The source all-simple early return
does not apply this later epsilon clamp. `qvalue` is not returned because the
EasyCellType label workflow does not consume it.

Randomness is reproducible with `rng` under the Python implementation; matching
R/C++ random streams is not claimed. A ranking whose prepared weights are all
zero is rejected for full inference at positive exponent. Observed scores
remain available through `easycelltype_gsea_es`; exponent zero uses unit
weights. Extreme weight ranges that cannot preserve the source pilot floor
are rejected explicitly.

## Cell-type labels

```python
from mdanderson_stats import easycelltype_gsea_labels

labels = easycelltype_gsea_labels(inference)
for label in labels:
    print(label.cluster, label.cell_type, label.method, label.core_enrichment)
assert [(label.cell_type, label.method) for label in labels] == [
    ("A", "hard_enrich"),
    ("C", "soft_enrich"),
]
```

Only rows marked `reported` enter label selection. The helper reconstructs
DOSE's order by increasing adjusted p-value and decreasing absolute NES, then
ranks labels by raw p-value, as the EasyCellType source does. Every tie at the
smallest raw p-value is a hard label. Soft selection takes the first five rows
by raw p-value and includes every tie at the fifth position. Consequently a
cluster can have multiple hard labels or more than five labels overall.

All hard rows precede all soft rows. A cell type appears only once within a
cluster, with hard taking precedence; cluster blocks follow the inference
result's first-occurrence order, a Python convention. Exact ties in both DOSE
ordering keys keep reference order. Core genes use DOSE's `core_enrichment`,
including `None` for an undefined core. Empty eligible results produce an empty
tuple. This helper applies no additional cutoff and runs no new sampling.

## Bounds and validation

Inputs and work are bounded, results are immutable, and simulation is serial.
The interface permits up to 200,000 ranked rows, 20,000 tested sets per cluster,
100,000 pilot permutations and 501 splitting samples, subject to a shared work
budget and retained-gene limits. Work-budget exhaustion raises an error rather
than returning incomplete tail estimates. Larger inputs can reach that budget
before these individual limits. Current and duplicated splitting samples
together are capped at two million stored positions before allocation.
The separate [bundled references](easycelltype-builtin-reference.md),
[versioned gene converter](easycelltype-gene-mapping.md) and
[annotation plots](easycelltype-annotation-plots.md) complete the Python input
and result workflows. Entry 159 is implemented with these documented choices;
exact native annotation versions, Shiny interfaces and plot aesthetics are not
claimed.

Native base-R fixtures cover thirty observed set/core scores. Separate native
fixtures cover 66 cumulative C++ pilot scores, 66 R single-set pilot scores
and 22 pairs of splitter statistics. A bounded twelve-seed comparison also
checks adaptive tail estimates against the pinned native splitter; statistical
agreement is assessed with Monte Carlo uncertainty, not exact RNG equality.
The [source audit](../research/easycelltype-gsea-audit.md) records the harness
scope and remaining workflow differences.
