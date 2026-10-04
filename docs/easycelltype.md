# EasyCellType Fisher annotation

`easycelltype_fisher` provides the source-compatible modified Fisher branch of
catalog entry 159, [EasyCellType](https://biostatistics.mdanderson.org/shinyapps/EasyCellType/).
It accepts a marker list with cluster labels and expression scores, plus
caller-supplied gene/cell-type associations. It does not require a full
single-cell expression matrix or a single-cell analysis stack.

```python
import numpy as np
from mdanderson_stats import easycelltype_fisher, easycelltype_labels

result = easycelltype_fisher(
    query_genes=["g1", "g2", "g3", "g7"],
    clusters=["cluster1"] * 4,
    scores=[1, -3, 5, 2],
    reference_genes=[
        "g1",
        "g2",  # Type A
        "g2",
        "g3",
        "g4",  # Type B
        "g1",
        "g4",
        "g5",
        "g6",  # Type C
        *[f"g{i}" for i in range(8, 19)],  # Type D
    ],
    reference_cell_types=["A"] * 2 + ["B"] * 3 + ["C"] * 4 + ["D"] * 11,
)
a = result.clusters[0].tests[0]
np.testing.assert_allclose(a.pvalue, 0.135475051264525, rtol=1e-12)
np.testing.assert_allclose(a.adjusted_pvalue, 0.314764183185236, rtol=1e-12)
assert a.overlap_genes == ("g1", "g2")
assert a.mean_score == -1
assert not result.clusters[0].tests[-1].tested

labels = easycelltype_labels(result)
assert [label.cell_type for label in labels] == ["A", "B", "C"]
assert labels[0].method == "hard_fisher"
assert all(label.method == "soft_fisher" for label in labels[1:])
```

Gene identifiers must use the same namespace in both inputs. Filter the
reference associations for the desired species, tissue and database before
calling the function. String and integer identifiers are accepted and normalized
to strings; missing identifiers and nonfinite scores are rejected. Inputs are
parallel one-dimensional vectors. Reference associations must be nonempty;
an empty query returns an empty result.

## Selecting rows from a local reference table

For the packaged CellMarker, Clustermole and Panglao snapshots, use
[`easycelltype_builtin_reference`](easycelltype-builtin-reference.md). It
returns the same reference object without a local data-preparation step.
The following loader supports a caller-supplied table instead.

`easycelltype_reference` performs the source's database/species/tissue row
selection on a caller-provided CSV or `.csv.gz` file. It requires the author
table columns `celltype`, `spe`, `organ` and `entrezid` in that order. The
loader preserves row order and duplicate associations and exposes its gene and
cell-type tuples to the Fisher or GSEA API:

```python
from mdanderson_stats import easycelltype_fisher, easycelltype_reference

reference = easycelltype_reference(
    "cellmarker.csv.gz",
    database="cellmarker",
    species="Human",
    tissues=["Blood", "Peripheral blood"],
    source_version="EasyCellType 1.5.4 author snapshot",
    source_provenance="Author repository commit e85e8187c540f66994b5ca12fe95f5d9eb95f1f5",
)
result = easycelltype_fisher(
    query_genes=["7157", "1956", "7422"],
    clusters=["cluster1"] * 3,
    scores=[2.1, 1.2, -0.4],
    reference_genes=reference.genes,
    reference_cell_types=reference.cell_types,
)
print(reference.source_sha256, reference.selected_rows)
```

For this path-based loader, the caller supplies the source-form data file;
`source_version`, `source_provenance`, the file's
SHA-256, total rows and selected rows are retained with the immutable
reference. `requested_tissues` records the deduplicated request (`None` means
all organs), while `selected_tissues` records the actual retained organs in
source order. Tissue names are validated after species filtering; any unknown
requested name raises. Empty tissue lists mean all organs. The table's genes
are EntrezIDs; symbol conversion is not guessed or performed by the loader.
Use the separate [versioned gene converter](easycelltype-gene-mapping.md)
for Human or Mouse symbols, applying its retained row indices to clusters and
scores together.
Blank source `organ` values are retained for an unfiltered selection and can
be selected explicitly with `tissues=[""]`.
Input limits are 100 MB on disk, 32 MB expanded text, 65,536 characters per
physical line and 200,000 rows; this also bounds compressed-input expansion.
See the [local-reference source audit](../research/easycelltype-reference-audit.md)
for provenance and limits.

## Exact source convention

For each cluster and cell type, let `N` be the total number of reference rows,
`T` the number of rows for that cell type, `L` the number of query rows and `k`
the number of distinct overlapping gene IDs. A test is performed only if
`k/L > T/N`. The code compares integer products to make this gate exact.

The original R implementation uses this table for a greater-tail Fisher test:

| | First column | Second column |
| --- | ---: | ---: |
| First row | `k` | `T` |
| Second row | `L-k` | `N-T-L+k` |

Its total is `N+k`. This is the author's modified test, rather than the usual
unique-gene-universe enrichment table. The Python function preserves the
source convention; replacing `T` with `T-k` would implement a different test.
The [paper](https://academic.oup.com/bioinformaticsadvances/article/3/1/vbad029/7085606)
describes Fisher testing and ranking but does not specify these table cells.
The [source audit](../research/easycelltype-next-audit.md) records the pinned
author source and verification against newer Bioconductor source.

Reference rows retain multiplicity, even for repeated gene/type associations.
Duplicate query IDs count repeatedly in `L` and in the mean expression score,
but contribute only once to `k`. The matched-score mean uses every matching
query row. Contributing IDs retain their first occurrence in the query.

Benjamini–Hochberg adjustment is performed separately within each cluster over
the tested cell types. Untested types do not enter the denominator. This agrees
with the original R call's handling of missing p-values. The returned frozen
records retain untested types with `tested=False` and NaN p-values/scores, so
an absent test cannot be mistaken for a nonsignificant test. R omits these rows.

## Annotation labels and interpretation

`easycelltype_labels` orders tested types by increasing adjusted p-value, then
decreasing absolute mean score. Exact ties preserve first-seen reference-type
order. It returns at most five labels per cluster: the first is `hard_fisher`
and the remaining rows are `soft_fisher`. The hard label appears only once.
`top_n` can request fewer rows, from one through five.

There is no p-value cutoff in the source Fisher branch. A hard label means
the highest-ranked tested type, including when its adjusted p-value is large;
it is not a confidence statement. Raw and adjusted p-values remain available
for downstream filtering. Clusters with no tested types have no labels.
Python preserves first-seen cluster order rather than R's sorted split names.

## Scope and validation

The port covers supplied-association Fisher tests, adjustment, score summaries,
contributing genes and hard/soft label selection, plus the separate
[ranked-enrichment workflow](easycelltype-gsea.md), including its probability
outputs. The local reference loader applies source database/species/tissue
selection to caller-supplied author-format tables; the bundled loader supplies
the pinned author snapshot. [Symbol/Entrez conversion](easycelltype-gene-mapping.md)
uses an explicit offline Bioconductor 3.18 annotation release and exposes
ambiguous mappings. [Candidate bars and annotation dots](easycelltype-annotation-plots.md)
display the selected Fisher or GSEA results with explicit Python visual conventions.
Catalog entry 159 is implemented as a Python workflow. Exact Shiny interfaces,
unrecorded native annotation versions and incidental ambiguous-key ordering
are not reproduced. See the [local-reference source audit](../research/easycelltype-reference-audit.md)
for provenance, blank-organ handling and input bounds.

Nine independent original-R cases check row multiplicity, overlap ordering,
adjusted probabilities, exact ties, score ranking, top-five selection, empty
matches, the strict gate and a probability near `1.84e-31`. The reference
generator uses base R without installing the single-cell stack. Resource limits
bound each input to 200,000 rows, the cluster/type result to 200,000 records and
retained overlap IDs to one million. Computation uses one cluster at a time.
