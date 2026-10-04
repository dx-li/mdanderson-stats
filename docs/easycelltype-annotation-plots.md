# EasyCellType annotation plots

The plotting helpers render label records already selected by the Fisher or
ranked-GSEA workflow. They do not rerank candidates, add a significance cutoff,
or calculate new inferential quantities. Matplotlib is optional; install the
repository's `plot` extra to use the helpers.

```python
from mdanderson_stats import easycelltype_fisher, easycelltype_labels
from mdanderson_stats.easycelltype_annotation_plots import (
    plot_easycelltype_annotation_dots,
    plot_easycelltype_candidates,
)

fisher = easycelltype_fisher(
    query_genes=["g1", "g2", "g3", "g4", "g5"],
    clusters=["cluster A"] * 5,
    scores=[1.4, 0.9, -0.2, 0.6, 0.7],
    reference_genes=["g1", "g2", "g3", "g5", "g6", "g7", "g8", "g9"],
    reference_cell_types=[
        "Type 1",
        "Type 1",
        "Type 2",
        "Type 2",
        "Type 2",
        "Type 3",
        "Type 3",
        "Type 3",
    ],
)
labels = easycelltype_labels(fisher)
bar_axes = plot_easycelltype_candidates(labels)
dot_axes = plot_easycelltype_annotation_dots(labels, pvalue="adjusted")
bar_axes.figure.savefig("easycelltype-candidates.png", dpi=160)
dot_axes.figure.savefig("easycelltype-significance.png", dpi=160)
```

The candidate bar chart preserves input order and displays Fisher mean marker
score or GSEA normalized enrichment score (NES), including negative values.
Hard labels are blue and soft labels orange. The dot plot places cluster on X
and candidate cell type on Y. Color and its colorbar show `-log10` of the
explicitly selected raw or adjusted p-value; marker area grows with the number
of overlapping Fisher genes or GSEA core-enrichment genes. Circles denote hard
labels and triangles denote soft labels. A zero p-value is displayed at the
smallest positive float64 value's finite `-log10` ceiling; the stored p-value
is unchanged. GSEA rows with undefined core enrichment use the minimum marker
area.

These are documented Python display conventions. The cached EasyCellType 1.5.4
author processing source defines candidate ranking and hard/soft selection,
but the plot-generation code and its aesthetics are not present in the cached
source. Therefore these plots make no claim of reproducing native plot style.
The helpers accept at most 100 labels and 20 clusters per plot and return
Matplotlib axes without showing, saving, or changing global plotting state.
See the [source audit](../research/easycelltype-annotation-plots-audit.md).
