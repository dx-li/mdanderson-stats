# EasyCellType annotation plots: source audit

This audit records the distinction between source-backed candidate selection
and Python-specific display encodings. The pinned source is the author
EasyCellType 1.5.4 snapshot cached under
`research/raw/EasyCellType/R/`; its repository provenance is commit
`e85e8187c540f66994b5ca12fe95f5d9eb95f1f5`.

## Source-defined annotation content

In `process_results.R`, the Fisher branch sorts within each cluster by
increasing adjusted p-value and then decreasing absolute score. It assigns the
first result as hard and the next four as soft. The selected rows carry the
source score and contributing overlap genes. Existing
`easycelltype_labels` implements that selection and returns `mean_score`, raw
and adjusted p-values, method, rank, and overlap-gene IDs.

For GSEA, `process_results.R` selects every row tied at the minimum raw
p-value as hard, then rows through the fifth raw-p rank as soft, including
ties at that rank. A cell type selected as hard is not repeated as soft.
`coremarkers.R` splits `core_enrichment` on `/`, maps Entrez IDs to symbols,
and retains distinct cluster/ID memberships. The Python GSEA label helper
retains the selected `core_enrichment` tuple (or `None` when undefined), raw
and adjusted p-values, and NES.

## Python display conventions

The cached author processing functions do not contain the plotting
implementation: they indicate that plot processing follows, but the actual
bar/dot geoms, axes and aesthetic mappings are not part of the cached R files.
Accordingly the new plot module consumes selected label records without
changing their order or significance values. Its bars encode Fisher
`mean_score` or GSEA NES. Its dots encode cluster and cell type; color is
`-log10` of a caller-selected raw or adjusted p-value; size uses the number of
Fisher overlap genes or GSEA core genes; and marker shape distinguishes hard
from soft. Exact zero p-values are floored only for display at the float64
finite ceiling. Undefined GSEA core sizes are shown at the minimum marker
area.

No plot adds a cutoff, interval, score, or inferential result. No source claim
is made about native colors, dimensions, dot-size scaling, or plot order.
The Python bar chart includes a hard/soft legend; output dimensions and the
100-label / 20-cluster plotting limits are Python resource/display choices.
Independent references for the Fisher and GSEA label data remain the existing
source audits `easycelltype-next-audit.md` and `easycelltype-gsea-audit.md`;
this plotting layer only checks that those result values are mapped to chart
geometry as documented.
