# EasyCellType gene mapping audit

This adds the source-advertised Human/Mouse symbol-to-Entrez input conversion
that the Python EasyCellType workflow previously left to callers. The pinned
EasyCellType 1.5.4 R/mapsymbol.R calls AnnotationDbi::mapIds with
keytype="SYMBOL", column="ENTREZID", and multiVals="first" using org.Hs.eg.db
or org.Mm.eg.db. R/easyct.R then applies na.omit to the mapped table before
cell-type analysis, preserving repeated mapped rows and removing unmapped rows.
R/coremarkers.R also uses the reverse EntrezID-to-SYMBOL lookup for displayed
core markers.

The author package records no OrgDb package version. This implementation
therefore selects Bioconductor 3.18 explicitly, rather than claiming to have
recovered the native annotation release. Official source archives:

- Human org.Hs.eg.db 3.18.0:
  <https://bioconductor.org/packages/3.18/data/annotation/src/contrib/org.Hs.eg.db_3.18.0.tar.gz>
- Mouse org.Mm.eg.db 3.18.0:
  <https://bioconductor.org/packages/3.18/data/annotation/src/contrib/org.Mm.eg.db_3.18.0.tar.gz>

The package DESCRIPTION files identify Marc Carlson as author and
Artistic-2.0 licensing; the NCBI Entrez Gene source date 2023-Sep11 comes
from the OrgDb SQLite metadata. The compact TSV snapshots were exported by
tools/export_easycelltype_gene_mappings.py from the packaged SQLite
gene_info/genes tables, ordered by gene_info._id. Only the TSV snapshots are
runtime assets; the source archives and databases remain in the ignored
research cache. data/easycelltype/gene-mapping-bioconductor-3.18.json records
archive, SQLite, and exported-file SHA-256 hashes, pair counts, URLs, license,
and ordering.

The selected first candidate is deterministic in Python: ascending _id.
Inspection of the AnnotationDbi 1.64.1 selection path confirms that duplicate
values are reduced by taking the first row returned, but its SQLite query has
no explicit ORDER BY for duplicates. Consequently the Python order is not
asserted to match the author's incidental query order. All candidate IDs are
retained in the result so ambiguous mappings remain visible.

The converter accepts a bounded vector, preserves input order and duplicate
rows, supports both directions, and offers explicit keep/drop/error handling
for unmapped values. Drop is the source wrapper's effective behavior; callers
must apply the returned retained_indices to clusters, scores, and any other
row-aligned data. This implements an input transformation only; it does not
claim full Shiny UI, annotation-version, or native duplicate-order parity.
