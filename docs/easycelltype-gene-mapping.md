# EasyCellType gene identifier mapping

EasyCellType's Human and Mouse symbol workflow uses AnnotationDbi::mapIds
from the matching OrgDb package to convert the first input column from SYMBOL
to ENTREZID, selecting multiVals="first". The Python package bundles an
offline Bioconductor 3.18 snapshot (org.Hs.eg.db and org.Mm.eg.db, version
3.18.0). This is a documented Python annotation release; the EasyCellType
source did not pin its OrgDb dependency version.

```python
from mdanderson_stats.easycelltype_gene_mapping import easycelltype_gene_mapping

symbols = ["TP53", "BRCA1", "not-a-symbol"]
clusters = ["A", "A", "B"]
scores = [2.1, 1.8, 0.7]

mapped = easycelltype_gene_mapping(symbols, "Human", unmapped="drop")
kept_clusters = [clusters[i] for i in mapped.retained_indices]
kept_scores = [scores[i] for i in mapped.retained_indices]
entrez_ids = list(mapped.retained_ids)
```

Use retained_indices for every row-aligned input when choosing unmapped="drop".
The default reproduces the source wrapper's removal of unmapped symbols while
preserving duplicate input rows. With unmapped="keep", mapped_ids and
retained_ids remain input-aligned and contain None at unmapped positions;
unmapped="error" rejects any unmapped ID. Reverse Entrez-to-symbol conversion
is available with direction="entrez_to_symbol".

For duplicated keys, the result exposes every ordered candidate in
candidate_ids, the selected first value in mapped_ids, and ambiguous positions
in ambiguous_indices. Candidate order is ascending OrgDb _id, a stable rule
chosen for this Python snapshot. The source's SQL selection has no explicit
duplicate ordering, so this does not claim to reproduce native multiVals="first"
in ambiguous cases.

The packaged Human and Mouse files contain 191,179 and 102,092 mapping pairs.
The mapping data is licensed under Artistic-2.0; see
[Artistic-2.0 notice](../notices/Artistic-2.0.txt) and
[third-party notices](../THIRD_PARTY_NOTICES.md). Snapshot URLs, package
hashes, row counts, compressed-file hashes, and the deterministic export order
are recorded in the package data manifest. Conversion is offline and does not
install or call R/Bioconductor.
