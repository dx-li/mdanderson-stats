# Next uncovered software: EasyCellType

Catalog entry 159 has an accessible primary R implementation. The official app
URL returned an internal reader error, but the
[2023 paper](https://academic.oup.com/bioinformaticsadvances/article/3/1/vbad029/7085606)
links the author's [repository](https://github.com/rx-li/EasyCellType).
GitHub reads verified commit `e85e8187c540f66994b5ca12fe95f5d9eb95f1f5`
(2024-02-22), whose DESCRIPTION reports 1.5.4 and Artistic-2.0. This is an
author-source snapshot, not the current Bioconductor release. A later
Bioconductor source should be checked before final version claims.

Four source-only files are saved under ignored `research/raw/EasyCellType/R`:
`easyct.R`, `test_fisher.R`, `process_results.R` and `coremarkers.R`.
Their Git blob identities are respectively
`992c175a7d8735bebb7ee6c7879f1298e2f68fa8`,
`cbbe227b6561eeb12eaa1b6ba45d663247b4042a`,
`0157d42169d17db3792adc5a590ffb18bc7ff8c3` and
`5b3771c83bc4f5b81a67bf803cbba1f4d232eaf6`.

The software accepts marker-gene lists with cluster labels and expression
scores, rather than requiring a full single-cell expression matrix. That
provides a useful bounded Python workflow without installing the large R
single-cell stack. Its two analysis branches are ranked GSEA and a modified
Fisher analysis. Species/tissue filtering, marker-database choice, gene-ID
conversion, top-label summaries and plots surround those methods.

## Source contracts that must not be guessed

The Fisher function counts **reference association rows**, not a unique-gene
universe. Let N be reference rows, T rows for a cell type, L query rows and k
the number of unique overlapping IDs. It tests only when `k/L > T/N`. The
actual two-by-two table is `[[k,T],[L-k,N-T-L+k]]`, with a greater-tail Fisher
test. This unusual table must be compared with the article and newer source;
do not silently replace it with a conventional gene-universe enrichment test.
Negative table cells need an explicit failure contract.

For tested types it reports mean expression score among matching query rows,
the overlapping IDs, raw p and BH-adjusted p. Adjustment includes NA placeholders
for untested reference types, so the exact R `p.adjust` denominator must be
verified independently. Result processing sorts by increasing adjusted p and
then decreasing absolute mean score, selects one hard label and up to five
soft labels per cluster, and retains the contributing genes. Empty-result,
duplicate-gene and exact-tie behavior require explicit contracts.

The GSEA branch sorts scores and delegates to `clusterProfiler::GSEA` with
`minGSSize=1`, caller cutoff and std/pos/neg score type. It cannot be claimed
implemented by substituting unranked Fisher enrichment; normalization,
permutation/adaptive-tail behavior and reference versions need separate audit.

Next: delegate a bounded Fisher/annotation implementation to one Luna worker
after the waterfall checkpoint, while root verifies the unusual table against
the primary paper/newer source and creates independent base-R references.
Use explicit caller-supplied marker associations first if database redistribution
terms or binary retrieval remain unresolved; identify that as partial scope.
No database snapshot, gene-annotation package, GSEA implementation or clinical
cell labels have yet been ported. No installation or numerical job was run for
this scouting audit, and catalog entry 159 remains pending.
