# Next uncovered software: EasyCellType

Catalog entry 159 has an accessible primary R implementation. The official app
URL returned an internal reader error, but the
[2023 paper](https://academic.oup.com/bioinformaticsadvances/article/3/1/vbad029/7085606)
links the author's [repository](https://github.com/rx-li/EasyCellType).
GitHub reads verified commit `e85e8187c540f66994b5ca12fe95f5d9eb95f1f5`
(2024-02-22), whose DESCRIPTION reports 1.5.4 and Artistic-2.0. This is an
author-source snapshot, not the current Bioconductor release. A later
Bioconductor source was also checked: `bioc/EasyCellType`'s `devel` branch
reports 1.15.0, and its `R/test_fisher.R` has the identical blob
`cbbe227b6561eeb12eaa1b6ba45d663247b4042a`. This verifies the unusual Fisher
table persists in newer development source, not just the older author snapshot.

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
The accessible article's section 2.2 confirms Fisher testing, BH adjustment and
expression-score ranking, but does not specify the table entries. Source
compatibility must therefore be labeled as such, without claiming that its
unusual table is the unique interpretation of the paper.

For tested types it reports mean expression score among matching query rows,
the overlapping IDs, raw p and BH-adjusted p. Adjustment receives NA placeholders
for untested reference types. Base R 4.4.1 source and execution verify that its
lazy default `n=length(p)` is evaluated after NA removal:
`p.adjust(c(.01,NA,.04), "BH")` gives `(.02,NA,.04)`. The denominator is the
number of tested types, not all reference types.
Result processing sorts by increasing adjusted p and then decreasing absolute
mean score, selects up to five labels per cluster including one hard label,
and retains the contributing genes. Empty-result,
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
No database snapshot, gene-annotation package or GSEA implementation has yet
been ported. Catalog entry 159 remains pending until the implementation is
integrated and independently verified.

## Original-R reference checkpoint

`tools/reference_easycelltype.R` loads the pinned `test_fisher.R` unchanged,
verifies its source checksum, and uses base R only. It generated nine cases
covering mixed overlap, duplicate query/reference rows, exact ties, absolute
score tie-breaking, no overlap, equality at the enrichment gate, top-five
selection and a tail probability of approximately `1.84e-31`. Inputs and
outputs are preserved in four `tests/fixtures/easycelltype-*.csv` files.
The top-label ordering uses the original base-R ordering expression; no
single-cell or dplyr installation is required. The generator also asserts
the NA/BH denominator example above.

The original Fisher branch uses unsorted query rows, so contributing genes
retain first-occurrence query order. It counts duplicate query rows in the
query size and matched-score mean, but overlap cardinality remains unique.
The new Python interface will reject missing/nonfinite inputs instead of
silently dropping rows, and retain untested rows for inspection. The source
wrapper drops such result rows. These differences must be documented explicitly.

## Integrated implementation and verification

Luna committed `85aeb52`, integrated as `a3720d5`. The new
`easycelltype_fisher` and `easycelltype_labels` APIs expose immutable cluster,
test and label records. Root added public exports and the
[usage guide](../docs/easycelltype.md). Entry 159 moves from pending to partial:
the catalog now has 62 implemented, 65 partial and 11 pending entries.
GSEA, bundled databases, metadata filters, gene-ID conversion and native plots
remain open.

Review corrected implicit identifier coercion, avoided repeated full query
scans for every type and preserved contributing-gene order within each cluster.
Per-gene score summaries retain duplicate-row weights and scale only the
matched summaries; an unrelated `1e308` score no longer erases matched
`1e-100` values. Input/result/retained-ID limits bound memory before large
outputs are built. One Luna worker was active, and numerical jobs were serial
with one BLAS/OpenMP thread.

The worker passed three focused tests in 1.16 seconds, all nine original-R
cases, Ruff formatting/lint, targeted mypy and diff checks. Root independently
compared the nine R cases and recomputed all nineteen tested tails with exact
integer combinations and rational arithmetic, matching within `2e-12` relative
tolerance even at the smallest tail. Separate checks covered multicluster
contributing-ID order, finite means at `1e308`, exact cancellation, empty input,
missing/nested identifiers and excessive result dimensions. The root check took
0.008 seconds after import, peaked at 113.64 MiB and recorded zero swaps.

A rebuilt wheel was imported in an isolated interpreter with warnings as errors.
All six EasyCellType exports and the documented example passed; changed module,
catalog and notice bytes matched source. The source archive contains the new
guides and reference fixtures and excludes raw downloads and environments.
This check took 0.135 seconds after import, peaked at 117.55 MiB and recorded
zero swaps. No full suite, large simulation, CI expansion or installation was
performed. Numerical checks target the source contract and meaningful edge cases.

The next uncovered-method lead is recorded in
[survival-contour-next-audit.md](survival-contour-next-audit.md). The full
catalog/publication goal remains active. GitHub publication is still blocked:
automatic approval review rejected the write and required approval is unavailable
under this session's permissions. No alternate write transport was attempted.
