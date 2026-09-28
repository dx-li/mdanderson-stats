# EasyCellType ranked-enrichment source profile

The author snapshot and Fisher implementation are recorded in
[the earlier audit](easycelltype-next-audit.md). Its GSEA branch does not pin
dependency versions. This project therefore selects the following historical
Bioconductor 3.18 release-branch profile for reproducibility; it is not a claim
that every native EasyCellType deployment used exactly these dependencies.

| Package | Version | Pinned GitHub mirror commit |
| --- | --- | --- |
| clusterProfiler | 4.10.1 | `072a273e06ca98f2c6a2c60a549fe3df97d99d38` |
| DOSE | 3.28.2 | `9e236d97ad30d3db1b00895815f23c88b17a04fe` |
| fgsea | 1.28.0 | `6c19b1edca90ef064c2246d52f3473d18dab0bcc` |

The `bioc/clusterProfiler`, `bioc/DOSE` and `bioc/fgsea` GitHub mirrors expose
these commits on `RELEASE_3_18`. Their DESCRIPTION files and the sources below
were retrieved read-only on September 28, 2026, and saved in ignored
`research/raw/EasyCellType/Bioc-3.18/`. No R dependency installation is needed
to execute the scalar statistic reference.

| File | SHA-256 |
| --- | --- |
| [clusterProfiler/R/enricher.R](https://github.com/bioc/clusterProfiler/blob/072a273e06ca98f2c6a2c60a549fe3df97d99d38/R/enricher.R) | `41d18c7fbca5ad7a652fe311c84261768b69189e82dc0c2e65e270eb6ef9fc0d` |
| [DOSE/R/gsea.R](https://github.com/bioc/DOSE/blob/9e236d97ad30d3db1b00895815f23c88b17a04fe/R/gsea.R) | `e5dcac4c10ba73b2e9a6ecbc6cb2b2e46a3f9966c74f039b727b42a3c1db6eaa` |
| [fgsea/R/fgsea.R](https://github.com/bioc/fgsea/blob/6c19b1edca90ef064c2246d52f3473d18dab0bcc/R/fgsea.R) | `a72f309cbf7139527afc9199794b7c1c14003de8d764dfa2e6d62ac360063ec5` |
| [fgsea/R/fgseaMultilevel.R](https://github.com/bioc/fgsea/blob/6c19b1edca90ef064c2246d52f3473d18dab0bcc/R/fgseaMultilevel.R) | `f1423355e1042bc00c3fd76be513be97ac0990a5da650c19568648fb64775ab0` |

## Wrapper and statistic contracts

EasyCellType sorts each cluster's named score vector in descending order and
calls `clusterProfiler::GSEA` with its cutoff, `minGSSize=1` and `scoreType`.
The selected profile defaults to exponent 1, maximum set size 500, BH adjustment,
`eps=1e-10` and the fgsea backend. Without an explicit permutation count, fgsea
selects its multilevel procedure. Plain permutation testing is not equivalent
to that default and must not silently replace it.

The statistic preparation keeps query rows, including duplicate gene IDs. A
reference gene maps to the first occurrence in the sorted query; repeated
reference matches are deduplicated. The gene-set size is the number of distinct
matched positions. The maximum allowed size is also capped at `N-1`, so an
all-query set is excluded. Absent genes do not contribute. Nonzero tied scores
and duplicate query names produce native warnings; a Python API must state its
ordering and diagnostic policy explicitly.

The weighted running-sum score uses absolute ranking scores to the specified
power. Zero total hit weight falls back to equal hit increments. Standard ES
selects the larger absolute excursion, returning zero for exact opposite-sign
ties; positive and negative modes select their respective extrema.

There are two different leading-edge calculations. `fgsea::calcGseaStat` returns
its own edge, determined by the standard extrema even with positive/negative
score mode. DOSE discards that edge, recomputes `gseaScores` and `leading_edge`,
then exposes `core_enrichment`, which EasyCellType consumes. These fields must
not be conflated. DOSE also drops missing p-values, filters by both raw and
adjusted cutoff, and sorts by adjusted p then descending absolute NES before
EasyCellType's final raw-p label ranking.

## Independent reference checkpoint

`tools/reference_easycelltype_gsea.R` parses and executes the unchanged pinned
`calcGseaStat` function in base R, checking the source MD5 first. It uses base
`match` for the source's first-match mapping contract; no fastmatch/data.table
or single-cell package is installed. Eight small cases cover signed modes,
unweighted and squared scores, size filters, duplicate query/reference IDs,
tied ranks and zero hit weights. Inputs and results are stored in four
`tests/fixtures/easycelltype-gsea-*.csv` files. These references establish the
ES kernel and its fgsea edge. The generator additionally executes unchanged
DOSE `gseaScores` and `leading_edge` for core ES, genes and rank, retaining its
duplicate-row behavior and negative-edge indexing. DOSE's zero-weight core
calculation fails even though fgsea defines an ES; fixtures label that core
undefined. Full normalization and adaptive tail probabilities still need their
own implementation checkpoint.
