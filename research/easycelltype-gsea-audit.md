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

## Full backend follow-up

The selected call defaults to `sampleSize=101`, `nPermSimple=1000` and
`eps=1e-10` (clusterProfiler overrides fgsea's smaller epsilon). The simple
pilot supplies signed null means for NES and tail counts. Relevant sign-mode
count below ten makes p, adjusted p, NES and log2-error unavailable; preserve
that state rather than substituting a numeric probability.

The wrapper compares the pilot's beta-interval log2 error with its estimated
multilevel error. Keep the simple estimate when `multError >= simpleError`;
use adaptive splitting when `multError < simpleError`. Splitting is grouped by
set size and uses odd-sized samples, median survival, duplication of survivors
and constrained perturbations of uniform fixed-cardinality gene subsets. NES
still comes from the pilot. Multilevel p divides the C++ tail estimate by
`(modeFraction+1)/(nPermSimple+1)` and caps at one. Its conditional-probability
flag can make the log2-error unavailable without deleting the p-value.

For the mixed/multilevel path, values below epsilon are clamped and their
log2-error set unavailable before BH adjustment. The all-simple early return
bypasses that later clamping/adjustment block; DOSE subsequently recomputes BH
in either case. Preserve the exact zero/sign inequalities: standard-mode NES
and extreme-count selection use `ES > 0`, while mode-fraction selection uses
`ES >= 0`. This is a source convention, not interchangeable algebra at ES zero.

Four pinned C++ files are also saved under the ignored fgsea `src/` directory:

| File | SHA-256 |
| --- | --- |
| `fgseaMultilevel.cpp` | `d5485a129144b9d8c81f439dc7f6f8420ef7147a44ca6b08f37c9c48524a8196` |
| `fgseaMultilevelSupplement.cpp` | `30bebe1044dbab9f41c1aa0093928ae72b97c9e27373cf8ea879e7e577180394` |
| `fastGSEA.cpp` | `c9bd0fdf2373477d41f0972e459db4a0b7ae0f4ef40257b84aaa7c1730e38557` |
| `esCalculation.cpp` | `94d7db77c6440d40a4c5be5e9ecd530f52a4febb4180d66588faa2e50a601a64` |

Next implementation should cover the pilot/NES, adaptive splitter, uncertainty,
BH and cutoff together, using explicit serial work limits. R/C++ random-stream
parity remains distinct from statistical algorithm parity. The pinned fgsea
license is MIT with copyright 2016–2019 Alexey Sergushichev; DOSE and
clusterProfiler use Artistic-2.0. Retain applicable notices for adaptations.

## Observed-score implementation checkpoint

Luna's `31673a0` was integrated as `9f9be3d`. The public
`easycelltype_gsea_es` API returns immutable per-cluster/per-set ES, fgsea edge,
DOSE core score/genes/rank and explicit reasons for unavailable core results.
The separate fields preserve the two source conventions. The default maximum
set size is 500; a caller's explicit override is a Python extension, still
subject to the `N-1` cap.

Review corrected scaling against an unrelated non-hit score, the unique-set
miss denominator with duplicate query IDs, and cumulative miss arithmetic.
Cluster grouping is indexed once, and retained-ID bounds count actual returned
core rows, including duplicates. Original-R references match all thirty rows;
root independently checked ES, both gene lists, ranks and undefined fields,
plus a `1e308` non-hit next to `1e-100` hits. That check took 0.0036 seconds after
import, peaked at 110.77 MiB and reported zero swaps. Seven focused existing/new
worker tests, Ruff formatting/lint and targeted mypy passed. No dependency
installation, large simulation or new CI workflow was used. Full multilevel
probabilities remain the next implementation tranche described above.

## Independent null-distribution and native pilot references

`tools/reference_easycelltype_gsea_null.R` enumerates all fixed-size subsets
of a twelve-gene ranking and executes unchanged pinned R `calcGseaStat`.
Eighteen cases record the exact signed null means, normalization, inclusive
tail counts and exhaustive-count probability transformations. The two
`easycelltype-gsea-null*.csv` fixtures describe mathematical R-score population
targets, not equality of a sampled C++ pilot or adaptive random stream.

The native pilot has a distinct zero-weight convention. Its C++ `gseaStats1`
uses `min(1e-5, minimum_positive_selected_weight) / 1024` as a positive floor.
The floor is computed from the entire ordered subset of maximum requested
size K, then reused for its smaller prefixes. The R preparation has already
converted the ranking to `abs(score) ** exponent`, and the pilot receives
`gseaParam=1`. Applying the epsilon to raw scores before the user exponent, or
recomputing it separately for each prefix, would change the source algorithm.
The observed-score R function keeps mixed zero weights at zero instead.

`tools/reference_easycelltype_gsea_cpp.R` compiles the unchanged scalar and
cumulative definitions preceding `calcRandomGseaStatCumulative` in pinned
`fastGSEA.cpp`, after checking its source MD5. The only removed line is the
unused `util.h` include; a small exported Rcpp wrapper passes explicit ordered
subsets. This needs the locally available Rcpp, but no Boost/BH installation,
native RNG implementation or full fgsea dependency tree. A single-threaded
compile succeeded. Sixty-six native scores in the two
`easycelltype-gsea-cpp-*.csv` fixtures cover ordinary and powered scores,
unweighted scores, mixed and all-zero hits, exact ties and a tiny later hit
that controls the shared prefix epsilon. These are native cumulative-kernel
references only; they do not establish full multilevel or RNG parity.

The same generator additionally compiles unchanged `esCalculation.cpp`
function bodies and records 22 signed/positive splitter score pairs, including
its tie and zero-weight conventions. These are separate functions from the
observed R score and cumulative pilot score.

`tools/reference_easycelltype_gsea_splitter.R` then assembles the pinned native
`EsRuler`, uniform-subset RNG, perturbation, duplication, sign correction and
probability assembly in one Rcpp translation unit. Source MD5 checks cover all
seven inputs. To avoid a dependency installation, its harness substitutes
Rmath digamma/trigamma for the corresponding Boost calls and removes resolved
local includes; the algorithm bodies and RNG are otherwise unchanged.
Ninety-six tail estimates use twelve seeds, twelve ranking positions, set size
three, sample size 101, both sign modes and four positive/negative thresholds.
They are bounded stochastic references, not a demand that NumPy reproduce
the native random streams or bitwise Boost results.

Review of the Python backend draft identified integration-blocking differences
in the beta log-mean denominator, perturbation acceptance direction, signed
correction counts and ordering of shared pilot subsets. The native references
support correcting these substantive statistical issues before publication;
the full backend is not yet part of the validated public checkpoint.

With more than one retained gene set, the source pilot draws one ordered
K-subset per iteration and derives every smaller set-size null score from its
prefixes. Null samples for different set sizes are therefore coupled.
Inclusive `<=`/`>=` comparisons are intentional.

There is a separate source branch for exactly one retained set:
`fgseaSimpleImpl` calls R `calcGseaStat` for each sampled subset instead of the
C++ cumulative kernel. It keeps mixed zero weights at zero and falls back to
equal weights only when all hits are zero. This branch depends on the number
of tested sets, not the number of genes in a set. The full Python pilot must
preserve it; C++ epsilon fixtures alone cannot validate the one-set branch.
The base-R null generator now also writes
`easycelltype-gsea-single-prefix.csv`: 66 R scores using the same explicit
ordered subsets as the C++ fixture. For example, the mixed-zero two-hit case
has R ES `5/6`, versus cumulative C++ ES `0.833333330078125`. Repeated draws
of that subset therefore exercise different inclusive-tail counts at `5/6`,
making this a meaningful branch regression rather than a rounding-only detail.

Additional pinned source dependencies retrieved for backend implementation:

| File | SHA-256 |
| --- | --- |
| `util.h` | `c73d1b46e65f70a79ffbb43585e4d0136a2b8b0b882b06738b3fd6096ac70fd9` |
| `util.cpp` | `35cd0525038f82c336eb33d3d117c7660947cd8ba62761975d75ff43d68e7fd5` |
| `esCalculation.h` | `03a0baa7d4c085a3a186c790fa9e6e870dfefa8ef7237394f1d1db038d73a284` |
| `fgseaMultilevelSupplement.h` | `1db51e0c11d7574bec802b51a332f5dc75e0fc31b06950ed78a4a793d7e51d07` |
