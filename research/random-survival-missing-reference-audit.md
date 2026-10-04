# RF-SRC missing-data reference audit

This reference package records a small, deterministic subset of the missing-
data contract for `randomForestSRC` 3.2.2. It does not redistribute or execute
the native forest. Its R generator uses explicit uniform tapes so that the
source selection rule can be checked independently of both RF-SRC's RNG and
the Python implementation's RNG.

## Provenance and license

The pinned source is the CRAN Git mirror revision
`b4d099e262423362a8872c13c468e6dbe2f9e9da`, version 3.2.2 (2023-05-23),
authors Hemant Ishwaran and Udaya B. Kogalur. Relevant file blobs are recorded
in `research/random-survival-forest-audit.md`: `src/randomForestSRC.c`
`e9c6e896c4f93c6eb1b85bdf37992964d74376ea`, `R/rfsrc.R`
`5be0607d365a5422e4b1d58e03c3e0dfe92c8045`, `R/utilities.R`
`25b40cd53edc9489f53e7799383d5db74494f6ba`, and `man/rfsrc.Rd`
`6f441a8d027d20764ce75d26a3aefcc54188ee82`; `R/utilities.data.R` is blob
`bac4cfb07e838cbcae029120d0bf5307c02d9b17`. The ignored source cache is under
`research/raw/randomForestSRC`; neither its full source nor compiled objects
are copied into this worktree. The upstream DESCRIPTION declares GPL (>= 3).
This reference independently expresses the observed source rules and does not
copy source code. It makes no broader licensing claim about RF-SRC.

## Source-defined behavior covered

- `R/rfsrc.R` calls `finalizeData` before fit (`99–121`, `169`). The manual
  (`man/rfsrc.Rd`, `181–184`) says `na.omit` drops a record if any predictor or
  response is missing. `R/utilities.data.R:parseMissingData` (`1013ff`)
  rejects responses that are wholly missing, drops wholly missing predictor
  columns unless every predictor column is missing (then errors), and removes
  records with every predictor and response missing. `finalizeData` (`204ff`)
  then applies the selected `na.action`; complete-case `na.omit` removes any
  remaining incomplete record. `R/utilities.R:get.na.action.bits` (`404–420`) maps
  `na.omit` to zero and `na.impute`/`na.random` to bit 16;
  `randomForestSRC.h` names that bit `OPT_MISS_SKIP`.
- `growTreeRecursive` imputes before each first-pass split (`randomForestSRC.c`
  `35339–35418`, with a corresponding path at `35658–35757`). Each imputation
  node receives bootstrap/in-bag `repMembrIndx` as donors and all-member
  `allMembrIndx` as recipients. Donor rows preserve bootstrap multiplicity.
  `imputeNode` (`4768–5065`) draws missing values from the node's in-bag values
  that were originally observed for that variable. If the nonterminal local
  donor pool is empty, it leaves the current parent completion untouched.
- Under first-pass `OPT_MISS_SKIP`, `getPreSplitResultGeneric`
  (`21825–21903`) first excludes rows with any originally missing response.
  Split candidate construction then filters that set again for rows with the
  candidate predictor originally observed (`stackAndConstructSplitVectorGenericPhase1`,
  `22092–22260`); candidate-specific survival score construction also uses
  that observed subset (`logRankNCR`, `19794ff`). Thus missing X is
  imputed for routing but its completed value is not used to generate that
  candidate's first-pass unique cutpoints or score. Later multiple-imputation
  passes set `multImpFlag` and disable this skip rule (`33940–34049`); this
  package does not model those later passes.
- Prediction's node walk also invokes `imputeNode(RF_PRED)` before routing
  (`randomForestSRC.c`, `5153–5195`), using that tree's in-bag `repMembrIndx`
  and current-node training distribution to complete missing new-profile X.
  For ordinary one-variable splits, a still-missing predictor can instead be
  routed by native daughter-polarity handling (`35072–35108`). Training OOB
  records are already among all-member recipients during tree growth, while
  remaining OOB performance response values are separately aggregated through
  terminal memberships; they must not become donors in their own tree.
- Terminal processing calls `imputeNodeAndSummarize` before
  `updateTerminalNodeOutcomes` (`35484–35512`). For a terminal missing outcome,
  `imputeNode` retries with originally missing in-bag rows' parent-completed
  values if no originally observed local donor exists. Time outcomes use the
  local mean and `getNearestMasterTime` (`5964–6005`); status uses a modal value
  and random tie choice via `getMaximalValue` (`6014–6068`). An empty terminal
  donor pool errors by default; `OPT_OUTC_TYPE` instead writes NaN. Leaf risk
  and event counts are subsequently formed using the completed tree arrays.
- The time grid used for reported forest output is formed from original
  complete observed events before the C fit (`R/utilities.survival.R`,
  `get.grow.event.info`, `53–106`). The master grid used to snap a node mean is
  initialized from all original finite times, including censored outcomes
  (`randomForestSRC.c`, `24835–24880`). An imputed terminal event may therefore
  occur at a master time absent from the original complete-event output grid.
- `getNearestMasterTime` has asymmetric near-tie behavior (`5964–6005`): it
  chooses the lower time when its distance is strictly smaller; only in the
  opposite branch, when absolute distance difference is below `EPSILON`, does
  a uniform `<= 0.5` choose the lower time. The exact tie is not a symmetric
  lower-on-half rule for all comparisons. `getMaximalValue` selects among
  sorted tied modes using `ceil(U * number_of_modes)`; `getSampleValue` selects
  donor `ceil(U * pool_size)` (`6089–6099`). The generator models these helpers
  with explicit `U` values and does not claim native RNG replay.
- `imputeCommon` (`5506–5930`, especially `5800–5919`) first uses completed
  terminal values from eligible trees. If that pool is empty, it draws from
  original full-data values observed for the field; if an outcome still has no
  donor, the native path errors. Predictor failure is left to the surrounding
  prediction handling. These common OOB/ensemble imputations are distinct from
  each tree's own in-bag donor pool. In a nonempty tree-terminal pool,
  continuous variables and times use the mean, while categorical fields use
  mode. Only an empty tree-terminal pool triggers the random full-data donor
  fallback. There are two response-completion call
  paths: `stackPerfResponse` (`11265–11331`) overlays OOB terminal imputations
  into a temporary response for performance calculation and does not apply
  `imputeMultipleTime`; final response summaries do apply that time-grid step
  (`10696–10704`). Thus temporary OOB metric times can be unsnapped means even
  though a final imputed response time is snapped. The Python reference should
  not collapse these contracts into one universal completed response array.

- A bootstrap replicate with no observed donor for a response is rejected at
  the root: `getNodeSign` (`2180–2250`) sets the bootstrap result false when a
  response's missingness signature is entirely absent. `growTreeRecursive`
  (`35470–35476`) then suppresses terminal creation for that replicate. This
  is a skipped tree, not a terminal that falls back to a global response donor.
  Consequently, Python `requested_trees` and effective `n_trees` can differ
  under `na_action="impute"`; reported curves average over retained trees.

## Reference scope and use

`tools/reference_random_survival_missing.R` generates small CSV ledgers under
`tests/fixtures` (or a caller-supplied output directory). Cases cover bootstrap
multiplicity and OOB recipients, both masks used for first-pass split
construction, terminal mode ties and time-snap asymmetry, completed terminal
risk/event counts, terminal no-donor branches, and common OOB fallback pools.
The numeric uniforms are fixture inputs, not source RNG seeds.

These ledgers are semantic oracles for the specified narrow cases. They do not
reconstruct the native imputation tree, the full forest, the default 500-tree
workflow, or `nimpute > 1`; they cannot establish whole-program parity. The
Python implementation uses a distinct NumPy random stream. Comparisons should
target masks, donor pools, values under a fixed tape, leaf counts, and observable
packed-tree/OOB outputs, rather than requiring native and Python seeded forests
to be identical.
