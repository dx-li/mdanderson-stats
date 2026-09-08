# SEQBIN coverage audit

Catalog entry 54 is implemented with the numerical, interface and corrected
source behaviors documented in [seqbin.md](seqbin.md). The audit covers SEQBIN
1.5's study-design and probability-reporting capabilities, not its terminal
prompt wording, pager or file-menu byte format.

The authoritative archive is `SEQBIN_V1.5.zip`. Source/manual SHA-256 hashes:

| File | SHA-256 |
| --- | --- |
| SOURCE/seqbin_bayes.f90 | aac97d770aab0582e6196ca20698b89e9b41e253623f48b5164593e7d1fe41df |
| SOURCE/design_properties_module.f90 | 8866442b1eed41013445e0449c106dbaa3d63b90cbad9253142277c2f80cc16f |
| DOC/seqbin_doc.txt | e1e56d54db39eaffbbea9779d6ff2c0cee2305dde78aaa3343b1ebd11da4c95e |

| Original capability | Python implementation | Evidence |
| --- | --- | --- |
| Less, greater, two-sided binomial trials | `SeqBinDesign.alternative` | 36 native boundary/probability cases, rational beta identities and exhaustive paths |
| Noninformative, uniform or arbitrary beta priors | Explicit shape pairs, including (0.5,0.5) and (1,1) | Native priors, strong-prior boundary checks and input validation |
| Prior mean/effective sample size | `seqbin_prior`, exact and legacy conversions | Actual beta mean/shape-mass checks and nonpositive-shape rejection |
| Common or separate posterior stopping cutoffs | Scalar or [low,high] `tail_probability` | Posterior identities, strict threshold equality, independent-side composition and overlap precedence |
| Sequential or scheduled group analyses | `looks`, separate `max_subjects` | Native sequential/group, earlier-final-analysis and empty-schedule cases; independent completion-time checks |
| Maximum size 2–10,000 | Validated constructor and scalable boundary search | Maximum-size boundary test and 10,000-subject no-analysis design |
| One-sided or common-cutoff two-sided frequentist calibration | `seqbin_calibrate`, conservative/nearest selection | 12 unchanged native FIND_STOP comparisons; independent enumeration of every small-trial attainable level |
| Separate two-sided error targets | `seqbin_calibrate_tails` | Independent calibrations, combined actual errors and boundary checks |
| Exact termination probabilities and expected sample sizes | Batched forward Bernoulli propagation | Native probability fixtures, exhaustive six-subject outcomes, zero/one probabilities and rare rejection cases |
| Per-side conditional expected sample size | `SeqBinProperties` conditional fields | Native checks outside source fallback region, exhaustive/rare-event tests, undefined-event handling |
| Compact or verbose boundary tables | `seqbin_boundary_table`, study table/report | Six native COMPACT_TABLE cases; unscheduled rows, cumulative totals, empty tables and first-look totals |
| Null and alternative-probability summaries | `SeqBinStudySpecification.probabilities` and full report | Full-precision parsed reports, input ordering, 101-value limit and complete manual alternative table |
| Input lists, changes and repeated studies | Explicit Python arrays, `.revise()`, JSON specifications | Independent snapshots, validation, all three modes replayed and maximum-size revision |
| File output and retained study settings | `write_report`, `write_specification` | UTF-8 export, explicit replacement, JSON round trips and I/O failure checks |

## Substitutions and corrected behavior

The Python API uses posterior **tail** cutoffs; the terminal interface describes
the complementary large posterior probability. Inclusive continuation bounds
and strict stopping inequalities are explicit. Both sides can use independent
cutoffs. The source's low-side precedence is retained when unusually large
cutoffs make the stopping regions overlap.

The source clamps continuation bounds to [0,n], even if every outcome should
stop under its posterior criterion. Python represents that empty continuation
set directly; `legacy_bounds=True` reproduces the clamping. The original
conditional-sample-size fallback at rejection probabilities <=1e-8 is replaced
by the actual conditional expectation for positive probabilities and NaN when
it is undefined. Beta mean/size conversion defaults to the described mean;
`conversion="legacy"` preserves the source formula that changes that mean.

Calibration searches discrete designs over binary64 cutoffs and exposes
attainable levels on either side of the target. The source uses a continuous
root finder with finite tolerances. Nearest selection is honored for separate
tails, correcting the source dialog's unconditional choice of conservative
cutoffs. Combined two-sided errors are recomputed after combining separately
calibrated boundaries, rather than presented as the sum of independent trials.

The maximum sample size is independent of the final analysis time. Surviving
paths complete at that maximum, including a schedule with no analyses. Outcome
propagation stops after the last analysis because subsequent Bernoulli results
cannot change any reported stopping event. Explicit look lists are validated
within the study's maximum; Python does not silently retain irrelevant indices.

Display compaction leaves all numerical state intact. This corrects the source's
recalculation of null probabilities using already-compacted boundary arrays.
Every per-look total is initialized, including the first row. Reports contain
numeric results and settings; instructions/explanatory terminal pages are
replaced by documentation. Python sequences replace interactive list editing
and equally spaced list-entry menus. JSON is a new replay format, not a parser
for the original terminal transcript.

## Validation scope and performance

Native fixtures compile unchanged numerical source with independent drivers.
They do not execute the complete interactive program. Source, dependency,
extraction, driver and compiler hashes identify the tested routines.
`test_seqbin_manual.py` additionally builds the manual's 50-subject nearest-0.05
study with a Beta(0.5,0.5) prior and null probability 0.2. All nine alternative
rows reproduce the printed rejection probabilities and unconditional/conditional
sample sizes. The correct null rejection probability matches the manual's
boundary-table total, approximately 0.05060; its compacted null summary is the
source bug described above.

`tools/benchmark_seqbin.py` verifies numerical equivalence while timing one
batched call against 101 scalar calls. Recorded three-run medians show about
8.5× throughput for a 500-subject sequential design and 13× for a 500-subject
design whose last scheduled analysis is at 200 subjects. See
[seqbin-benchmark.json](seqbin-benchmark.json) for workloads and versions.
These measurements do not compare against Fortran or benchmark calibration.
