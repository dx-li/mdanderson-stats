# MTADF coverage audit

Baseline main: 7ce1e7b. The previous batch integrated BMS and Occam-window CRM
aggregation, so the broader catalog goal made progress and remains active.
Entry 114 starts this batch pending. One Luna implementation agent uses the
existing mda-efftox-core checkout; root verifies sources, prepares an independent
reference and integrates. Numerical jobs are serial with BLAS/OpenMP threads
limited to one. No new dependencies, CI expansion or full-suite run is planned.

The current MTADF app was checked on 2026-09-26: version 1.0.5.0, updated
2026-01-06. It explicitly names double-sided isotonic regression. The authors'
institutional paper gives the relevant mathematical specification in Sections
2.1 and 2.3. Exact source links and choices are in docs/mtadf-sources.json.

The paper pools posterior overdose probabilities, rather than counts or
posterior toxicity means. Its safety inequality is strict. Its small-concentration
prior calibration depends on the toxicity limit and cutoff. Efficacy PAVA uses
patient counts as weights (the paper's example gives 4/15), while its candidate
split score is an unweighted sum of squared deviations. The apparent k index
inside that sum is interpreted as j. Lowest-dose efficacy ties are explicit.

The rendered author R program visibly differs: fixed prior calibration,
inclusive safety cutoff and retention of at least one dose. Its less-than
expressions are truncated by the web reader, and a direct download failed DNS
under local network restrictions. No raw file was saved. An alternative text
retrieval service returned a credits-exhausted error. Those access limitations
do not justify inventing the missing source or claiming native equivalence.
The Python contract follows the paper and stops when no dose is admissible.

Safety overrides normal one-level movement when necessary. Observed doses may
have gaps: starting above the lowest dose and later moving directly to an
admissible dose can create such a history. Fit only observed rates in dose order
and map them back to their original indices; do not invent untried outcomes.
Final selection is restricted to observed admissible doses.

The independent base-R reference uses the weighted min-max characterization of
isotonic regression, rather than the implementation's PAVA stack. It covers
seven efficacy curves, including unequal counts, plateaus and multiple peaks.
Separate direct-alpha beta-CDF inversion and complementary beta tails cover
five priors and zero, ordinary and concentrated data. This is mathematical
cross-validation, not a native app comparison or full simulation-table replication.

Core checkpoint 0cf8dd5 was integrated as 529205f. Luna's three focused tests,
Ruff and targeted mypy passed. Root generated independent R fixtures and ran
both reference tests with warnings as errors: two passed in 1.44 seconds.
The external timing wrapper could not read the sandboxed kernel clock rate,
so it exited nonzero after pytest succeeded and supplied no peak-memory metric.
This does not change the observed test result; use Python's process resource
measurement for the later isolated-wheel check. A separate memory-pressure
reading reported 48% free system memory.

## Integrated validation

Luna's c168ada and 818c2c1 were integrated as e8982fc and c6154f3. The former
preserves a tiny beta prior in an all-toxic cohort by forming failures before
adding the prior; the latter adds the serial simulator. Root review also fixed
the preliminary decreasing-segment transformation and required strict scalar
parsing, finite posterior tails, representable weight ratios and explicit
no-enrollment stopping semantics before integration.

The final R generator runs with warnings treated as errors. Root's combined
MTADF checks passed: six tests with Python warnings as errors in 1.38 seconds;
process elapsed time 1.48 seconds, peak 132.86 MiB, zero process swaps. Ruff
check/format passed for both modules, exports and both test files. Targeted
mypy passed for both numerical modules. No full suite or new CI job was run.

The wheel and source distribution built using cached Hatchling. An isolated
wheel process verified exports and source/catalog byte equality, excluded raw
upstream archives, and executed both documentation examples. Small deterministic
histories independently checked plateau exploration/return, final selection,
early-versus-final safety stopping, a prior preventing all enrollment, strict
safety-cutoff equality and invalid-input rejection without consuming the RNG.
That process took 2.16 seconds, peaked at 115.48 MiB and reported zero swaps.
These are process measurements for bounded validation workloads, not guarantees
about total system memory or every allowable simulation.

Catalog entry 114 moves from pending to partial. Counts are now 62 implemented,
58 partial and 18 pending (138 total). Remaining native app differences are
documented; there is no claim of exact executable output or random-stream
equivalence. UAROET is the next uncovered method with an accessible primary
mathematical paper; see uaroet-next-audit.md.

GitHub publication still has the previously observed approval-required write
restriction under this session's never-approve policy. No alternate write
transport was attempted. The overall goal remains active and incomplete because
scientific coverage can continue locally.

## Full author-reference recovery, 2026-10-04

The full 20,787-byte author R file was recovered in one direct retrieval after
the earlier DNS limitation. Its SHA-256 is
`e27be5581fdcb7c71a7739a4f1b25026dc054896fcbf7134ff9b1a1c8960caac`.
The new [author-reference audit](mtadf-author-audit.md) and
[dependency/reference audit](mtadf-author-reference-audit.md) supersede the
raw-source limitation above for the isotonic workflow. The independent Python
kernel and bounded replay/simulator preserve its fixed prior, inclusive
floor-one safety, equal-weight efficacy fitting, rightmost ties, all-dose
final epsilon rates and distinct fresh/lagged safety-cap timing.

Entry 114 remains partial: the author global `arm::bayesglm` coefficient-mode
rule and local adjacent-window conduct differ from the existing paper-policy
logistic APIs and require separate coverage. The retrieved file is evidence
for that reference program; it does not establish the current live app's exact
runtime or hidden defaults.
