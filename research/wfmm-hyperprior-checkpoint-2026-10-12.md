# WFMM explicit optional shrinkage updates — local checkpoint

Base: 608a15c3e311d3bdba816d8e88932d79c84c0c0f. Last verified remote:
369d3f80b0cb004384b9a2b62c339dbc0514002b. Remote environment access remains
unresolved; the separate fresh-instance identity diagnostic returned Forbidden.
That does not establish token validity or cause. No authentication retries,
alternate transports, credential changes or remote writes were attempted here.

The coefficient sampler now optionally updates partition-level pi and tau using
explicit independent Beta and inverse-gamma shape/scale priors. The conditionals
use included counts and included squared coefficients; included zero values
still count. Parameters broadcast by fixed effect and sorted retained partition.
Masks freeze complete pairs; all-false masks consume no hyperparameter draws.
Retained pi/tau samples, labels and masks are owned read-only arrays. Thinning
preserves exact dense-run subsequences. The separate random-effect work budget
now also counts discarded post-warmup transitions.

The optional mode/variance calibration helper is an independently defined
scientific policy, not a copy or correction presented as native intent. Beta
parameters solve the mode and variance equations; inverse-gamma shapes exceed
two and reproduce requested finite variances within a checked tolerance. Direct
proper shape/scale priors can have infinite moments. Extreme calibration ratios
are explicitly bounded/rejected rather than silently approximated.

Saved studies serialize the explicit prior object, initialize active shared pi
at its Beta mean and tau at its inverse-gamma mode, and preserve empirical-Bayes
initialization holds. Masked pairs retain empirical-Bayes starts and must satisfy
within-partition constancy. Historical studies default to disabled updating.
Reports contain pi/tau means, requested quantiles, R-hat and MCSE. This is not a
native MAT/report interoperability claim or complete native report-field parity.

Independent checks enumerate all four inclusion patterns in a two-coefficient
model and numerically integrate the shared inverse-gamma slab variance. They
compare inclusion probabilities and pi/tau posterior means against the sampler.
A zero-information design checks prior marginals; exact conditional algebra,
included zeros, masked updates, thinning, boundary starts, historical JSON and
saved replay are also covered. Mode/variance equations use SciPy distribution
moments as independent checks. No original executable or copied binary enters
production artifacts.

WFMM remains partial. Outstanding gates include interval/reduced boundary scope,
imputed Vstar semantics, native file/report compatibility and separate exact
RNG identity. The guide's missing_data flag distinguishes normal Y from imputed
Vstar; it is not yet evidence that arbitrary missing observations are supported.
Native conversion discrepancies remain documented separately. Clinical validity
is not established. Catalog counts remain 103 implemented / 31 partial / 4 pending.

Final acceptance evidence belongs in wfmm-hyperprior-validation-2026-10-12.json.

## Acceptance

77,167 full tests passed in 490.23s with all extras, four workers and warnings as
errors. Focused WFMM: 1,035 passed in 44.72s. Fresh installed wheel outside the
checkout: 1,063 passed in 103.71s, including 28 plotting/TIFF checks; six installed
examples and dependency checks passed. All 779 production Python files, five
JSON files, five compressed data tables and 1,610 test/fixture files are unchanged.
Both artifacts match all 789 package files and 29 retained notices; no original
binaries are included. These counts explicitly include the compressed data files.
Lint, formatting, types and build passed. Artifacts predate final research
acceptance and the new example; their production bytes match the tested source.
