# WFMM thinning and pyramid spatial pass controls (local checkpoint)

Base 00ac2e3f251d2a271295c00e5e2420eb7ce79b47 is locally committed and saved in
private Library, but not pushed. Last verified remote base was
369d3f80b0cb004384b9a2b62c339dbc0514002b. Environment GitHub authentication is
invalid; remote retries are stopped. No alternate credentials/transports or
permissions/settings changes are used. This checkpoint is local until a
separately verified push succeeds.

The sampler now accepts thin=1..100. Draws counts retained samples, so total
transitions per chain are warmup + draws*thin. The retained indices are thin-1,
2*thin-1, ... in the post-warmup stream. Conditional random effects are still
sampled on discarded post-warmup transitions. Exact comparison with dense runs
checks fixed effects, indicators, variances, random effects, log likelihoods,
proposal/acceptance counts and final RNG state, for thinning 1/2/3/5 and fixed/
estimated variances. All transitions count toward work limits. This is a storage
control, not an efficiency or native-RNG claim. Saved historical studies default
to one, and reports distinguish retained draws and iterations; held runs have
zero retained draws and no completed iteration count.

Pyramid pass flags select whole spatial levels, including every orientation.
Positive highpass removes the approximation and the requested coarsest details;
lowpass removes finest details. Native descriptions are reversed, as previously
audited for 1D. Forty native analysis-only configurations cover 2D/3D, odd/even,
one/two levels, db1/db2 and periodic/reflection/zero boundaries; 15 selections
include one unfiltered rectangular baseline, and 25 native failures are retained.
All eight nonzero rectangular flag cases fail; rectangular flags remain rejected.
Pyramid empty sets raise a clear error. Independent Haar low/high reconstruction
and orthogonality checks validate mathematical meaning. Filtering precedes
compression, preserving original indices and full-grid restoration.

The pass fixture and the earlier 52 spatial maps regenerate byte-identically
through the refactored child-only capture helper. No native fitting success is
inferred from analysis capture. Interval/reduced failures remain qualified.
A separate [hyperprior audit](wfmm-hyperprior-audit-2026-10-12.md) recovers 16
native conversion outputs and exposes finite-variance mismatches; no flawed
conversion is copied into production. Proper optional pi/tau conditional updates
and remaining imputed-data/interval scope still need closure. WFMM remains
partial and catalog counts remain 103 implemented / 31 partial / 4 pending.

Final test/artifact evidence is recorded in the adjacent validation manifest.

## Acceptance

77,132 full tests passed in 464.43s (all extras, warnings as errors), 1,000
focused checks in 30.83s, and 1,028 installed-wheel checks in 79.94s plus five
examples. All 778 production Python files, five data files and 1,609 test/fixture
files stayed unchanged. Lint, formatting, types, build and dependency checks
pass. Both artifacts retain 783 matching package files and 29 unchanged notices,
with original binaries excluded. Artifacts precede final research evidence.
See [validation](wfmm-controls-validation-2026-10-12.json). No remote push claimed.
