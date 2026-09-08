# CUMINC coverage audit

Catalog entry 39 is implemented. The audit covers the statistical and plotting
workflows distributed in `CUMINC_V1.tar.gz`; the Python package replaces the
S-PLUS interpreter, loader, device setup and generated installation Makefile.
No archived code or binary is bundled. Implementation details, conventions and
examples are in [cuminc.md](cuminc.md).

Audited source/data SHA-256 hashes:

| File | SHA-256 |
| --- | --- |
| cuminc.f | a88c7c37f56a5dfdb9a931e19f7bcc5690abef9fe455ddb597c0f519e74c7cd1 |
| cuminc.s | 6d261d55e31b1450f42462c5c8de7bed0beb00cb8379dd6f43638e54aa4e68f4 |
| summary.cuminc.s | b4f0e504fd36a424f631065c09bc3ca1b8bad064020e57b894838a4f4f203774 |
| print.cuminc.s | e9fe710bca990f737796aa5c91284a32959c6760433f7cbb0f7f6e8c3813479b |
| plot.cuminc.s | c6ac918eab956b66ebd4d6d665e07e478aa557ef873b417053a5286ffb75401b |
| plot.cuminc.one.s | bce47b6431373de28703039d5608f41927a62d2e2324aa9e1a063944f3a6950a |
| test.data | 451b42ca57923f0058966e97f64f2d4e863efce88f416b0c08679dd77a47e275 |
| test.q | 0619cb13007ecde4b89daa1c1b7c9d91ed3f9e0f0ea4e2b1fc53dbc5f0f72edb |

| Original capability | Python implementation | Evidence |
| --- | --- | --- |
| CINC competing-risk curves and Aalen variance | `cumulative_incidence` | 13 unchanged Fortran fixtures; independent empirical incidence, risk-set and variance checks |
| Tied failures/censoring, curve corners | Native-compatible corners and right-continuous `.at` | Tied censoring, time-zero jumps, singleton and absent-target tests |
| CRST/CRSTM weighted and stratified group comparisons | `gray_test` | 57 native score/covariance cases, 2–4 groups, four rho values, partial group presence within strata |
| Quadratic statistic and p-value | Linear solve and chi-square upper tail | Hand-calculated score/variance, closed-form chi-square tail identities, reference-group invariance |
| Singular comparison | Explicit None statistic/p-value and rank | Fully and partially unidentifiable comparisons; no silent group deletion |
| All causes within all groups; optional group/strata | `cuminc` and immutable keyed result mappings | 48 native Gray cases through string labels, native single-group curves, full individual counts |
| Complete-case deletion | Explicit `missing="drop"`, counted exclusions | Joint time/cause/group/stratum deletion and invalid/empty-data checks |
| Group/event counts and tests in printed output | `CumIncStudy.report` | Counts checked against individual observations; NA output for singular comparisons |
| Calling context | Retained censor, rho, confidence, group/cause/stratum labels | Single-group settings regression; explicit report metadata instead of S expression text |
| Summary estimates, errors and normal confidence limits | `IncidenceSummary` | Archived formula on native curves at two levels using independent normal quantiles; boundary and nesting checks |
| Summary times and group/cause subsets | `.summaries` with exact labels | Scalar/vector order, duplicates, step limits, empty queries and invalid selections |
| Printed precision and file output | Configurable significant digits; UTF-8 `write_report` | Numeric round trip, file replacement and propagated I/O errors |
| All-curves overlay | `plot_cuminc(overlay=True)` | Exact plotted coordinates, legends, axis labels and PNG export |
| Selected curves, separate panels with confidence limits | `plot_cuminc(overlay=False, causes=..., groups=...)` | Cause/group grid and one-group horizontal layout; exact interval coordinates and SVG export |
| Legend coordinates and graphical control | Data-coordinate `legend_at`, labels, returned Matplotlib axes | Coordinate-transform checks and no global style changes |
| Supplied test workflow | Native fixtures and rendered plots of `test.data` | Both causes, both groups, stratified/unstratified tests; overlay and 2×2 panel visual inspection |

## Explicit adaptations

The source S wrapper applies a normal tail to a squared group statistic; the
Python result uses the chi-square distribution specified in CRSTM. Event counts
count people rather than distinct event times. Confidence-limit headers reflect
the requested level rather than always saying 95%. Missing-data deletion is
explicit instead of silent. Numerical labels or string labels are supported;
each vector must be homogeneous, rather than relying on implicit S coercion.

Requested summary times use documented right-continuous lookup and flat extension
beyond follow-up, replacing the unavailable S-PLUS survival-index helper. Default
summaries preserve stored left/right corners. Selection uses exact labels rather
than partial string matching. Report precision is significant digits rather than
the source's decimal-place rounding. The report includes concrete analysis
settings instead of a literal S calling expression.

Source Aalen finite-risk-set variance conventions, including terminal singleton
behavior, are retained and documented. Empty/all-censored data behavior and
nonfinite input handling are explicit. Plots preserve the source statistical
content and panel organization, with modern styling and finite axes for zero
curves; they do not reproduce device pixels. The original loader/Makefile and
empty force-loading helper are installation machinery, replaced by the package
build and optional plotting dependency.

Curve sorting/aggregation and queries use NumPy, and Gray influence/covariance
updates use array and matrix operations across groups. Plotting and summaries
reuse fitted curves and tests. No claim of a measured speedup over Fortran is made.
The archive's source, help files, sample and installation documentation were
reviewed; no additional statistical workflow remains pending for this entry.
