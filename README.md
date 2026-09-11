# mdanderson-stats

Work in progress: one Python package for the methods in the [MD Anderson
biostatistics software catalog](https://biostatistics.mdanderson.org/SoftwareDownload).
The scope includes all desktop and online entries. A catalog entry is not an
implementation; `catalog.json` explicitly tracks pending work and validation.

The implementation uses NumPy broadcasting and compiled SciPy numerical kernels.
Numba will be considered for measured simulation bottlenecks. This is an independent
project and is not an MD Anderson release.

## Development

```
uv sync --group dev --extra plot
uv run --extra plot pytest
uv run ruff check .
uv run mypy src
uv build
```

Refresh the catalog with `uv run python tools/inventory.py`. Original downloads
and research snapshots are kept under the ignored `research/raw/` directory;
source URLs and validation evidence are recorded separately.

## Numerical API

```python
from mdanderson_stats import binomial_interval, poisson_interval

binomial_interval(12, 30)  # exact 95% Clopper–Pearson interval
poisson_interval(10000, confidence=0.99, exposure=100)
```

Inputs broadcast as NumPy arrays. Counts must be nonnegative integers, exposure
must be positive, and confidence must be strictly between zero and one.

`poisson_interval` uses the standard exact (Garwood) interval. For reproducing
BP1CI 2.0 output, `bp1ci_poisson_interval` uses its documented implementation's
different lower-tail inversion. That compatibility function is not presented as
a corrected exact interval; see `docs/validation.md`.

`bp1ci_binomial_interval` extends the beta-tail formulas to fractional counts.
`bp1ci` provides the original percentage and success/failure or total-trial entry
conventions, plus a readable result table. See [BP1CI coverage](docs/bp1ci.md).

Normal tail probabilities (`normal_tails`), chi-square goodness-of-fit tests
(`chi_square_gof`), and monotone function inversion (`invert_monotone`) are also
available. See [numerical methods](docs/numerical-methods.md) for examples,
compatibility differences, and validation against the original programs.

`range2` and `kwrange` provide group-mean and rank-based multiple-range comparisons.
[Range-test documentation](docs/range-tests.md) explains the critical-value
conventions and the optional original grouping behavior. Adapted portions carry
the original [redistribution notices](THIRD_PARTY_NOTICES.md).

MULTI's nine adjustment/threshold procedures, two sharpened procedures, Schweder
line fitting and bootstrap are available through `multiple_testing`,
`sharpened_testing`, `schweder_fit`, and `schweder_bootstrap`.
`order_statistic_diagnostics` provides the S library's OSFIT diagnostics, and
`clustered_pvalues` simulates dependent one-sided p-values with explicit random state.
`nonparametric_pvalues` implements the S library's local-quadratic diagnostic with
stable regression solves and explicit errors for undefined fits.
`nonparametric_testing` provides the desktop's separate subset fitting,
local bandwidth selection, and rejection decisions.
`BetaMixture`, `beta_mixture_start`, `fit_beta_mixture_em`, and
`fit_beta_mixture_ml` provide mixture evaluation, posterior null probabilities,
initialization, EM fitting, and direct constrained likelihood fitting.
See [beta mixtures](docs/beta-mixtures.md) for endpoint conventions, validation,
and model-selection semantics. `select_beta_mixture` implements the three S
stopping rules, with an explicit desktop workflow option. `fit_beta_mixture_k`
supports manual component counts; `beta_mixture_bootstrap` refits simulated samples
for CVM checks.
`beta_mixture_testing` returns the desktop reciprocal-density scores and decisions,
with explicit rank-order or entered-order processing.
See [multiple testing](docs/multiple-testing.md) for examples, historical naming
differences, and the MULTI features that remain pending.

`plot_schweder(fit)` reproduces the S plot; install the optional `plot` extra
(`uv sync --extra plot` in this checkout). `write_schweder_data(fit, path)` exports
all plot coordinates as CSV without requiring Matplotlib. See the
[Schweder output example](docs/multiple-testing.md#schweder-plot-and-coordinate-export).

STUKEL's `stukel_log_odds`, `stukel_probability`, and `predict_stukel` provide
its generalized logistic link and prediction from supplied coefficients.
`stukel_objective` evaluates its likelihood and analytic derivatives for all six
parameter families. `fit_stukel` fits those families with bounds, dispersion and
observed-information covariance. `scan_stukel` profiles likelihood over fixed-shape grids.
`plot_stukel` provides dose/link plots with the plot extra; `format_stukel` returns
regression tables. `stukel_demo("beetles")` or `stukel_demo("warsaw")` runs the bundled
six-family comparison; `compare_stukel` accepts supplied data. See
[STUKEL coverage, examples, and compatibility differences](docs/stukel.md).
`parse_multi_data` and `read_multi_data` import MULTI p-value text with explicit
ignored-token diagnostics and original input indices.
`MultiSession` runs procedures on replaceable data and writes structured reports
with settings, results, diagnostics, and random-state provenance. Its
`format_report` and `write_text_report` provide readable Markdown tables.
[The MULTI coverage audit](docs/multi-coverage.md) records its remaining I/O gaps.

ONESAMPLE's `binomial_test` and `poisson_test` return inclusive one-sided p-values,
with explicit compatibility cutoffs. `one_sample` exposes all four test/interval
operations, both binomial entry modes, and readable reports with file output. See
[ONESAMPLE coverage and validation](docs/onesample.md).

`KStageBinomial` implements KSB1CI confidence intervals for binomial trials with
early stopping, including vectorized stage-ordered tails and design reports.
See [KSB1CI definitions, validation and examples](docs/ksb1ci.md).

`ksbin1_operating_characteristics` evaluates fixed multistage binomial designs,
including rejection/quitting probabilities and expected sample sizes.
`ksbin1_study` adds single-stage comparison, boundary assistance, design revision,
and report/design file output. See [KSBIN1 coverage and validation](docs/ksbin1.md).

`ksbin2_statistic` and `ksbin2_ordering` provide vectorized two-sample binomial
evidence scores and tied outcome groups. `ksbin2_probability_table` adds ordinary
single-stage power and null-grid significance. [KSBIN2 coverage](docs/ksbin2.md)
documents its mid-p reporting, rejection-region selection and multistage workflows.

`KStageTwoSampleBinomial` evaluates fixed KSBIN2 multistage designs, with cached
surviving paths, broadcast probability pairs and expected sample sizes per group.

`ksbin2_boundary_table` adds cumulative rejection-boundary assistance and optional
reference-completion power-loss tables for those multistage designs.

`ksbin2_study` provides full null-grid scans, paired-hypothesis numerical reports,
and study revision, keeping actual null probabilities separate from grid maxima.

KSBIN2 decision grids and inclusive count-range reports can be inspected and
exported with `decision_grid`, `region_report`, and `write_regions`.

`single_design_precision` evaluates local slope and quantile precision for fixed
logistic/log-log dose-response designs. `single_two_sample_precision` evaluates
location or slope differences with the other parameter shared across groups.
`single_uniform_criterion` averages these precision criteria over independent
uniform parameter priors using batched quadrature.
`single_normal_criterion` handles correlated normal/log-normal latent priors,
with an explicit option to reproduce SINGLE's original covariance scaling.
`single_prior_parameters` converts marginal moments and latent correlations,
with exact and original log-normal conversion options.
`single_design_correlation` supplies reference-design prior correlations.
`single_optimize_allocations` chooses continuous subject counts at fixed dose
points for one-sample point-prior slope or quantile precision.
`single_optimize_prior_allocations` optimizes arithmetic or harmonic prior-averaged
SD/variance using explicit quadrature nodes and analytic gradients.
Two-sample optimization accepts `group_totals` to fix each group size; their sum
must equal `total_subjects`. Without that option,
`single_optimize_two_sample_allocations` allocates a shared subject total across
both groups under point or weighted priors.
`single_optimize_design` jointly moves dose locations and allocations for a fixed
number of dose entries, supporting one/two samples and weighted priors.
Optimized SINGLE results provide `report` and `write_report` for TSV design and
convergence summaries.
`single_search_design` scans starting doses, adds dose entries, and retains a
reviewable history under SINGLE's relative-improvement stopping rule.
`SingleStudySpecification` adds complete settings/prior reports, study revision
and JSON input replay.
[SINGLE coverage](docs/single-coverage.md) records the completed source/manual
audit, validation evidence and documented numerical/solver substitutions.

`SeqBinDesign` constructs SEQBIN beta-posterior sequential or group-sequential
binomial boundaries and computes exact stopping probabilities and expected sample
sizes. `seqbin_prior` converts prior mean/size inputs; `seqbin_calibrate` and
`seqbin_calibrate_tails` choose attainable frequentist error levels.
`SeqBinStudySpecification` adds full numerical reports, compact tables, revision
and JSON replay. [SEQBIN coverage](docs/seqbin-coverage.md) records the completed
source/manual audit, compatibility differences and batch benchmarks.

`cumulative_incidence` estimates competing-risk incidence curves and Aalen
variances; `gray_test` compares groups with optional stratification and weighted
Gray tests. `cuminc` combines all causes and groups with pointwise confidence
intervals and numerical reports. `plot_cuminc` adds overlays and confidence-limit
panels. [CUMINC coverage](docs/cuminc-coverage.md) records the completed source audit
and compatibility differences.

`muhaz_fixed` adds fixed-bandwidth censored-data hazard smoothing with four
kernels and boundary corrections. [MUHAZ coverage](docs/muhaz-coverage.md) records the completed numerical,
reporting and plotting audit, compatibility differences and batching benchmarks.

`pehaz` adds MUHAZ's piecewise-exponential estimator, including bin event counts,
person-time, risk counts and numerical reports, with explicit legacy bin semantics.

`kphaz` adds stratified Nelson and product-limit hazard/variance estimates over
consecutive failure-time intervals, with explicit source compatibility.

`muhaz_mse` computes pilot-based bias, variance and MSE across candidate
bandwidths, with per-cell quadrature convergence diagnostics.

`muhaz_global` selects a common hazard bandwidth using the MSE grid, with
documented default settings, complete diagnostics and a single-bandwidth bypass.

`muhaz_local` selects and smooths pointwise bandwidths, retaining full candidate
MSE diagnostics and evaluating the resulting variable-bandwidth hazard in chunks.

`muhaz_neighbor_bandwidths` provides failure-count and survival-mass radii;
`muhaz_knn` selects the neighbor count, smooths bandwidths and fits the hazard.

`summarize_muhaz` provides structured settings and results, significant-digit
text reports, explicit bypass status and quadrature convergence counts.

`plot_muhaz`, `plot_pehaz` and `plot_kphaz` provide optional hazard plots and
overlays, including bin edges, strata and gaps for undefined estimates.

`exploratory_survival` provides EXPSURV survival curves, plotting corners and
inverse-survival queries. The complete port is documented in the
[coverage audit](docs/expsurv-coverage.md), including validation and benchmarks.

`survival_cutpoint` and `plot_cutpoint` add EXPSURV's covariate split comparisons
and linked density/survival slider, including explicit empty-group handling.

`plot_survival_alignment` adds EXPSURV's accelerated-failure and proportional-hazards
alignment sliders, reusing fitted survival curves.

`plot_survival_scatter` links EXPSURV covariate selections to a survival curve,
with rectangle selection, Shift-add and programmatic original-row selection.

`plot_event_scatter` links covariate selection to EXPSURV event charts, showing
follow-up segments at arrival times with failure and censoring endpoints.

`censored_box` and `plot_censored_box` add EXPSURV life-table box geometry and
linked selection, including unreached quartiles and all-censored samples.

`ExploratoryTable` supplies EXPSURV numeric file round trips and stable aligned
sorting; `generate_exponential_samples` supplies reproducible two-group examples.

`generate_exploratory_data` supplies EXPSURV covariates, arrival times and
follow-up with sample-standardized survival and study-end censoring.

EXPSURV linked matrices now support point clicks and continuous brushing: B
toggles mode, +/- resizes the brush, Shift adds selections and Escape clears.

`contingency_chi_square` starts [CTA](docs/cta.md) with batched contingency-table
statistics, native Fortran comparisons and explicit correction conventions.

`mcnemar_analysis` adds CTA paired-category statistics and its pooled/heterogeneity
decomposition, with explicit handling of pairs without discordant observations.

`cohen_kappa` adds CTA agreement coefficients and variances, with corrected
multinomial calculations and an explicit legacy-formula option.

`diagnostic_accuracy` adds CTA sensitivity, specificity and predictive values
with explicit table orientation and probability standard errors.

`odds_ratio` adds CTA relative odds and log-Wald limits, with explicit
compatibility for the source confidence-limit formula.

`fisher_exact` adds CTA fixed-margin probabilities, standard exact-test
alternatives and explicit compatibility for the original selected, truncated tail.

`binomial_comparison` adds CTA conditional Poisson-model comparisons with
explicit event selection, inclusive tails and corrected/source two-sided conventions.

`CTAStudySpecification` combines the CTA analyses with reusable settings,
automatic Fisher selection, independent study snapshots and UTF-8 summary reports.

CTA study reports also provide optional per-cell and per-probability listings,
with explicit output-size limits and source-compatible term traversal.

CTA is complete, including all analysis workflows and detailed reports. Its
[source audit](docs/cta-coverage.md) records native comparisons, independent
mathematical checks and measured batch performance.

`ranlist_seeds`, `ranlist_integers` and `ranlist_uniform` begin
[RANLIST](docs/ranlist.md) with reproducible phrase seeds and indexed random
streams, validated against the original Fortran.

`ranlist_unrestricted` adds weighted treatment assignments by patient number
and stream, with explicit source rounding compatibility.

`ranlist_restricted` adds fixed and random balance blocks, with exact treatment
counts per completed block and an explicit mode for the archived source's
sampling and indexed allocation behavior.

`RanlistSpecification` and `RanlistSession` add named strata, reusable list
definitions, batch enrollment with per-stratum counters and prior-patient
inquiries. Sessions return immutable updates after successful allocation.

RANLIST sessions can now be saved and resumed as validated JSON snapshots, or
exchanged with the original program through fixed-width parameter files.
Four complete native Fortran enrollment/inquiry/save workflows match Python.

`ranlist_summary` and `ranlist_report` provide parameter/enrollment summaries and
paginated treatment lists for enrolled or planned patients. Eight native print
workflows validate assignments and page boundaries.

RANLIST is complete for the archived source/manual workflows. Its
[source audit](docs/ranlist-coverage.md) covers creation, enrollment, inquiry,
persistence and printing, including native source defects and measured batch
performance. `ranlist_starting_seeds` reproduces original setup phrase handling.

`RandlibGenerator` implements [RANDLIB](docs/randlib.md): 32 independent
streams, antithetic control, block reset/advancement, phrase/time seeding,
uniform and bounded-integer draws, permutations, and exponential, normal,
gamma, beta, chi-square, F, binomial, Poisson, negative-binomial, multinomial
and multivariate-normal sampling. Noncentral chi-square and F are included.
`RandlibMultivariateNormal` prepares immutable reusable covariance factors.

Vectorized defaults provide mathematical distribution sampling with numerical
checks. Legacy modes retain recorded C/Fortran algorithms, rounding and draw
consumption. Native fixtures validate values, factors and generator states;
additional tests cover moments, tails, covariance, resource limits and rollback.
The [archive audit](docs/randlib-coverage.md) accounts for all 88 members and
documents repairs, compatibility limits and the treatment of demonstration
programs. [Benchmarks](docs/randlib-benchmark.json) compare batched draws with
repeated scalar calls to the same Python API.

`multinomial_power` provides exact one-sample multinomial power with Pearson
chi-square and likelihood-ratio ordering, multiple alternatives, and complete
tie groups. See [MULTINOMPOW](docs/multinomial-power.md) for validation and the
complete archive and report coverage.

`tdtasp_genetics` provides TDTASP's vectorized two-locus family and offspring
probabilities, with explicit compatibility for the original ASP weighting.
`tdtasp_ascertainment` adds family/individual selection, conditional offspring
moments and contribution-weighted test probabilities. See [TDTASP](docs/tdtasp.md)
for native validation, model assumptions and archive coverage.

`tdtasp_fixed_power` and `tdtasp_power` calculate discrete binomial power and
average it over eligible-family counts, with explicit mean-contribution and
source-compatibility conventions described in the TDTASP notes.

`tdtasp_fixed_sample_size` and `tdtasp_sample_size` find the first qualifying
integer design within specified bounds, preserving discrete power oscillations.
The family search uses cached conditional powers and a monotone upper bound;
[benchmarks](docs/tdtasp-search-benchmark.json) compare it with exhaustive scanning.

`tdtasp_study` runs genetics, ascertainment and either power or sample-size
calculations in one call. Search results include a fixed-observation comparison;
`format_tdtasp_study` produces a reproducible text report.
`TDTASPTemplate`, `parse_tdtasp_template` and
`format_tdtasp_template` support validated legacy forms and file-to-study workflows.

The [TDTASP archive audit](docs/tdtasp-coverage.md) covers all 44 archived files.


`cdf_beta`, `cum_beta`, `ccum_beta` and `inv_beta` provide CDFLIB90
beta tails, quantiles and shape inversions. They broadcast arrays and retain
small probability/coordinate complements. [CDFLIB90 notes](docs/cdflib90.md)
describe native validation, corrected source defects and numerical limits.


`cdf_normal`, `cum_normal`, `ccum_normal` and `inv_normal` add CDFLIB90's normal
location/scale calculations, including mean and standard-deviation inversion,
small complementary probabilities and explicit rejection of unidentified scales.


`cdf_gamma`, `cum_gamma`, `ccum_gamma` and `inv_gamma` provide gamma tails,
quantiles, shape and rate inversions. The explicit `rate` argument preserves the
archived implementation's convention: its parameter named SCALE multiplies x.


`cdf_chisq`, `cum_chisq`, `ccum_chisq` and `inv_chisq` add chi-square tails,
quantiles and inversion for real degrees of freedom, with the original df bounds.


`cdf_poisson`, `cum_poisson`, `ccum_poisson` and `inv_poisson` preserve
CDFLIB90's fractional-count Poisson extension, with count and mean inversions.


`cdf_neg_binomial`, `cum_neg_binomial`, `ccum_neg_binomial` and
`inv_neg_binomial` add fractional failure/success counts and probability inversion,
including explicit zero-success behavior and preserved small complements.


The [CDFLIB90 inventory](docs/cdflib90-coverage.md) accounts for all 106 archived
files and maps the complete distribution, legacy-library and public-support scope.


`cdf_t`, `cum_t`, `ccum_t` and `inv_t` provide Student's t tails, quantiles and
bounded degrees-of-freedom inversion with preserved small probability tails.


`cdf_binomial`, `cum_binomial`, `ccum_binomial` and `inv_binomial` preserve the
continuous binomial extension, with success-count, trial-count and probability
inversion. Native validation covers both archived binomial source versions.


`cdf_f`, `cum_f`, `ccum_f` and `inv_f` implement the F95 F-distribution tails
and quantiles. The older C/F77 degrees-of-freedom inversion modes are available
through the separate `cdff` interface described below.

`cdf_nc_chisq`, `cum_nc_chisq`, `ccum_nc_chisq` and `inv_nc_chisq` provide
noncentral chi-square tails, quantiles and bounded degrees-of-freedom/noncentrality
inversions. Validation includes high-precision Poisson mixtures, native source
profiles and explicit detection of inconsistent extreme-tail inverse results.

`cdf_nc_f`, `cum_nc_f`, `ccum_nc_f` and `inv_nc_f` add noncentral F tails,
quantiles and noncentrality inversion, with central-case correction and bounded
refinement for failed inverse kernels. The separate legacy interface also
implements both df inversions.

`cdf_nc_t`, `cum_nc_t`, `ccum_nc_t` and `inv_nc_t` add noncentral t tails and
all parameter inversions, with explicit brackets for multiple df roots and
quadrature repair for small negative tails. All twelve F95 distribution modules
are implemented with documented Python semantics.

`cdff` and `cumf` implement the [legacy DCDFLIB F interface](docs/dcdflib-f.md),
including numerator/denominator df inversions, source search bounds through
1e-100..1e100, and explicit brackets for multiple roots. Both archived C and
Fortran implementations provide independent reference fixtures.

`cdffnc` and `cumfnc` implement the [legacy noncentral F interface](docs/dcdflib-nc-f.md),
including both df inversions, wider input domains and the source's ignored-q
inversion contract. Tests select multiple roots, retain native false-success
evidence and independently check all modes using high-precision beta mixtures.

`cdfnor` and `cumnor` implement the [legacy normal interface](docs/dcdflib-normal.md):
tails and all parameter inversions over unrestricted finite locations and positive
scales, with overflow repair, subnormal tails and independent native validation.

`cdft` and `cumt` implement the [legacy Student t interface](docs/dcdflib-t.md),
with wider input domains, df inversion down to the legacy search limit and
logarithmic repairs for representable tails lost by the original implementation.

`cdfgam` and `cumgam` implement the [legacy gamma interface](docs/dcdflib-gamma.md),
including all four computed groups, explicit rate semantics, wide finite domains,
logarithmic scaling and tiny-shape tail repairs validated against independent
high-precision calculations and unchanged C/F77 references.

`cdfchi` and `cumchi` implement the [legacy chi-square interface](docs/dcdflib-chisq.md),
with wide finite inputs, bounded x/df inversions, subnormal rounding repairs and
unchanged C/F77 references checked against independent high-precision identities.

`cdfpoi` and `cumpoi` implement the [legacy Poisson interface](docs/dcdflib-poisson.md),
including zero mean, wide finite inputs, bounded continuous count/mean inversions
and independent repairs for native overflow and false-success results.

`cdfnbn` and `cumnbn` implement the [legacy negative-binomial interface](docs/dcdflib-neg-binomial.md),
with all four computation modes, wider counts, paired chance inversions and
independently validated repairs for extreme shapes and small inverse targets.

`cdfbin` and `cumbin` implement the [legacy binomial interface](docs/dcdflib-binomial.md),
with separate success/trial search bounds, complementary chance inversion and
independent checks of native process failures, tiny counts and wide inputs.

`cdfbet` and `cumbet` implement the [legacy beta interface](docs/dcdflib-beta.md),
with wide positive shapes, bounded shape inversions, complementary quantiles
and positive recurrences that preserve tails when both shapes are tiny.

`cdfchn` and `cumchn` implement the [legacy noncentral chi-square interface](docs/dcdflib-nc-chisq.md),
with ignored-q inversion, wide inputs, bounded searches and independent repairs
for native small-tail truncation and invalid wide central results.

`cdftnc` and `cumtnc` implement the [signed legacy noncentral-t interface](docs/dcdflib-nc-t.md),
including all four modes, explicit df brackets within the executable bounds,
and conditional-tail repairs for wide inputs and subnormal probabilities.
All twelve legacy distribution families have independent C/F77 validation.

`sort_list` implements the [CDFLIB sorting generic](docs/cdflib-sort.md), with
all four value types, prefix sorting and custom comparators. It preserves stable
ties and full string contents, repairing native duplicate and truncation defects.

The [CDFLIB string module](docs/cdflib-strings.md) supplies ASCII-only case
conversion and `qlex`, a reentrant command lexer with safe quoted strings,
explicit malformed-token/overflow results and bounded exponent processing.

The [elementary CDFLIB helpers](docs/cdflib-elementary.md) provide stable
`alnrel`, `rexp`, `rlog`, `rlog1` and batched `evaluate_polynomial`, including
independent high-precision validation of small and subnormal results.

The [CDFLIB error/exponential helpers](docs/cdflib-error-exponential.md) add
`erf`, `erfc1`, `esum` and `exparg`, with batch evaluation, preserved subnormal
complementary-error tails and repaired intermediate exponential overflow.

The [CDFLIB gamma/digamma helpers](docs/cdflib-gamma-support.md) add `alngam`,
`gamln`, `log_gamma`, `gamln1`, `gam1`, `gamma` and `psi`, preserving their distinct
real domains and repairing lost remainders, subnormal tails and intermediate overflow.

The [CDFLIB gamma-ratio foundations](docs/cdflib-gamma-ratios.md) add `algdiv`,
`bcorr` and `gsumln`, preserving tiny ratios, subnormal corrections and close sums.

The [CDFLIB beta/combinatorial helpers](docs/cdflib-beta-support.md) add `betaln`,
`log_beta` and real-valued `log_bicoef`, retaining large-shape and tiny-result accuracy.

The [CDFLIB gamma scaling factor](docs/cdflib-gamma-factor.md) adds `rcomp`,
including huge centers, tiny signed shapes and real reciprocal-gamma continuation.

The [CDFLIB incomplete-gamma support](docs/cdflib-incomplete-gamma.md) adds
`grat1` and `gratio`, with explicit source contracts and subnormal-tail repairs.

The [CDFLIB beta scaling factors](docs/cdflib-beta-factors.md) add `brcomp` and
`brcmp1`, including compensated large-shape centers and complete exponential scaling.

The [CDFLIB beta shape shift](docs/cdflib-beta-shift.md) adds `bup`, with
positive finite sums, bounded remainders and efficient large-shift paths.

The [CDFLIB tiny-companion beta series](docs/cdflib-fpser.md) adds `fpser`,
with strict source-domain checks, full normalization and subnormal recovery.

The [CDFLIB small-first-shape upper beta tail](docs/cdflib-apser.md) adds
`apser`, preserving tiny complements and repairing native endpoint/overflow failures.

The [CDFLIB beta power series](docs/cdflib-bpser.md) adds `bpser`, with bounded
signed sums, stable near-one evaluation and extreme-shape normalization.

The [CDFLIB accumulated beta increment](docs/cdflib-bgrat.md) adds `bgrat`,
preserving signed accumulators and explicit tiny complementary coordinates.

The [CDFLIB beta tail from a displacement](docs/cdflib-basym.md) adds `basym`,
retaining tiny displacements and bounding asymptotic work and scratch memory.

The [CDFLIB beta continued fraction](docs/cdflib-bfrac.md) adds `bfrac`,
with reflected tails, bounded iteration and extreme-shape normalization.

The [CDFLIB paired beta integral](docs/cdflib-bratio.md) adds `bratio`,
completing all 35 F95 mathematical procedures while preserving tiny paired tails.

The [CDFLIB constants namespace](docs/cdflib-constants.md) preserves all 27 native
parameters, completing the constants and mathematical support modules.

The [CDFLIB root-finder audit](docs/cdflib-root-reference.md) records direct and
reverse-communication contracts and independently identifies native root/state defects.

The [CDFLIB root finders](docs/cdflib-root.md) implement direct and reverse searches
with independent state, corrected exact roots, working tolerances and bounded work.

The [CDFLIB auxiliary namespace](docs/cdflib-aux.md) adds all 13 native distribution
descriptors, batched validation, complement/range helpers and root-state adapters.

The [CDFLIB console](docs/cdflib-console.md) adds reentrant typed numeric and text
input with explicit streams, finite-value validation and bounded retries.

The [CDFLIB numeric list editor](docs/cdflib-number-list.md) adds all eight native
actions, persistent state, stable spacing and corrected duplicate removal.

The [CDFLIB array formatter](docs/cdflib-array-format.md) adds checked fixed/scientific
fields, long records and matching console/report output.

[Message templates and controls](docs/cdflib-message-format.md) complete the F95
console module: repeatable Python substitutions, optional help, suppression and
explicit output routing. All seven F95 support modules now have validated Python
mappings.

The [legacy DCDFLIB support namespace](docs/dcdflib-support.md) adds machine
parameters, polynomial prefix evaluation and checked vectorized C translation
helpers. The [31 legacy mathematical helpers](docs/dcdflib-math.md) also have
direct C/F77 validation, including their distinct exponential-limit contract.
The [normal/t quantile helpers](docs/dcdflib-quantile-helpers.md) preserve the
starting formulas and add refined normal inversion. The [incomplete-gamma
inverse](docs/dcdflib-gamma-inverse.md) adds checked starting values and extreme-tail
repairs. The [legacy root finders](docs/dcdflib-root.md) complete all 49 legacy
support mappings with independent search state and their distinct stopping rule.
The [completion audit](docs/cdflib90-completion.md) verifies all public mappings,
reconciles the manuals and marks CDFLIB90 implemented. The separate
[STATTAB application](docs/stattab-research.md) is implemented. Its
[discrete probability terms](docs/stattab-probability.md) provide vectorized
binomial, negative-binomial and Poisson masses with consistent count truncation,
small complementary chances and explicit degenerate boundaries.

The [STATTAB result layer](docs/stattab-results.md) connects all twelve families
and 42 computed groups to structured broadcast results, source column order,
extra probabilities and neighboring integer rows.

[STATTAB sessions](docs/stattab-sessions.md) now parse positional requests and
reuse completed values safely, including table snapshots and tiny saved complements.
The [console application](docs/stattab-console.md) adds all eight list-editor actions,
formula help and report-file dialogs. Run `python -m mdanderson_stats.stattab`
after installation. The [shared-source audit](docs/stattab-shared-source.md) compiles
259 public imports and reconciles all nineteen shared modules. The
[completion audit](docs/stattab-completion.md) covers all 34 archive files and
25 manual pages, including worked examples and documented corrections.

## SPPCR source audit

The [SPPCR audit](docs/sppcr-research.md) identifies misplaced SOGS documentation,
rebuilds all 27 sources and checks interior allele-frequency fits independently.
The [SPPCR fitting core](docs/sppcr-fit.md) provides vectorized Poisson-mean fits,
observed-information variances and explicit boundary policies.
[Frequency summaries](docs/sppcr-frequencies.md) add calibration, mutant frequency,
delta-method uncertainty and stable forward transforms.
[Data generation](docs/sppcr-generation.md) adds explicit probability models and
reproducible batched binomial samples. [Bootstrap analysis](docs/sppcr-bootstrap.md)
adds replicate fitting, stable population summaries and undefined-frequency
diagnostics. [Confidence limits](docs/sppcr-intervals.md) add support-aware
calibration, reciprocal calibration and transformed frequency intervals.
[Batch input](docs/sppcr-batch.md) adds validated experiment data and canonical
parsing/formatting. [FileMaker-style exports](docs/sppcr-filemaker.md) use the
same validated data model. [Interactive entry](docs/sppcr-interactive.md) adds
bounded data collection and corrections. [Analysis and replicate reports](docs/sppcr-reporting.md)
retain identities, units and diagnostics. [Historical random streams](docs/sppcr-random.md)
are reconciled through RANDLIB with an explicit legacy sampling path.
[Truth designs](docs/sppcr-truth.md) normalize allele weights and prepare explicit
simulation parameters in model DNA units, with bounded interactive entry and
parameter reports. [Analysis workflows](docs/sppcr-analysis.md) connect data or
truth input to modern/historical sampling, fitting and reports.
[File workflows](docs/sppcr-files.md), a [menu/CLI](docs/sppcr-console.md)
and an [output-file dialogue](docs/sppcr-output.md) complete the application.
The [coverage mapping](docs/sppcr-coverage.md) reconciles all 27 source files and
documents deliberate changes to the historical behavior.

## Bayesian updating for binary outcomes

[BU1BB and BU2BB](docs/beta-updating.md) provide vectorized beta-binomial
posterior updates, prior sensitivity, highest-density credible sets, sequential
cohort histories, seeded trial simulation and independent two-arm ordering
probabilities. Distribution values and histories are available for community
analysis and plotting without a Shiny session.

```python
from mdanderson_stats import BetaBinomialPosterior, compare_beta_binomial

control = BetaBinomialPosterior().update(successes=1, failures=3)
treatment = BetaBinomialPosterior(0.5, 0.5).update(successes=6, failures=4)
probability = compare_beta_binomial(control, treatment).treatment_greater
# approximately 0.86272321
```

## Bayesian updating for normal outcomes

[BNORM](docs/normal-updating.md) provides normal and normal–inverse-gamma
conjugate updates for known or unknown observation variance. It supports raw
and summary data, sequential updating, batched prior sensitivity, marginal
mean/variance distributions, credible intervals and data-only confidence
intervals. The documentation identifies and corrects the manual's variance
interval discrepancy.

## Diagnostic populations and ROC curves

[DIAG and DTROC](docs/diagnostic-and-roc.md) provide classification-table estimates,
prevalence-dependent predictive values, population projections and binormal ROC
analysis. Thresholds, density arrays, ROC curves and AUC are available as batched
Python calculations, with direct log tails for extreme diagnostic thresholds.

## Bayesian trial monitoring

[BTOX, BEMPO and BEMPR](docs/bayesian-monitoring.md) provide toxicity,
posterior-efficacy and predictive-efficacy monitoring designs. Their Python APIs
calculate stopping boundaries, patient histories and exact operating
characteristics, including sample-size distributions and stopping-induced bias
in observed rates. Predictive calculations use a vectorized backward
beta-binomial recursion.

## Hierarchical binomial data

[BHM-BLN](docs/hierarchical-binomial.md) adds a logistic-normal hierarchical
model with multiple-chain posterior sampling, independent and pooled beta
comparisons, retained group/global draws and Monte Carlo diagnostics. NumPy
elliptical slice and Gibbs updates replace the source application's JAGS
dependency. Diagnostic limitations and weak-prior mixing concerns are explicit.

## Hierarchical normal data

[BHM-NN](docs/hierarchical-normal.md) adds a normal-normal hierarchy with unknown
group and between-group precisions. Vectorized Gibbs updates retain multiple
chains for group means, the overall mean and both precision levels. Empirical
normal comparisons are explicitly distinguished from posterior mean uncertainty.

## Simon's two-stage design

[Simon2S](docs/simon-two-stage.md) provides exact bounded searches for optimal
and minimax designs, vectorized power and early-stopping probabilities, enrollment
moments, stage decisions and statistical protocol paragraphs. Published design
tables and independent exhaustive enumeration validate the implementation.

## Continuous-endpoint sample size

[Nnormal](docs/continuous-sample-size.md) covers one- and two-sample equality,
equivalence, noninferiority and superiority tests, paired differences, correlation
and balanced ANOVA. Power curves broadcast over scenarios, and integer searches
report achieved and predecessor power. Conservative planning formulas are
distinguished from optional exact normal-model power calculations.

## Binary-endpoint sample size

[Nbinary](docs/binary-sample-size.md) adds one- and two-group proportion planning,
equivalence and directional margin tests, continuity-corrected score planning,
exact Fisher power and enrollment search, paired McNemar tests and kappa agreement
designs. Approximate planning power is explicitly distinguished from exact
binomial enumeration, with all twelve source examples reproduced.

## Time-to-event sample size

[Nsurvival](docs/survival-sample-size.md) adds event and enrollment planning for
one- and two-arm survival comparisons, including equivalence and directional
margins. Median and hazard inputs broadcast over accrual/follow-up scenarios.
Source approximations and exact uniform-accrual event probabilities are explicit,
with conservative and joint-normal equivalence options.

## BOIN dose finding

[BOIN](docs/boin.md) provides single-agent dose decisions, overdose safeguards,
weighted isotonic MTD selection and trial simulation with accelerated titration. Published
boundaries and original R results validate the core. This catalog entry remains
partial while animation and integrated report export are pending. English and
Chinese statistical protocol text includes the numerical decision table. Custom rate cutoffs
can be entered directly, with numerically checked inversion to BOIN alternatives.
The conventional 3+3 comparator supports cohort expansion and matching BOIN
enrollment caps to realized 3+3 sample sizes.

## Keyboard dose finding

[Keyboard](docs/keyboard.md) adds posterior interval decisions, overdose safeguards,
isotonic MTD selection and batched simulation. The paper's complete-key convention
and the R package's adjusted endpoint convention are explicit. Native R comparisons
and independent exact interval probabilities validate the statistical core;
integrated reports remain pending.

## TITE-Keyboard interim decisions

[TITE-Keyboard](docs/tite-keyboard.md) adds uniform and informative follow-up
weights, effective sample sizes and dose decisions with pending toxicity outcomes.
The likelihood approximation, enrolled-count safety rule and accrual suspension
are explicit. Precomputed effective-follow-up boundaries provide numerical lookup
without rounded cutoffs. Calendar-time replay and simulation support staggered
enrollment and outcome-driven pauses, with calibrated Weibull/log-logistic toxicity
timing scenarios. Flowcharts and integrated reports remain pending.


## TITE-BOIN

[TITE-BOIN](docs/tite-boin.md) adds vectorized pending-outcome imputation,
standardized follow-up thresholds, and interim dose decisions with the current
completion and minimum-follow-up suspension rules. Calendar replay and simulation
include releases at minimum-follow-up thresholds and calibrated toxicity timing.
Optional 3+3 modifications follow the app’s pending-outcome rules. Rolling 6
comparison and integrated reports remain pending.


## Rolling Six

[Rolling Six](docs/rolling-six.md) provides patient-by-patient dose decisions,
calendar replay and simulation, including the six-patient capacity, pending-outcome
escalation rule and downward completion. Results distinguish a found MTD from a
highest-dose recommendation. Dedicated comparison reports remain pending.


## BOP2 efficacy and toxicity monitoring

[BOP2 binary efficacy and toxicity](docs/bop2-binary.md) provides posterior stopping boundaries,
exact operating characteristics and power-maximizing grid calibration. Calibration
and informative-prior analysis are reported separately.
[Ordinal and multiple efficacy](docs/bop2-paired.md) add Dirichlet monitoring and
exact correlated operating characteristics with grid calibration.
[Joint efficacy/toxicity](docs/bop2-efftox.md) adds separate assessment schedules
and calibration against global and partial null hypotheses.
[Categorical sample-size optimization](docs/bop2-sample-size.md) searches for minimum
expected enrollment or minimum maximum sample size under error and power constraints.
[Survival monitoring](docs/bop2-survival.md) adds exponential/inverse-gamma
posterior decisions, calendar replay, simulation and Monte Carlo parameter
calibration with independent validation, plus expected-enrollment and minimax
sample-size searches. Two-arm/joint survival models and integrated reports remain pending.

[Beta Binomial Distribution Demo](docs/beta-binomial-demo.md) combines sequential
updating, credible sets, cohort simulation and prior/posterior history plots.

[Predictive Probabilities: binary outcomes](docs/predictive-binary.md) provides
two-arm interim predictions and first-stage planning tables for frequentist or
Bayesian final comparisons. [Time-to-event prediction](docs/predictive-survival.md)
adds posterior simulation with patient, time and event accrual limits.

[Phase II Predictive Probability](docs/phase2-predictive.md) adds strict
Lee–Liu stopping rules and exact cutoff/sample-size searches for power or
expected enrollment.

[Parameter Solver](docs/parameter-solver.md) provides all six distribution families,
with parameter, moment and two-quantile input modes and vectorized density/tail evaluation.

[Inequality Calculator](docs/inequality-calculator.md) compares independent variables
from all six families, including additive shifts, direct complementary probabilities
and numerical error estimates.

[Bayes Factor Binary](docs/bayes-factor-binary.md) adds nonlocal iMOM trial monitoring,
exact operating characteristics, simulation and text-input/HTML reporting, preserving
superiority, inferiority and inconclusive conclusions.

[BFMonitor](docs/bfmonitor.md) extends iMOM monitoring to variable prior shapes and
inclusive cutoffs. Its default online boundaries are reproduced; ESS calibration
and native protocol/export coverage remain pending.

[Bayes Factor TTE](docs/bayes-factor-survival.md) adds exponential/iMOM posterior
monitoring and continuous time-on-test boundaries. Calendar simulation and native
input/report workflows remain pending.

[PerfectMatch](docs/perfectmatch.md) adds quantile normalization and PDNN energy,
signal and conditional gene-expression calculations, plus joint fitting of stacking
energies, position weights, expression and background. Native CEL/file/QC/display
workflows remain pending.

[Toxicity Probability Intervals](docs/mtpi.md) adds mTPI decision tables, paper
safety rules, isotonic final selection and batched trial simulation. Original TPI
calibration and native software workflow audits remain pending.

[CI of Interaction Index and SYNERGY](docs/interaction-index.md) share median-effect
regression and Loewe interaction indices with log-delta confidence intervals for
observed combinations and fixed-ratio curves, plus the normal-coefficient Monte
Carlo comparator with retained draws. Other models and native workflows
remain pending.

[Decentralized trial planning](docs/dct-normal.md) adds continuous and binary
sample sizes with onsite/offsite heterogeneity, unequal arm variances and repeated
measurements, explicit allocation rounding and achieved power. Native rounding
and reports remain pending.

[Bayesian Chi Square TTE Fit](docs/bayesian-chi-square.md) adds the complete-data
posterior diagnostic, exact exponential-posterior workflow and dependent
order-statistic bounds. Censoring, other family fits, BIC/DIC, native rank/trim
conventions and reporting remain pending.

[MERIT](docs/merit.md) adds isotonic dose selection, Bayesian interim decisions,
correlated endpoint simulation and sample-size/boundary optimization for randomized
dose-optimization trials. Trial replay and simulation support separate interim
schedules and permanent arm stops. Native pooling conventions and reports remain pending.

[ESS Regression](docs/regression-ess.md) adds normal and logistic regression prior
effective sample sizes, including parameter subvectors, using direct expected
curvature calculations. Native R inputs and covariate defaults remain pending.

[TOP](docs/top-binary.md) adds delayed binary-response posterior decisions,
accrual suspension, effective-sample-size boundary tables, and batched calendar
replay/simulation, and tuning-parameter grid calibration with independent validation.
Multiple endpoints and native reports remain pending.

[Original TPI](docs/tpi.md) adds posterior-SD intervals, original-paper decision
tables, two-patient safety gating, isotonic MTD selection and batched simulation,
alongside the existing mTPI implementation.

[aPCoA](docs/apcoa.md) adds covariate-adjusted principal coordinates, signed
spectral diagnostics and grouped before/after plots, checked against the original
R implementation and independent regression calculations.

[CondiS](docs/condis.md) adds censored-lifetime imputation using conditional
restricted survival means, with native linear and KM-step interpolation.
Its default CondiS-X linear refinement is also available, with explicit censoring
diagnostics. Seven other refinement learners remain pending.

[1+2+3 rare-disease design](docs/rare-disease-123.md) adds cohort-based
efficacy/toxicity dose assignment, OBD selection and batched trial simulation with
correlated endpoints and patient-allocation summaries. The efficacy prior is explicit
because the public protocol omits it; default decision tables are reproduced.

[iBOIN](docs/iboin.md) adds historical-prior elicitation, dose-specific decision
boundaries, optional robust historical borrowing and complete-outcome dose assignment,
verified against published and
live-app tables. Simulation and final MTD estimation options remain pending.

[Bayesian prior ESS](docs/conjugate-ess.md) adds seven conjugate-model calculations,
with vectorized inputs and an explicit choice between information-based and native
gamma–exponential conventions. Nonconjugate app workflows remain pending.

[Survival prior ESS](docs/survival-ess.md) evaluates the native censored-exponential
information criterion analytically, avoiding Monte Carlo noise and patient loops.

[CID2BP](docs/cid2bp.md) adds all nine confidence-interval menu options for independent
binomial differences, including Cox–Snell profile likelihood and native boundary
adjustments and exact binomial-tail inversion. Session/report interfaces remain pending.

[CONFINT](docs/confint.md) adds CI-length assurance, population-SD limits, and
minimum integer sample sizes for normal means, normal SDs, and independent
pooled mean differences. Binomial width assurance, attainable lengths, event-
probability limits and discrete sample-size planning are also available. Poisson
rate-interval planning includes width probability, length/rate limits and earliest
qualifying exposure. Binomial-difference Wald-width planning includes full
probabilities, event-probability limits and balanced sample sizes. Survival
hazard/mean width assurance is available for fixed counts and Poisson accrual;
bracketed survival quantile/design inversions and automatic hazard-range
searches are also available. Native reporting remains pending.

[IPDfromKM](docs/ipdfromkm.md) reconstructs approximate patient survival records
from Kaplan–Meier coordinates, with native coordinate cleaning, optional reported
risk counts and total events. It returns fitted curves and reconstruction errors;
two-arm Efron Cox comparisons, survival confidence intervals, landmark summaries
and survival quantiles are also available. Digitizing and native graphics remain
pending.

[ASYPOW](docs/asypow.md) adds information-matrix power, sample-size and
significance calculations with independent-group binomial, Poisson and
exponential-survival information, including linear/quadratic regression designs
and complementary-log-log binomial models. Raw ordinal and cumulative-link
ordinal regression information, general logistic/multiplicative-binomial designs,
multinomial information and information reparameterization are also available.
SMO binomial, Poisson, multinomial, ordinal and censored exponential-survival
fixed-null/equality designs are available with both df conventions.
Mixed fixed/equality constraints are supported for binomial, Poisson and survival
groups. Partial fixed categorical nulls redistribute remaining probability using
expected-likelihood maximization. General categorical equality components use a
constrained likelihood fit. Generic SMO accepts a user-supplied expected log
likelihood, bounds and fixed/equality constraints. Logistic, complementary-log-log,
Poisson and censored
exponential-survival SMO regression supports polynomial and explicit design matrices.
Multiplicative-binomial log-linear SMO and logistic/cloglog ordinal regression
SMO are also available. Full native workflows remain pending.

[BERDS](docs/berds.md) adds regression variable selection through repeated data
splitting, with trimmed validation curves, automatic threshold selection and
final-model diagnostics. Stable SVD fits support corrected refitted scores and
an explicit original-software compatibility mode.

[WINDOWS](docs/windows.md) adds fixed-width and nearest-neighbor smoothing, local
polynomial derivatives, descriptive window statistics, and leave-one-out selection
of widths or neighbor counts, with documented native boundary weights and corrections.

[EXTSIG](docs/extsig.md) adds unconditional two-binomial tests with five outcome
orderings, mid-p calculations, and either the native probability grid or continuous
nuisance maximization with numerical bounds.

[DRDIST](docs/drdist.md) provides its seven distribution-calculator menu operations
through the shared numerical kernels, preserving the original tail conventions
while correcting legacy approximation and cancellation problems.

[TRAX](docs/trax.md) plots arbitrary transformations of either axis while retaining
original-unit labels, with grids, explicit limits, and overlays that reuse the
transform contract. A plotting-independent interface exposes transformed pairs
and any omitted row indices.

[SURVAN](docs/survan.md) adds multi-group log-rank and Gehan–Breslow tests,
including stratification, tied events, and group score/covariance diagnostics.
Its [Kaplan–Meier tables](docs/survan-km.md) include Simon–Lee survival and
quantile confidence calculations. [Logistic regression](docs/survan-logistic.md)
adds fitted probabilities and coefficient/likelihood inference.
[Cox regression](docs/survan-cox.md) adds multivariable Breslow-tie fits, with
[Kalbfleisch–Prentice baseline survival](docs/survan-baseline.md) and profile
prediction. [Frequency tables and descriptive summaries](docs/survan-descriptive.md)
complete its [seven advertised calculation families](docs/survan-coverage.md).

[BLiP](docs/blip.md) adds its standard grouped boxplots, histograms and frequency
polygons, with plotting-independent geometry and optional Matplotlib rendering.
Custom layouts include fixed/variable percentile boxes, percentile and mean/SD/SE
lines, centered/baseline placement, and all six point patterns.

[EVENTCHART](docs/eventchart.md) provides coded-event conversion and calendar or
elapsed-time subject timelines, with sorting, reference alignment, covariate
placement, interval overlays and immutable plotting geometry. Goldman entry-date
charts add current-date boundaries, optional native-boundary compatibility and
calendar date conversion/formatting. Categorical conversion supports string codes,
explicit factor order, unobserved categories and shared time columns. Grouped
styles, general calendar-covariate layouts, square plots and separate legend pages
complete the [advertised plotting families](docs/eventchart-coverage.md).

[SOGS](docs/sogs.md) implements chromosome-level genotype-selection breeding,
including marker error, all four selection rules, chromosome exclusions, variable
offspring schedules and replicate simulations. Reports align chromosome counts
and donor-length summaries to the same backcross generation.

[SORTF90](docs/sortf90.md) provides `sortf90(source)` to alphabetize Fortran
program units and nested contained procedures. It returns text for review and
matches the original separator and inter-unit comment handling.

[Misclib](docs/misclib.md) adds bounded scalar maximization with `fun_max` and
independent reverse-communication searches. Its 35 mathematical helpers and
shared root/string routines reuse the existing CDFLIB implementations. The
[coverage audit](docs/misclib-coverage.md) reconciles every archive file. Matrix-column sorting,
gather indices and direct/reversed permutations support numeric and string records.

Misclib’s `format_number` formats integer and floating values with alignment,
fixed/scientific thresholds, exponent scaling and explicit field-fit reporting.

`compile_misclib_messages` reads Misclib’s page-template syntax and renders
fixed-width substitutions; `print_misclib_message` uses the existing console’s
display and stream-routing controls.

`misclib_open_file` provides scripted or interactive file selection, with
read/create/append/overwrite actions and confirmation before changing a file.

[Proportional Density](docs/proportional-density.md) adds treatment-effect
estimation for censored survival using a density-ratio model, incidence estimates,
and disease-conditional survival curves. It includes the supplied goodness-of-fit
statistic, the paper’s failure-only goodness-of-fit bootstrap, and explicit
equal-censoring LR inference. Full-data disease-curve bootstrap and unequal-
censoring treatment-effect calibration remain pending.

[Response and Survival](docs/response-survival.md) combines early response
categories with censored survival for Bayesian adaptive randomization. It includes
conjugate posterior comparison and a two-arm trial simulator with stopping,
follow-up, allocation summaries and Monte Carlo errors.

[ANOVA DDP](docs/anovaddp.md) now supplies its nonlinear repeated-measurement
curve, Gaussian likelihood, subject-level parameter sweep and residual-variance
conditional, plus atom, cluster, covariance and concentration updates.
`fit_anovaddp` runs the complete fitting chain and returns immutable posterior
states. `predict_anovaddp` adds new-subject curves, native study components and
nadir summaries. Source-format readers, predictive text exports and all five
report figures complete the supplied workflow; see the documented MCMC mixing
limitations before scientific use.

[ACCFLF](docs/accflf.md) provides log-F probabilities and derivatives,
right-censored accelerated failure-time regression at fixed shape, and survival
prediction. It covers the source's fixed-shape Weibull, exponential, lognormal
and log-logistic submodels with their documented finite-df boundary convention.
Profile shape estimation, rectangular grids and all six named-model comparisons
are also available, with explicit local-search and boundary diagnostics.
Source-format table input, covariate changes, covariate-averaged survival and
fit/search/grid/model reports complete the supplied workflow.

[U2OET](docs/u2oet.md) adds PDS, conventional interaction and hybrid ordinal
efficacy/toxicity probabilities for two-agent combinations, with stable joint
log probabilities, likelihoods and expected utilities. Posterior-draw summaries
and new-cohort allocation include acceptability, patient-surplus randomization
and escalation restrictions. Multi-chain posterior fitting supports explicit
priors and complete outcomes. IID prior draws, beta-moment prior information
and pseudo-trial prior calibration are available. Additional coordinate and joint
link moves address diffuse-prior mixing. Gaussian-copula scenario construction
and native scenario/dose/utility readers are available. Patient snapshots,
toxicity-only likelihoods and open-cohort decisions are supported. Single-trial
calendar simulation includes pending outcomes and explicit final-selection
conventions. Multi-trial summaries report selection, enrollment, duration and
normalized utility performance with Monte Carlo errors. Native final-selection
parity and published operating-characteristic validation remain pending.

[CATBUB](docs/catbub.md) provides categorical-utility posterior comparisons,
Dirichlet Monte Carlo, a success-only binary comparator, sequential multinomial
simulation, group-sequential sample-size/boundary calibration and observed-data
analysis. It supports arbitrary categorical outcomes and utility vectors, with
explicit priors, alpha spending and first-stop operating characteristics.
Identical source-generated trials reproduce the R boundaries and operating
characteristics; source analysis/reporting corrections are documented.

[Parallel phase I/II](docs/parallel-phase12.md) now supports the archived four-arm
C workflow: phase-I escalation, beta-binomial adaptive randomization, toxicity
closure, efficacy/futility stopping, final selection and replayable simulation.
Python replay matches 24 native C decision histories, with independent R checks
of 179 posterior comparisons. The later six-dose C++ variant now has an integrated
calendar simulator combining the shared logistic response posterior, beta toxicity
updates, phase-I progression, blocked accrual and phase-II allocation/stopping.
Its component audits cover 100 native posterior-decision cases and 948 phase-I
transitions. Source eligibility quirks and final analysis with pending outcomes
are explicit; complete final follow-up is an optional extension. Native reporting
and published operating-characteristic replication remain outstanding.

[BlockARAND](docs/blockarand.md) now supports two-arm block adaptive randomization:
posterior allocation, rational block sizes, balanced burn-in, patient-wise stopping
and repeated-trial operating characteristics. The implementation preserves the
archive's separate final cutoff and exact burn-in transition. Original compiled
methods validate 5,914 block plans and 180 stopping cases; repeated simulations
share a bounded posterior cache.

[Adaptive Randomization](docs/arand.md) now has its binary and exponential
survival posterior core for up to ten arms: largest/smallest parameter
probabilities, mean/median survival parameterizations, threshold exceedance and
stable exponential tuning. Vectorized multi-arm integration agrees with 52
independent R calculations. Calendar conduct and native stopping/allocation
controls remain pending.

[PRT](docs/prt.md) now computes predictive toxicity risks from aligned posterior
draws and applies the published cohort-suspension, dose-movement and final-selection
rules. A bounded-memory count recursion replaces exponential enumeration of pending
outcomes while preserving posterior dependence. Probit model components are also
available, including the state-space posterior fit and full-covariance isotonic
formula. The guide-history pilot exposes out-of-range projected risks, reported
explicitly; native projection safeguards and calendar simulation remain pending.

[ROSE](docs/rose.md) provides one- and two-stage dose-selection designs using
normal approximation or exact binomial constraints, including unequal allocation
and O'Brien–Fleming interim spending. Exact operating characteristics, strict
response-count decisions and seeded binary-response simulations with Monte Carlo
errors are available. Published design sizes and independent outcome enumeration
validate the port; the supplement's free interim-boundary optimization is outside
this API's scope.

[Bayesian success calibration](docs/success-calibration.md) now evaluates distinct
design and analysis priors for single-arm binary, arbitrary-margin two-arm binary,
and one-/two-arm normal models, including the paper's log-hazard-ratio
approximation. It reports joint decision/truth probabilities, Bayesian power and
error metrics, and searches candidate cutoffs for a target probability of
incorrect decision. Reusable binary probability tables avoid repeating quadrature
when evaluating or calibrating many cutoffs.
