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


`cdf_beta`, `cum_beta`, `ccum_beta` and `inv_beta` begin the CDFLIB90 port with
beta tails, quantiles and shape inversions. They broadcast arrays and retain
small probability/coordinate complements. [CDFLIB90 notes](docs/cdflib90.md)
describe native validation, corrected source defects and the remaining scope.


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
files and tracks the remaining distribution, legacy-library and public-support scope.


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
are implemented; CDFLIB90's additional legacy and support interfaces remain open.

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
All twelve legacy distribution families now have independent C/F77 validation;
CDFLIB90 remains partial while its public support interfaces are being ported.

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
