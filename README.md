# mdanderson-stats

Work in progress: one Python package for the methods in the [MD Anderson
biostatistics software catalog](https://biostatistics.mdanderson.org/SoftwareDownload).
The scope includes all desktop and online entries. A catalog entry is not an
implementation; `catalog.json` explicitly tracks pending work and validation.

The [software coverage index](docs/software-status.md) lists all 138 entries:
85 implemented, 45 partial, and 8 pending. Each method's guide explains its
supported scope, validation and remaining limitations. Implemented means a
usable, validated Python workflow for the recovered software specification;
it does not mean identical native interfaces, file bytes or random streams.
Partial entries retain unfinished calculations, input/output workflows or
source-contract uncertainties.
The [remaining-work review](research/statistical-coverage-priority-2026-10-04.md)
separates unresolved statistical specifications from input, report and
application-compatibility work across the partial entries. Keyboard now includes
its saved protocol workflow. BOIN and Keyboard add source-defined overdose-allocation
risks, while BOIN and TITE-BOIN add reproducible scenario reports. TITE-BOIN's
remaining uncertainty about native evaluation labels is recorded explicitly.

Validated community checkpoints are published on
[`master`](https://github.com/dx-li/mdanderson-stats/tree/master); `main`
mirrors those checkpoints. The guides and coverage index describe which
parts of each program are ready to use.

With Python 3.12 or newer, obtain and install the stable source:

```sh
git clone --branch master https://github.com/dx-li/mdanderson-stats.git
cd mdanderson-stats
python -m pip install .
```

Use `python -m pip install '.[plot]'` for optional figures, or
`python -m pip install '.[image]'` for Pinnacle TIFF input. For reproducible
analyses, record the source revision with `git rev-parse HEAD` and retain
the method settings and random seeds. Development setup is described below.

Original contributions use the [MIT License](LICENSE.md). Adapted material
retains its [upstream terms](THIRD_PARTY_NOTICES.md), including commercial-use
conditions for some legacy routines; the complete distribution is mixed-license.

The implementation uses NumPy broadcasting and compiled SciPy numerical kernels.
Numba will be considered for measured simulation bottlenecks. This is an independent
project and is not an MD Anderson release.

[MDS-HOPE](docs/mds-hope.md) implements the recovered published Cox score,
predictor contributions, reference-profile hazard ratios and the published
five- and six-group cutoffs.
The cytogenetic score must be explicitly encoded; risk grouping requires an
already-standardized score or caller-supplied reference constants. The source
does not supply the original calibration or baseline survival, so native app
equivalence and absolute survival prediction remain open.

[WFMM functional mixed models](docs/wfmm.md) now support orthogonal wavelets and
[supplied custom transform pairs](docs/wfmm-custom-transform.md),
Bayesian fixed/random-effect fitting with coefficient-specific
variances, and reconstructed posterior curves with contrasts and simultaneous
bands, variance functions and covariance reconstruction. Empirical-Bayes
shrinkage calibration is available conditional on supplied variance estimates.
[Posterior prediction](docs/wfmm-prediction.md) covers future latent curves and
replicates, with explicit existing/new random-effect designs and shared-level
dependence propagated through posterior variance draws.
Explicit coefficient or wavelet-band selection preserves original positions
for reconstruction after fitting a reduced model.
A bounded REML initializer estimates starting random-effect and residual
variances under an explicit Python policy. Variance priors and proposal settings
remain explicit; native initialization and additional native workflows remain open.

## Development

```
uv sync --locked --group dev --extra plot
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv build
uv run --extra plot pytest
```

GitHub validation runs on pushes to `main`, pull requests targeting `main` or
`master`, and manual dispatches. Pushes outside `main` do not trigger validation
by themselves; updates to open pull requests targeting `main` or `master` still
run validation. Stable community checkpoints are collected on `master` after
numerical and package validation; [publication records](research/publication-checkpoint-2026-09-28.md)
distinguish local checkpoints from confirmed GitHub publication.
Formatting, lint, types and packaging must pass before the Python 3.12–3.14
test matrix starts. At most two matrix jobs run at once, with a 25-minute limit
per job; a newer run cancels a superseded run for the same branch or pull request.

The old `.github/workflows/ci.yml` workflow is disabled in GitHub because retained
historical branches still contain its unrestricted push trigger. Keep it disabled;
the active workflow is `.github/workflows/validation.yml`.

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
boundaries and original R results validate the core. The validated Python workflow
is complete; native animation and file formats remain compatibility limits.
A [saved HTML report](docs/boin-protocol-report.md)
combines English or Chinese methods text, the numerical decision table and newly
computed scenario summaries, including source-defined overdose-allocation risks. Custom rate cutoffs
can be entered directly, with numerically checked inversion to BOIN alternatives.
The conventional 3+3 comparator supports cohort expansion and matching BOIN
enrollment caps to realized 3+3 sample sizes.
The [desktop coverage map](docs/boin-desktop.md) connects its single-agent,
delayed-toxicity and combination capabilities to the corresponding Python APIs.

## Keyboard dose finding

[Keyboard](docs/keyboard.md) adds posterior interval decisions, overdose safeguards,
isotonic MTD selection and batched simulation. The paper's complete-key convention
and the R package's adjusted endpoint convention are explicit. Native R comparisons
and independent exact interval probabilities validate the statistical core;
the [integrated HTML report](docs/keyboard-protocol-report.md) now connects
captured design settings, integer boundaries, scenario simulations and
overdose-allocation risks in a saved community workflow.

## TITE-Keyboard interim decisions

[TITE-Keyboard](docs/tite-keyboard.md) adds uniform and informative follow-up
weights, effective sample sizes and dose decisions with pending toxicity outcomes.
The likelihood approximation, enrolled-count safety rule and accrual suspension
are explicit. Precomputed effective-follow-up boundaries provide numerical lookup
without rounded cutoffs. Calendar-time replay and simulation support staggered
enrollment and outcome-driven pauses, with calibrated Weibull/log-logistic toxicity
timing scenarios. [Adaptive timing weights](docs/tite-keyboard-adaptive.md)
use shared timing inference with observed-event and pending-survival information,
explicit priors, and sampling diagnostics.
[Adaptive calendar simulation](docs/tite-keyboard-adaptive-calendar.md) fits
these weights at each interim using only observed information, with separate
outcome and sampler random streams and cumulative work limits.
[Saved protocol reports](docs/tite-keyboard-protocol-report.md) capture settings,
decision flow, numerical boundaries and scenario summaries with Monte Carlo errors.
An explicit true-MTD dose also enables the paper's trial-level risks of allocating
fewer than six patients to the MTD and more than half of enrolled patients above it.


## TITE-BOIN

[TITE-BOIN](docs/tite-boin.md) adds vectorized pending-outcome imputation,
standardized follow-up thresholds, and interim dose decisions with the current
completion and minimum-follow-up suspension rules. Calendar replay and simulation
include releases at minimum-follow-up thresholds and calibrated toxicity timing.
Optional 3+3 modifications follow the app’s pending-outcome rules. A
[Rolling Six comparison](docs/boin-time-comparison.md) supplies shared scenario
inputs, numerical summaries and a Markdown report. The new
[protocol report](docs/tite-boin-protocol-report.md) records effective settings,
replayable scenario seeds, the decision sequence and saved HTML results.
Native correct-MTD/regret evaluation conventions remain unresolved; the report
preserves selection probabilities and mean per-dose allocations without inventing those labels.


## Rolling Six

[Rolling Six](docs/rolling-six.md) provides patient-by-patient dose decisions,
calendar replay and simulation, including the six-patient capacity, pending-outcome
escalation rule and downward completion. Results distinguish a found MTD from a
highest-dose recommendation. The TITE-BOIN comparison preserves these statuses
alongside selection probabilities, allocation, toxicity and duration summaries.


## BOP2 efficacy and toxicity monitoring

The [retired BOP2 desktop entry](docs/bop2-desktop.md) directs users to the
online method family implemented below.

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
sample-size searches. [Saved protocol reports](docs/bop2-protocol-reports.md)
cover all six advertised endpoint families with captured settings and exact or
simulated operating characteristics. Native optimizer equivalence remains open;
two-arm/joint survival is outside the advertised single-arm endpoint scope.

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
monitoring and continuous time-on-test boundaries. Its
[calendar extension](docs/bayes-factor-survival-calendar.md) adds explicit event
and censoring tapes plus bounded serial simulation with replayable seeds and
early/final operating characteristics. [Saved scenario reports](docs/bayes-factor-survival-report.md)
distinguish terminal stopping from later follow-up and include optional continuous
boundaries. Native timing, integer-day boundaries and input parity remain open.

[PerfectMatch](docs/perfectmatch.md) adds quantile normalization and PDNN energy,
signal and conditional gene-expression calculations, plus joint fitting of stacking
energies, position weights, expression and background. A grouped log-intensity
correlation summary compares observed and fitted probes within each probeset.
The paper's array-wide mean-500 expression scaling uses stable log arithmetic
and preserves probeset identifiers.
Native CEL/file workflows, other QC outputs and displays remain pending.

[Toxicity Probability Intervals](docs/mtpi.md) adds mTPI decision tables, paper
safety rules, isotonic final selection and batched trial simulation. Original TPI
calibration and native software workflow audits remain pending.
[Posterior isotonic intervals](docs/mtpi-isotonic-posterior.md) add the paper's
draw-then-transform uncertainty estimates, with bounded sampling and optional
joint draws across the supplied dose grid.
[Prior sensitivity](docs/mtpi-prior-sensitivity.md) applies the paper's common
beta-prior choices consistently to decisions, safety, selection and simulation
while retaining its fixed loss calibration.

[CI of Interaction Index and SYNERGY](docs/interaction-index.md) share median-effect
regression and Loewe interaction indices with log-delta confidence intervals for
observed combinations and fixed-ratio curves, plus the normal-coefficient Monte
Carlo comparator with retained draws. The
[pooled-error fallback](docs/interaction-index-pooled-error.md) estimates
observed-combination uncertainty when replicate measurements are unavailable,
using an explicit residual-df pooling convention on the logit-effect scale.
[Optional figures](docs/interaction-index-plots.md) display the fitted
median-effect lines and pointwise interaction intervals. The
[published case studies](docs/interaction-index-case-studies.md) reproduce both
paper datasets with saved figures and numerical results.
The [three-drug simulation study](docs/interaction-index-study.md) adds
bounded coverage and interval-length comparisons with captured settings.
SYNERGY also provides a
[semiparametric response surface](docs/synergy-surface.md) with raw/log-dose
baselines and REML thin-plate smoothing. Its
[wild-bootstrap workflow](docs/synergy-surface-bootstrap.md) generates Mammen
resamples and refits each marginal baseline and spline, returning departure
draws and descriptive sample standard deviations. Other parametric surfaces,
the original bootstrap interval convention and native workflows remain pending.

[Decentralized trial planning](docs/dct-normal.md) adds continuous and binary
sample sizes with onsite/offsite heterogeneity, unequal arm variances and repeated
measurements, explicit allocation rounding and achieved power. Native rounding
and reports remain pending.

[Bayesian Chi Square TTE Fit](docs/bayesian-chi-square.md) covers posterior
fitting for all seven distributions in the BCSTTE guide.
Exponential and fixed-shape Weibull models have exact Gamma-prior posteriors;
unknown-shape Weibull, Gamma, inverse-Gamma, log-logistic and log-odds-rate
models use explicit Gaussian priors on transformed parameters. The complete-data
[lognormal workflow](docs/lognormal-bayesian-gof.md) uses a proper
Normal-Inverse-Gamma prior. Its
[right-censored workflow](docs/lognormal-right-censored-bayesian.md) uses that
same explicit prior with a Gibbs sampler; censor integration does not leave a
conjugate posterior. All seven families accept noninformative right censoring.
The [rounded-time workflow](docs/rounded-tte-bayesian-gof.md) fits all seven
families using interval probabilities and explicit transformed-Gaussian priors,
with a paired randomized Johnson diagnostic. The generic diagnostic also accepts
discrete/rounded CDF bounds from an appropriately fitted posterior. The dedicated
right-censored lognormal result has no such diagnostic. Native prior defaults,
censored diagnostics, rank/trim conventions and reporting remain pending.

[MERIT](docs/merit.md) adds isotonic dose selection, Bayesian interim decisions,
correlated endpoint simulation and sample-size/boundary optimization for randomized
dose-optimization trials. Trial replay and simulation support separate interim
schedules and permanent arm stops. [Interim calibration](docs/merit-interim-search.md)
searches final boundaries and maximum sample size under that stopping policy,
with corner-specific power, error and enrollment summaries. Native pooling
conventions and reports remain pending.

[ESS Regression](docs/regression-ess.md) adds normal and logistic regression prior
effective sample sizes, including parameter subvectors, using direct expected
curvature calculations or the original uniform-covariate simulation workflow.
Replicate-averaged information paths, interpolated crossings and explicit
not-reached outcomes are checked against the original R calculator.

[TOP](docs/top-binary.md) adds delayed binary-response posterior decisions,
accrual suspension, effective-sample-size boundary tables, and batched calendar
replay/simulation, nonuniform analysis timing, and tuning-parameter grid calibration
with independent validation.
[Two-endpoint TOP](docs/top-endpoints.md) supports co-primary efficacy and
efficacy/toxicity monitoring, endpoint-specific pending outcomes and nonuniform
timing weights. Its [calendar replay and simulation](docs/top-endpoints-simulation.md)
preserve joint endpoint outcomes and report observed decisions, enrollment and
duration. [Finite-grid calibration](docs/top-endpoints-calibration.md) evaluates
explicit joint null scenarios and one alternative with independent holdout
validation. [Saved community reports](docs/top-community-report.md) capture all
three endpoint modes, actual design settings, named scenarios, boundaries,
seeds and simulation uncertainty. Native optimizer/template parity and guarantees
over the entire composite null are not claimed.

[Original TPI](docs/tpi.md) adds posterior-SD intervals, original-paper decision
tables, two-patient safety gating, isotonic MTD selection and batched simulation,
alongside the existing mTPI implementation. [Dose-specific Beta priors](docs/tpi-informative-priors.md)
provide an explicit conjugate extension throughout posterior summaries, conduct,
selection and simulation, using compact lookup tables for distinct priors.
[Posterior intervals](docs/tpi-isotonic-posterior.md) reuse the mTPI paper's
beta-draw/isotonic inference procedure with TPI priors. Transformation weights,
random draws and quantile conventions are explicit; native TPI tuning remains open.

[aPCoA](docs/apcoa.md) adds covariate-adjusted principal coordinates, signed
spectral diagnostics and grouped before/after plots, checked against the original
R implementation and independent regression calculations. Optional group
ellipses and medoid connectors follow the source's covariance and matrix-row
profile conventions; their geometry is also available without plotting.

[CondiS](docs/condis.md) adds censored-lifetime imputation using conditional
restricted survival means, with native linear and KM-step interpolation.
All eight CondiS-X learners are available: linear, ridge, lasso, nearest-neighbor,
neural, radial SVM, random forest and Gaussian gradient boosting. They include
learner-specific tuning, full-sample refits and explicit
censoring diagnostics. The [survival comparison](docs/condis-survival-comparison.md)
adds the vignette's two curves, censor marks and an explicit risk table.
Neural fits expose iteration-limit diagnostics; their nonconvex fitting paths
can differ from R even with identical starting weights.
SVM fits expose scaling transformations and numerical optimality diagnostics.
Boosting reuses tree prefixes during tuning and requires at least 43 training
rows in each fold. Forest prediction averages bootstrap trees and retains
optional in-bag/per-tree diagnostics; tied native paths can diverge.
The app's remaining input/report and separate prediction workflows stay partial.

[1+2+3 rare-disease design](docs/rare-disease-123.md) adds cohort-based
efficacy/toxicity dose assignment, OBD selection and batched trial simulation with
correlated endpoints and patient-allocation summaries. The efficacy prior is explicit
because the public protocol omits it; default decision tables are reproduced.

[iBOIN](docs/iboin.md) adds historical-prior elicitation, dose-specific decision
boundaries, optional robust historical borrowing and complete-outcome dose assignment,
verified against published and live-app tables. Patient-level replay supports
accelerated titration, cohort top-up, dose exclusions and safety/precision stops
within an explicit enrollment budget. [Final selection and serial simulation](docs/iboin-final-simulation.md)
add optional historical borrowing, explicit isotonic weights and candidate policies,
replayable trial seeds, and selection/allocation summaries with Monte Carlo uncertainty.
Native final-selection defaults and report workflows remain open.

[Bayesian prior ESS](docs/conjugate-ess.md) adds seven conjugate-model calculations,
with vectorized inputs and an explicit choice between information-based and native
gamma–exponential conventions. [CRM prior ESS](docs/crm-prior-ess.md) adds adaptive
empiric-model trial simulation and expected subset-information paths, with explicit
full versus native posterior moments and continuous versus coarse-grid ESS.
[TITE-CRM prior ESS](docs/tite-crm-prior-ess.md) adds delayed-toxicity trial
histories, observed follow-up assessment and an explicit legacy arrival-time
criterion, preserving signed information. Unknown-mean variance ESS remains pending.

[Survival prior ESS](docs/survival-ess.md) evaluates the native censored-exponential
information criterion analytically, avoiding Monte Carlo noise and patient loops.

[CID2BP](docs/cid2bp.md) adds all nine confidence-interval menu options for independent
binomial differences, including Cox–Snell profile likelihood and native boundary
adjustments and exact binomial-tail inversion. [Repeated comparison sessions](docs/cid2bp-session.md)
support both count-entry modes, per-case settings and cumulative reports.

[CONFINT](docs/confint.md) adds CI-length assurance, population-SD limits, and
minimum integer sample sizes for normal means, normal SDs, and independent
pooled mean differences. Binomial width assurance, attainable lengths, event-
probability limits and discrete sample-size planning are also available. Poisson
rate-interval planning includes width probability, length/rate limits and earliest
qualifying exposure. Binomial-difference Wald-width planning includes full
probabilities, event-probability limits and balanced sample sizes. Survival
hazard/mean width assurance is available for fixed counts and Poisson accrual;
bracketed survival quantile/design inversions and automatic hazard-range
searches are also available. An immutable [calculation log](docs/confint-session.md)
records repeated analyses, complete settings and diagnostics in a saved report.

[IPDfromKM](docs/ipdfromkm.md) reconstructs approximate patient survival records
from Kaplan–Meier coordinates, with native coordinate cleaning, optional reported
risk counts and total events. It returns fitted curves and reconstruction errors;
two-arm Efron Cox comparisons, survival confidence intervals, landmark summaries
and survival quantiles are also available. The separate
[reconstruction report](docs/ipdfromkm-diagnostics.md) adds the native rounded
precision summaries and KS discrepancy, with its legacy nominal p-value
explicitly distinguished from calibrated inference. The
[image workflow](docs/ipdfromkm-digitization.md) adds manual axis/point selection,
stable coordinate calibration, image previews and reconstruction/risk plots.
Original patient records cannot be recovered exactly from a published curve.

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
equal-censoring LR inference. The [full-data disease-curve bootstrap](docs/proportional-density-full-bootstrap.md)
resamples both events and censoring records, refits both curves and retains
failed-replicate calibration bounds. Unequal-censoring treatment-effect null
calibration and parameter uncertainty remain pending.

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
priors and complete outcomes. An [adaptive precision controller](docs/u2oet-adaptive-precision.md)
continues PDS/CMI/hybrid chains toward per-chain corner-utility MCSE targets,
with a draw cap and separate convergence diagnostics.
IID prior draws, beta-moment prior information
and pseudo-trial prior calibration are available. Additional coordinate and joint
link moves address diffuse-prior mixing. Gaussian-copula scenario construction
and native scenario/dose/utility readers are available. Patient snapshots,
toxicity-only likelihoods and open-cohort decisions are supported. Single-trial
calendar simulation includes pending outcomes and explicit final-selection
conventions. Multi-trial summaries report selection, enrollment, duration and
normalized utility performance with Monte Carlo errors. Native final-selection
parity and published operating-characteristic validation remain pending.
[Adaptive trial sampling](docs/u2oet-adaptive-trial.md) applies the corner-utility
precision target to interim and final PDS/CMI/hybrid fits, with bounded work,
recorded diagnostics and explicit failure if the draw cap is insufficient.
[2017 GAO comparison probabilities](docs/u2oet-gao.md) additionally support explicit
raw-dose coefficients, a shared interaction and Gaussian-copula likelihoods;
[GAO posterior fitting](docs/u2oet-gao-fit.md) adds explicit normal-prior
coordinates, complete/partial outcomes and retained chain diagnostics.
[Adaptive GAO precision](docs/u2oet-gao-adaptive-precision.md) continues those
chains toward a supplied four-corner utility MCSE/SD target and reports
whether the target was reached within bounded draws and work.
[GAO calendar trials](docs/u2oet-gao-trials.md) connect that fitter to pending
outcomes, cohort allocation and final selection, with cumulative work limits
and replay inputs. Optional adaptive precision applies the supplied corner
target to interim and final analyses, reuses unchanged observed-data fits,
and fails explicitly if the target or cumulative work limit cannot be met.
Native prior interpretation and GAO calibration remain open.
The separate [original 2010 GAO model](docs/u2oet-gao2010.md) now supports
centered doses, endpoint-specific signed interactions, ordinal probabilities
and complete/toxicity-only likelihoods. Its separate
[posterior fitter](docs/u2oet-gao2010-fit.md) uses caller-supplied Gaussian
coordinates jointly restricted to the valid dose-grid domain and a uniform
association prior. Elicited prior calibration and native mapping remain open;
its parameters are distinct from the 2017 comparison model.
[Original-model cohort trials](docs/u2oet-gao2010-trials.md) apply posterior
utility selection, the paper's global all-dose toxicity stopping rule and
upper-neighbor escalation limits. Explicit truth and evaluability inputs
support complete and toxicity-only observations with replayable outcomes.

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
[Serial operating-characteristic summaries](docs/parallel-phase12-oc.md) add
selection, stopping, enrollment and pooled outcome rates with trial-level
Monte Carlo errors and reproducible per-trial seeds.
[Named scenario reports](docs/parallel-phase12-scenario-report.md) retain
those inputs and support the native four-arm probability-file order. Six-dose
calendar results also provide source-indexed duration summaries with an explicit
population-variance label for the native output misnamed standard deviation.
Phase-I summaries now include the source-defined 3+3 enrollment, toxicity and
admissibility tallies, with explicit handling of interrupted trials.
Python replay matches 24 native C decision histories, with independent R checks
of 179 posterior comparisons. The later six-dose C++ variant now has an integrated
calendar simulator combining the shared logistic response posterior, beta toxicity
updates, phase-I progression, blocked accrual and phase-II allocation/stopping.
An optional [importance sampler](docs/parallel-phase12-importance.md) adds
the source's mixture weighting and vector stopping rule, with bounded work,
posterior uncertainty and direct use in the six-dose decision functions.
The [probability summary](docs/parallel-phase12-probability-summary.md) streams
an explicit sequence of importance fits into the source's 60 component means
and sample variances. Calendar and scenario simulations capture these summaries
automatically, including reused fits with separate refit counts. Available final
fits also provide posterior-mode means and Laplace mixture variances, with
explicit no-fit and nonconvergence counts.
The [calendar simulator](docs/parallel-phase12-importance-calendar.md) can use
this backend at interim and final analyses, with aggregate work limits and
per-analysis uncertainty diagnostics; the existing MCMC backend remains the default.
The [six-dose simulation summaries](docs/phase12-calendar-oc.md) add serial
multi-trial selection, stopping, enrollment and endpoint rates, with replay seeds
and trial-level Monte Carlo errors. Generated outcomes and outcomes observed at
stopping have separate totals and denominators; convergence diagnostics remain visible.
Its component audits cover 100 native posterior-decision cases and 948 phase-I
transitions. Source eligibility quirks and final analysis with pending outcomes
are explicit; complete final follow-up is an optional extension. The four-arm
and six-dose Python workflows are covered. Exact native random streams and
report formats are not reproduced; full published operating-characteristic
replication remains a validation limitation.

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
independent R calculations. [Calendar replay](docs/arand-calendar.md) now adds
delayed-outcome handling, adaptive allocation, reversible suspension, permanent
futility, enrollment/duration limits and final selection. Explicit controller
policies document choices the native guide leaves unspecified.
[Serial simulation](docs/arand-simulation.md) adds Poisson accrual, binary or
exponential scenarios and per-arm operating characteristics with reproducible
trial seeds and bounded storage. Native-engine and report parity remain open.

[PRT](docs/prt.md) now computes predictive toxicity risks from aligned posterior
draws and applies the published cohort-suspension, dose-movement and final-selection
rules. A bounded-memory count recursion replaces exponential enumeration of pending
outcomes while preserving posterior dependence. Probit model components are also
available, including the state-space posterior fit and full-covariance isotonic
formula. [Calendar replay](docs/prt-calendar.md) adds explicit arrivals, interval
follow-up, cohort enrollment, suspension and final selection. The guide-history
pilot exposes out-of-range projected risks, reported explicitly; native projection
safeguards and native timing/operating-characteristic replication remain pending.

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
error metrics, and calibrates cutoffs for a target probability of incorrect
decision. Automatic searches enumerate every single-arm binary decision state
or bracket a normal/survival cutoff to a specified tolerance. Supplied-grid
calibration remains available, and reusable two-arm binary probability tables
avoid repeating quadrature when evaluating many cutoffs.
An [automatic two-arm binary search](docs/success-two-arm-automatic.md) now
scans conservative error-separated decision states while respecting
posterior integration uncertainty and retaining nonmonotone PID behavior.

[KeyboardComb](docs/keyboard-combination.md) now supports two-drug dose decisions,
posterior safety monitoring, weighted two-dimensional isotonic MTD selection and
seeded cohort simulation with Monte Carlo errors. Independent R references cover
movement, selection and the numerical fit; source discrepancies are documented.
The published key2/key3/key4 movement options add diagonal candidates and
posterior-proportional randomization with auditable candidate probabilities.
The default native key1 behavior is preserved. Generated protocols and
source-ambiguous key5/scenario generation remain open.

[BOINComb](docs/boin-combination.md) adds ordinary combination dose decisions,
posterior safety monitoring, final MTD and contour selection, and waterfall
subtrial planning. [Full waterfall replay and serial simulation](docs/boin-waterfall.md)
cover staircase, row and special same-row searches, with auditable observations
and contour diagnostics. Seeded simulations report operating characteristics
and Monte Carlo errors. Native R references distinguish the interactive
selector from the simulator's unrounded selection rule and expose native
dropped-observation defects. [Accelerated combination titration](docs/boin-combination-titration.md)
adds the CRAN single-patient staircase and first-cohort transition. App-specific
titration options, waterfall titration, the 3+3 run-in and generated protocols remain open.

[BOIN12](docs/boin12.md) now includes toxicity/efficacy posterior calculations,
utility desirability tables, single-stage dose decisions, final OBD selection
and joint-outcome cohort simulation. Independent R calculations validate the
posterior, ranks and selection examples. Exact risk-benefit tradeoff mapping
also feeds the existing decision and simulation APIs. [Two-stage conduct](docs/boin12-two-stage.md)
adds toxicity-only escalation followed by joint-endpoint optimization, with
explicit threshold timing and retained trial outcomes. Unresolved 3+3 run-in
precedence and generated reports remain open; native
support for nonadditive RDS enumeration is unverified.

[BF-BOIN](docs/bf-boin.md) adds backfill eligibility, pooled dose decisions,
posterior safety exclusions and final MTD selection. Assigned and evaluated
patient counts are kept separate. Calendar simulation includes delayed DLT and
response observation, auditable patient histories and Monte Carlo errors.
[Optional post-escalation expansion](docs/bard-expansion.md) holds enrollment
one dose below the last escalation cohort, with assigned-count caps and
toxicity closure. [Accelerated titration](docs/bf-boin-titration.md) adds
single-patient escalation, DLT/grade-2 triggers and dose-cap transitions, with
explicit grade-2 probabilities and assessment timing. The guide's optional
1/3 stay action (targets 0.20–0.279), 2/6 de-escalation (targets 0.28–0.33),
strict BF extra-safety count and strict final-MTD bound are supported.
[Saved protocol reports](docs/bf-boin-protocol-report.md) capture
design settings, timing and compact scenario summaries. Unspecified native
modifier interactions and report aggregation formulas remain explicit.

[BOP2-DC](docs/bop2-dc.md) adds binary efficacy monitoring with distinct
go/consider/no-go outcomes and exact operating characteristics. Independent
R posteriors and a closed-form early-stopping example validate the core.
Finite-grid calibration maximizes correct-go probability or minimizes expected
sample size while controlling false decisions. [Paired-endpoint monitoring](docs/bop2-dc-paired.md)
supports multiple efficacy and efficacy/toxicity decisions with joint Dirichlet
priors. [Time-to-event monitoring](docs/bop2-dc-survival.md) uses an exponential/
inverse-gamma model with separate median survival criteria. Paired modes include
exact operating characteristics that preserve endpoint association and early
stopping. Survival adds calendar replay, bounded exponential-trial simulation,
replay seeds and Monte Carlo errors, with independent calendar and analytic
operating-characteristic references. [Survival calibration](docs/bop2-dc-survival-calibration.md)
selects from explicit parameter grids with independent holdout results and
visible validation failures. [Paired calibration](docs/bop2-dc-paired-calibration.md)
uses exact correlated-outcome probabilities over an explicit control grid.
The [continuous Normal endpoint](docs/bop2-dc-normal.md) adds conjugate posterior
monitoring, trial replay and simulation, with centered calculations preserving
precision under large measurement offsets. [Normal calibration](docs/bop2-dc-normal-calibration.md)
adds common-path grid selection and independent holdout evidence.
[Randomized binary comparisons](docs/bop2-dc-randomized-binary.md) add independent
arm posteriors, optional early graduation, fixed-allocation replay and exact
operating characteristics. [Randomized Normal comparisons](docs/bop2-dc-randomized-normal.md)
add independent Student-t posterior differences, complete-outcome replay and
bounded simulation. [Randomized survival comparisons](docs/bop2-dc-randomized-survival.md)
add median-time differences, as-of censoring and fixed/Poisson accrual simulation.
Both support graduation and retain numerical-error safeguards. Randomized binary
calibration uses exact conditional operating characteristics; [Normal and survival
calibration](docs/bop2-dc-randomized-calibration.md) reuse common paths and report
independent holdout feasibility without reselection.
[Randomized paired outcomes](docs/bop2-dc-randomized-paired.md) add joint
Dirichlet arm models for multiple efficacy or efficacy/toxicity, combined
monitoring and absorbing replay, exact conditional operating characteristics
and finite-grid calibration. Bounded serial simulation supports larger designs
while retaining endpoint association and per-trial replay seeds.
[General categorical designs](docs/bop2-dc-categorical.md) extend the joint model
to more than two binary indicators. [Saved community reports](docs/bop2-dc-community-report.md)
cover all supported endpoint and arm configurations, with actual inputs, seeds,
decision probabilities and Monte Carlo errors.

[General categorical endpoints](docs/bop2-dc-categorical.md) extend these
workflows to more than two decision endpoints. Explicit binary indicators over
joint categories support single-arm and randomized designs, mixed efficacy and
toxicity directions, monitoring, absorbing replay and serial simulation.
Finite candidate calibration reports false-decision rates, enrollment and
Monte Carlo uncertainty under both correct-go and futile-enrollment objectives.

[BARPO](docs/barpo.md) adds binary posterior monitoring with or without a control
and four adaptive allocation methods: BARCP, BARN2N, BARMTV and explicit-target
DBCD. It accounts for pending assignment counts and allocation floors.
[Trial conduct and simulation](docs/barpo-trials.md) add balanced burn-in,
cohort allocation, scheduled arm/trial stopping, replayable assignments and
operating characteristics with cumulative efficacy and false-declaration rates.
Native DBCD target construction and reports remain open.

[PLBARPO control monitoring](docs/plbarpo-control.md) adds entire-trial and
concurrent control comparisons. Explicit enrollment windows and observation
cutoffs exclude pending and future outcomes. [Active-arm allocation](docs/plbarpo-allocation.md)
recomputes posterior competition after arms close, preserving total enrollment
for BARN2N and supporting all four randomization methods. [No-control platform
trials](docs/plbarpo-trials.md) add queued replacement, entrant burn-in, replayable
patient assignments, global monitoring and final assessment at arm caps.
Compact simulation summaries add per-arm operating characteristics, enrollment,
Monte Carlo errors and error rates against explicitly supplied null arms.
[Persistent-control trials](docs/plbarpo-control-trials.md) add entire-trial or
concurrent-control assessment, simultaneous replacement and recorded comparison
windows. Their aggregate simulations include conditional-rate errors and
explicit-null error rates, with the control excluded from efficacy hypotheses.
Delayed outcomes remain open.

[TTEConduct](docs/tteconduct.md) adds single-arm exponential survival monitoring
against an uncertain historical standard, with an additive improvement margin
and continuous total-time-on-test stopping boundaries. Independent R integration
checks the published guide example. [Saved HTML reports](docs/tteconduct-report.md)
echo the design and boundaries in explicit caller-selected time units.

[One Arm Time to Event Simulator](docs/one-arm-tte.md) adds mean/median survival
priors, separate inferiority and superiority rules, and calendar simulation.
Periodic and pre-accrual monitoring, minimum enrollment, and final follow-up
follow the extracted native help. Simulations retain compact per-trial summaries
with Monte Carlo errors and central sample quantiles.
[Multi-scenario reports](docs/one-arm-tte-report.md) capture effective inputs,
independent seeds and compact results in saved HTML.

[rBOP2 binary designs](docs/rbop2-binary.md) add two-arm efficacy and toxicity
monitoring with signed margins, supplied look-specific cutoffs, boundary tables,
and exact operating characteristics. Declared arm sizes support unequal
allocation. [Finite-candidate calibration](docs/rbop2-calibration.md) maximizes
power under a declared null-error limit and reports calibration and analysis
priors separately. Native automatic cutoff-grid construction, allocation
rounding and paired-endpoint rules remain open.

[Phase2Delay](docs/phase2delay.md) adds interim monitoring for delayed response,
toxicity and progression using correlated piecewise-exponential hazards and
multiple imputation. Explicit priors, posterior traces and Monte Carlo error
estimates make the Python sampling choices inspectable. Complete data reduce
to exact Beta posterior monitoring. [Calendar simulation](docs/phase2delay-calendar.md)
adds scheduled trial replay, Poisson accrual, calibrated Weibull event times and
serial operating-characteristic summaries with replay seeds. Native calibration
and reports remain open.

[PoPdesign](docs/pop-design.md) adds predictive Bayes-factor boundaries,
sticky dose exclusions, weighted isotonic MTD selection, and memory-bounded
cohort simulation with accelerated titration. Published table values, native R
results and exact small-trial enumeration validate the numerical core.
[Saved protocol reports](docs/pop-protocol-report.md) capture complete integer
cutoffs, scenario seeds, selection and allocation summaries, and Monte Carlo
errors from the same configured simulations.
[Selection plots and portable inputs](docs/pop-community-workflow.md) preserve
original dose positions and support saving, reopening and running scenario files.

[BARD stage-two methods](docs/bard.md) add covariate-adaptive allocation using
combined stage-one/stage-two history, plus utility and noninferiority OBD
selection. [BF-BLRM model fitting](docs/bard-blrm.md) implements the paper's
raw-dose-ratio model with explicit log-parameter priors, bounded sampling and
target/overdose diagnostics. [BF-BLRM decisions](docs/bard-blrm-decisions.md)
add one-step dose movement, backfill eligibility and final MTD selection.
[BF-BLRM calendar replay](docs/bard-blrm-trials.md) combines those components
with explicit arrival/outcome/delay tapes, pending backfill, compact posterior
diagnostics and complete follow-up after enrollment stops.
[Stage-two continuation](docs/bard-two-stage.md) carries eligible stage-one
patients into covariate balancing, counts them toward the total enrollment
target, and connects new assignments to final OBD selection. Priors, safety
pooling weights and tie policies are explicit. [Accelerated titration](docs/bard-titration.md)
adds one-patient dose progression, grade-2 triggers and the distinct dose-cap
transitions. [BF-BOIN titration](docs/bf-boin-titration.md) provides the corresponding
option for that stage-one model, and [BF-BOIN expansion](docs/bard-expansion.md)
continues enrollment at the fixed lower dose. The complete BF-BOIN two-stage
scenario simulator, including covariate-dependent responses and balance/OBD
summaries, remains open. The [remaining-work crosswalk](research/bard-remaining-simulation-audit.md)
identifies the recovered response model and unresolved timing/quota conventions.

[TITE-BOIN12 AL methods](docs/tite-boin12.md) add patient-level handling of
pending toxicity and efficacy, joint utility posteriors, interim dose conduct
and complete-outcome final OBD selection. The declared approximate-likelihood
model is checked against base R; final selection reduces to ordinary BOIN12.
[Bayesian data augmentation](docs/tite-boin12-bda.md) adds joint Dirichlet
imputation with explicit priors, completed-data BOIN12 posterior averaging and
interim dose conduct. [Calendar replay](docs/tite-boin12-calendar.md) connects
either conduct method to staggered arrivals, delayed endpoint observation,
suspension and final ascertainment. [Operating-characteristic simulation](docs/tite-boin12-operating-characteristics.md)
adds repeated binary trials, explicit event-time policies, replayable random streams,
and selection, allocation and duration summaries with Monte Carlo uncertainty.
Native pending-data safety details and categorical outcomes remain open.

[U-BOIN joint utilities](docs/uboin.md) add exact categorical Dirichlet
posterior moments, toxicity/efficacy admissibility and winner, proportional or
equal allocation probabilities. Two-stage conduct includes safety monitoring,
3+3 run-in, exploration and final OBD selection. Bounded cohort simulation
accepts categorical scenarios or binary Gumbel probabilities. Priors and
candidate scope are explicit. Optional Stage-I accelerated titration adds
single-patient escalation, first-DLT/second-grade-2 triggers and the source's
dose-cap and cohort top-up rules, with an observed-path planner. Grade-2 outcomes
are explicitly distinguished from DLT. [Stage-II multiple imputation](docs/uboin-multiple-imputation.md)
averages complete-data posterior calculations from supplied efficacy-prediction
draws and preserves toxicity safety and trial conduct. The delayed-response
prediction model itself remains open.

[BaCIS subgroup borrowing](docs/bacis.md) adds deterministic low/high response
classification and within-cluster hierarchical inference, including native
singleton handling. Both documented adaptive cutoff definitions are supported;
posterior draws include convergence and Monte Carlo error diagnostics. The
native variance-matched equivalent sample size calculation reports all
admissible solutions and corrects the original zero-response root-selection defect.
Its classification-model DIC uses deterministic posterior integration, with the
native Plummer penalty and full uncertainty about subgroup classification.
The latent classification posterior has analytical density, tails and moments,
with independent sampling that needs no MCMC.
[Classification density plots](docs/bacis-plot.md) use those exact curves,
preserve their one-sided limits at zero and support native or automatic ranges.
`bacis_one_trial` combines the model and equivalent sample size in the native
ten-row numerical summary, retaining full precision alongside rounded output.
[Serial operating-characteristic simulation](docs/bacis-simulation.md) adds
classification, efficacy, familywise false-positive and single-cluster rates
with Monte Carlo errors and retained sampler diagnostics. Optional subgroup
ESS summaries retain each trial's estimate and report its mean and Monte Carlo
error. Source conflicts
prevent claiming reproduction of the paper's classification tables.

[BCHM subgroup borrowing](docs/bchm.md) adds patient-weighted clustering,
co-clustering similarities and a separate similarity-weighted hierarchy for each
subgroup. It preserves the native similarity floors and rounded efficacy rule,
with bounded sampling and independent R numerical references.
[Analysis plots](docs/bchm-plots.md) show subgroup clusters, posterior means
and intervals, and subgroup posterior densities using verified R conventions.
[Named analysis reports](docs/bchm-scenarios.md) round-trip subgroup inputs,
record actual priors and seeds, and export retained posterior probabilities
as streamed CSV with sampling diagnostics.

[EffTox dose finding](docs/efftox.md) adds the bivariate efficacy/toxicity
model, elicited-probability/ESS calibration, bounded posterior fitting, modern Lp and
legacy inverse-quadratic contours, interim/final dose selection and completed-outcome
trial simulation. Simulations preserve joint outcome association and report allocation,
selection, stopping and sampler diagnostics. Published no-skipping and exploration
rules are explicit. A separate continuation-ratio core fits mutually exclusive
efficacy, toxicity and neither outcomes, retaining both marginal and conditional
efficacy. Trinary contour elicitation, dose decisions and completed-outcome
simulation are supported with the same bounded workflow.
[Trinary prior calibration](docs/efftox-trinary-calibration.md) adds explicit
sequential elicitation from marginal efficacy/toxicity means and separate
beta-moment ESS targets, with stable induced moments and fit diagnostics.

[Multc Lean and Multc99 Phase IIa](docs/multc.md) add response/toxicity monitoring
against fixed or beta-distributed historical rates, shifted comparisons, cohort
rules, full and reachable boundaries, and exact joint stopping probabilities.
The calculation retains outcome association and separates sample-cap completion
from early stopping. A calendar replay handles separate endpoint availability,
look-ahead suspension and complete follow-up with explicit timing inputs.
Serial duration simulation provides replay seeds and Monte Carlo errors.
[Saved studies](docs/multc.md#save-a-python-study-input-and-scenario-report)
capture design and timing settings, named truths, exact operating characteristics
and calendar summaries in portable JSON inputs and readable reports.
Native timing conventions and general Multc99 designs remain unresolved.

[ToxFinder two-agent dose finding](docs/toxfinder.md) adds its six-parameter
toxicity surface, explicit gamma priors and Bayesian posterior fitting with
log-parameter draws that preserve the paper's very small interaction exponents.
It also supplies first-stage dose decisions and posterior target contours.
[Physician-prior elicitation](docs/toxfinder-prior-elicitation.md) solves the
published probability and odds-moment constraints with independent integration
checks; the documented Table 1 discrepancy is preserved.
Independent base-R calculations verify published scenario surfaces and a reduced
posterior. Native second-stage information selection and full simulations remain
open; the documentation records the source ambiguity.

[bCRM single-outcome dose finding](docs/bcrm.md) adds the fixed-intercept
logistic CRM, bounded asymptotes and deterministic uniform-slope posterior
summaries. It separates mean probabilities from probabilities at the mean
slope, with independent R checks of ordinary and concentrated posteriors.
Single-outcome allocation supports target selection, an escalation cap and
cohort stopping rules with explicit probability estimates. The guide-defined
extreme-dose extra-allocation correction is also available as a stable
scalar/vector probability calculation.
The bivariate association model and full native simulation workflow remain open.

[BMA-CRM and ordinary power-model CRM](docs/bmacrm.md) add posterior model
weights, dose toxicity estimates, and overdose probabilities using the current
prior-median skeleton convention. Stable, bounded quadrature is checked against
independent R calculations, including 10,000-patient and extreme-prior cases.
Complete-outcome dose decisions include safety stopping, no-skipping and raw-rate
escalation restrictions, and the three-patient final MTD rule.

[DA-CRM for delayed toxicity](docs/dacrm.md) adds joint inference for observed
and pending outcomes, piecewise exponential event timing, and source-based
hazard-prior calibration. Six independent base-R integrations check posterior
moments, overdose probabilities, and pending-outcome probabilities. Separate
paper and CRM Suite decision policies cover dose moves, waiting, safety stopping,
and final selection. Sampling is serial with explicit memory and work limits.
[Calendar decisions and pending-outcome look-ahead](docs/crm-conduct.md) connect
these methods to time-specific record replay. Look-ahead acts only when all
possible completed outcomes agree; completed DA records use deterministic CRM.
[Trial replay and operating-characteristic simulation](docs/crm-simulation.md)
add fixed-dose cohorts, enrollment waits, calibrated toxicity timing and serial
study replication. Compact decision records retain sampling diagnostics without
accumulating posterior draws. Native reports and older-version conduct
differences remain open for BMA-CRM Simulator and CRM Suite.
[Bayesian model selection and Occam's window](docs/crm-model-selection.md)
add alternative model aggregation throughout these CRM workflows, with original
priors preserved so excluded models can reenter as observations accumulate.

[MTADF optimal biological dose finding](docs/mtadf.md) adds double-sided
isotonic efficacy fitting, elicited beta priors and pooled posterior toxicity
monitoring. Adaptive decisions include exploration, lowest-dose efficacy ties
and safety stopping. A serial simulator reports dose allocation, selection and
Monte Carlo uncertainty. [Global quadratic and local linear logistic methods](docs/mtadf-logistic.md)
add the paper's Cauchy-prior efficacy models, slope-based local decisions and
posterior diagnostics. [Logistic trial simulation](docs/mtadf-logistic-simulation.md)
adds complete-cohort trials, replayable outcome/sampler seeds and compact
operating-characteristic summaries for both designs. Independent R calculations and numerical integration
check the models; native application settings and output equivalence remain open.

[UAROET ordinal dose finding](docs/uaroet.md) adds continuation-logit outcome
models joined by a Gaussian copula, explicit-prior posterior fitting and
utility-based adaptive randomization. Safety, near-optimality and probability
of being best determine acceptable doses; good-outcome probabilities determine
allocation weights. [Complete-outcome trial replay and simulation](docs/uaroet-trials.md)
add explicit analysis schedules, patient-level randomization and compact
selection/allocation summaries with bounded serial posterior fits. Independent
R integration checks the model, a reduced posterior and reference trial paths.
Prior calibration, delayed outcomes and native workflow equivalence remain open.

[Dose Schedule Finder](docs/dose-schedule.md) adds a time-to-toxicity model for
choosing dose and administration schedule together. Triangular hazards account
for each patient's actual administration times and dose changes. Approximate
prior elicitation, bounded posterior fitting and safety-constrained nearest-target
selection support study analysis. Independent R integration checks hazards,
histories and a reduced posterior. [Calendar trial replay](docs/dose-schedule-trials.md)
adds event generation, as-of-arrival posterior updates, actual administration
histories and final follow-up, with separate replayable random streams and
shared work bounds. [Aggregate simulations](docs/dose-schedule-simulation.md)
report selection, stopping, allocation, observed toxicity and duration with
Monte Carlo errors and replayable event/sampler seed pairs.
[Delayed-toxicity observations](docs/dose-schedule-observation.md) apply explicit
adjudications as they become known, backdate qualifying events to onset, and
separate delivered treatment from likelihood exposure. Grade-2 snapshots apply
the paper's persistence or attributed-dose-reduction rule to supplied episode
histories, with an explicit deadline convention. Automatic calibration,
generated low-grade episodes and native workflows remain open.

[CiBolus](docs/cibolus.md) models immediate and subsequent response to a bolus
plus continuous infusion, with response-dependent toxicity. Exact and interval
observations, explicit priors, bounded posterior fitting and utility-based
concentration/bolus selection preserve the published treatment structure.
Independent R quadrature provides probability, likelihood and reduced-posterior
references. [Complete-outcome trials](docs/cibolus-trials.md) generate joint
response/toxicity categories, update after each cohort and apply concentration
no-skip and unrestricted final selection, with replayable outcome inputs and
cumulative work limits. Serial aggregate simulation reports selection and
observed-outcome rates, Monte Carlo errors and per-trial replay seeds.
[Interpolated scenarios](docs/cibolus-scenarios.md) construct joint truth from
response and toxicity probabilities using the paper's four curve shapes.
Trial and aggregate simulations accept these tables independently of the
fitted model, supporting model-misspecification studies.
[Prior calibration](docs/cibolus-calibration.md) adds balanced pseudo data,
posterior-mean averaging, prior probability moments and beta ESS, with explicit
joint elicitation tables and prior variances. Automatic variance selection,
calendar conduct and native workflows remain open.

[Pinnacle](docs/pinnacle.md) detects and quantifies protein spots in aligned
two-dimensional gel images. It combines streaming image averaging, undecimated
Daubechies wavelet denoising, peak detection, background correction and
normalization. Optional per-gel denoising and rectangular local backgrounds
keep detection on the denoised average of raw gels and preserve raw image-volume
normalization. A replayable TIFF source reads aligned grayscale gels one at a
time, preserves numerical intensity samples and makes multi-frame selection
explicit. It feeds the same verified analysis pipeline and retains file/frame
identifiers for matching the result rows to input images.
[Exact peak editing and CSV export](docs/pinnacle-peak-selection.md) let users
revise selected coordinates and re-quantify the same streamed images.
Explicit resource limits bound image processing. Independent R
calculations and original Rice Wavelet Toolbox C outputs provide numerical
references. Unsupported TIFF variants, native project formats and application workflow equivalence
remain open. This product includes software developed by Rice University,
Houston, Texas and its contributors; see the [preserved license](notices/rice-wavelet-LICENSE.txt).

[EasyCellType Fisher annotation](docs/easycelltype.md) accepts marker lists and
caller-supplied cell-type associations. It preserves the author's modified
Fisher table and duplicate-row conventions, reports adjusted p-values and
contributing genes, and ranks hard/soft labels. The
[ranked-enrichment workflow](docs/easycelltype-gsea.md) adds weighted GSEA
statistics, the separate fgsea/DOSE contributing-gene conventions, normalized
scores, adaptive multilevel tail probabilities, uncertainty and BH adjustment.
GSEA hard/soft labels preserve ties and DOSE contributing genes.
[Reference loading](docs/easycelltype-reference.md) filters locally supplied
author-format CSV or gzip tables by species and tissue while preserving source
rows and recording a file checksum. [Bundled reference tables](docs/easycelltype-builtin-reference.md)
provide the pinned EasyCellType 1.5.4 CellMarker, Clustermole and Panglao snapshots
without R or a download. [Gene-ID conversion](docs/easycelltype-gene-mapping.md)
adds explicit versioned Human/Mouse mappings, and
[annotation plots](docs/easycelltype-annotation-plots.md) display selected
Fisher/GSEA candidates with documented Python visual conventions.

[SurvivalContour Cox surfaces](docs/survival-contour.md) fits ordinary or
stratified Efron/Breslow models for right-censored data and returns survival
surfaces, pointwise
confidence limits and curves at selected covariate quantiles. Numeric adjustment
profiles and optional two-/three-dimensional plots support exploration of a
continuous predictor. Stratified models share coefficients and estimate separate
baseline hazards for each group. Further implemented model families are described below.

[Neural survival models](docs/survival-neural.md) add DeepSurv, CoxTime,
DeepHitSingle, LogisticHazard and PCHazard through a bounded NumPy network.
The APIs fit models, predict survival and form covariate contours with retained
training transforms, loss/epoch diagnostics and explicit Python training choices.
Independent high-precision likelihood and gradient references validate all five
families. These point predictions do not include uncertainty intervals.

[Fine–Gray competing-risk regression](docs/fine-gray.md) provides target-cause
incidence predictions, fixed and time-interaction effects, separate censoring
distributions, and sandwich coefficient covariance. Its fixed-effect contour
workflow includes optional two-/three-dimensional incidence plots. Exact
censoring left limits and stable prediction arithmetic accompany bounded
array allocations. The interval-censored model below uses a separate joint
likelihood for both causes.

[Parametric AFT survival models](docs/parametric-survival.md) add exact Weibull,
log-normal and log-logistic fits, joint coefficient/scale covariance and
survival predictions. Continuous-covariate contours and selected percentile
curves include deterministic delta-method pointwise bounds.
[Generalized-gamma survival models](docs/generalized-gamma.md) extend this
workflow with stable Prentice and original Stacy fits, full joint covariance,
predictions and the same contour plots.
[Simulated pointwise survival limits](docs/survival-uncertainty.md) add joint
parameter draws for all five parameterizations and the three spline links,
reusable across profiles and time grids, with valid-draw counts and bounded
memory use. Spline draws retain the native unrestricted Gaussian convention
and report minimum slopes so rising simulated curves remain visible.
Parametric and spline contours can use these simulated limits with one shared
parameter-draw matrix for the main surface and selected quantile curves.
[Royston–Parmar spline models](docs/survival-spline.md) add hazard, odds and
normal links with configurable log-time knots, globally monotone survival
curves, joint covariance, predictions and contours.
[Random survival forests](docs/random-survival-forest.md) add log-rank trees
with continuous and explicitly declared categorical predictors, separately
averaged Kaplan–Meier survival and Nelson–Aalen hazards,
and contours from fitted forests. Sequential tree growth, sparse leaf curves
and explicit work limits bound computation. Optional [out-of-bag diagnostics](docs/random-survival-oob.md)
report held-out survival/hazard curves, contributor counts and concordance error.
[OOB Brier scores and integrated CRPS](docs/random-survival-oob-brier.md)
add either the source's global censoring estimate or a separately fitted
50-tree censoring forest, per-observation contributions and prediction-error
curves, including the native event-grid and tie conventions.
Permutation, anti-split and [random-routing importance](docs/random-survival-forest-random-importance.md)
add per-tree OOB perturbations and
blockwise error increases, with explicit counts for usable blocks and omitted
tail trees. Category maps are retained for consistent prediction and importance;
continuous contour axes support categorical adjustment profiles.
An optional [Hothorn–Lausen split rule](docs/random-survival-forest-logrankscore.md)
uses standardized survival rank scores, including maximum-rank time ties and
bootstrap multiplicities. The existing log-rank rule remains the default.
The optional [Brier-gradient split rule](docs/random-survival-forest-brier.md)
follows RF-SRC's scalar event-grid and censor-weight conventions, validated
against its original C helpers, with linear-size node workspace.
The [random split rule](docs/random-survival-forest-random-split.md) draws one
continuous cut or categorical partition on the first usable selected feature;
the censoring-forest Brier workflow uses this rule with the source's settings.
[Missing-data forest fitting](docs/random-survival-forest-missing.md) adds
complete-case omission with original-row maps and single-pass, node-local
imputation from observed in-bag donors. Seeded prediction can complete missing
profiles using retained training donors. OOB concordance completes missing
outcomes from OOB terminal summaries, and global-censor Brier scores support
imputed predictors with complete outcomes. Imputed-outcome Brier scores,
imputed-fit importance and repeated imputation remain open.
[Interval-censored proportional-hazards models](docs/interval-survival.md)
fit mixed exact, interval, left- and right-censored observations with a
nonparametric Turnbull baseline. Predictions and continuous-covariate contours
return explicit survival identification bounds, preserving uncertainty within
observation intervals and beyond the last finite observation.
[Coefficient bootstrapping](docs/interval-survival-bootstrap.md) adds ordinary
interval-PH regression covariance and standard errors, with weighted resampling,
explicit failed-fit records and replayable row selections.
[Cluster coefficient bootstrapping](docs/interval-survival-cluster-bootstrap.md)
resamples whole groups, retaining repeated selections and variable sample sizes
without materializing duplicate rows.
[Stratified interval models](docs/interval-survival-stratified.md) fit shared
covariate effects and separate group baselines jointly, with matching
group-specific predictions and contours.
[Stratified coefficient bootstrapping](docs/interval-survival-stratified-bootstrap.md)
provides shared-coefficient covariance with explicit within-stratum or pooled
resampling, retained row tapes and visible failed-refit records.
[Interval-censored competing-risk models](docs/interval-competing-risk.md) fit
two causes jointly with monotone spline baselines and generalized odds-rate
links. They return regression covariance, both incidence curves and
continuous-covariate contours, with optional two-/three-dimensional plots.
An explicit starting-boundary approximation and constrained-convergence
diagnostics accompany probability checks on requested profiles. Repeated-visit
data can be converted into subject-level intervals with baseline covariates
and source-row provenance. [Competing-risk coefficient bootstrapping](docs/interval-competing-risk-bootstrap.md)
adds covariance and standard errors from complete resampled refits, with explicit
failure records. Broader categorical encoding, remaining forest workflows and
full native application workflows remain open.

[STPLAN study planning](docs/stplan.md) provides all 25 forward power and retention
procedures for normal, log-normal, exponential, correlation, binary, and Poisson
outcomes. These include exact one-sample count tests, historical controls,
K-group comparisons, attrition, matched/unmatched case-control studies, and
two-sample Poisson counts, and all five censored-survival menu methods. Original
Fortran references and independent R sums/integration check numerical behavior,
including corrected matched/Poisson mixtures and piecewise-survival integrals.
[Bounded inverse planning](docs/stplan-planning.md) solves for sample sizes,
effects, significance, and timing, with integer attainment searches and K-group
allocation. [Exact count-test significance planning](docs/stplan-significance.md)
selects binomial and Poisson critical regions with explicit power attainment.
Native automatic planning workflows, other discrete inverse questions,
and reporting remain open. [Survival input conversions](docs/stplan-survival-inputs.md)
cover medians, survival percentages, historical person-time, and piecewise curves.
The [historical-control planner](docs/stplan-historical-planning.md) also finds
accrual duration and new-control allocation jointly, including boundary solutions.
The archive's [legacy matched-pairs binary method](docs/stplan-matched-pairs.md)
adds pilot-based power and inverse planning, plus preliminary-size recommendations
when pilot observations are unavailable. Its two-sided approximation retains only
the rejection tail in the direction of the effect, as in the original routine.
