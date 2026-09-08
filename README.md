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
`single_optimize_two_sample_allocations` allocates a shared subject total across
both groups under point or weighted priors.
`single_optimize_design` jointly moves dose locations and allocations for a fixed
number of dose entries, supporting one/two samples and weighted priors.
[SINGLE coverage](docs/single.md) tracks
the remaining prior-integration and optimization workflows.
