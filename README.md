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
[STUKEL fitting remains pending](docs/stukel.md).
[The MULTI coverage audit](docs/multi-coverage.md) records its remaining I/O gaps.
