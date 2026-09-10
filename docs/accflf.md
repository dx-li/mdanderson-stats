# ACCFLF: accelerated failure-time log-F models

MD Anderson entry 16, by Barry W. Brown, fits log-F accelerated failure-time
models with right censoring and covariates. The supplied Fortran 95 source,
LaTeX manual and example data are in `ACCFLF_V1.tar.gz` from the
[software page](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/16).
The archive contains a source directory labelled 1.0 and a Linux binary directory
labelled 1.1. The README incorrectly describes a binomial-design program; this
port follows the actual ACCFLF source and manual. File hashes are recorded in
[accflf-sources.json](accflf-sources.json).

**The supplied ACCFLF workflow is implemented.** The log-F probability/derivative engine, fixed-(p,q)
likelihood, regression fitting, survival prediction, profile shape searches,
rectangular grids and all six named-model comparisons are implemented.
Named-table input, covariate selection/addition, covariate-averaged survival
and fit/search/grid/comparison reports complete the supplied workflow.
No bundled ACM/Fortran implementation or original patient dataset is distributed.

## Model and shape convention

For positive event/censoring times, the model is

```
log(T_i) = intercept + X_i @ beta + sigma * W_i
exp(W_i) ~ F(numerator_df, denominator_df)
```

Let a=dfn/2 and b=dfd/2. The source's Prentice parameters satisfy
p=2/(a+b), q=(1/a-1/b)/sqrt(1/a+1/b). `accflf_shape(p,q)` converts these
parameters and reports the resulting degrees of freedom, tau and clipping flags.
It preserves the source's near-origin rule q²+2p<4e-10, which sets both
degrees of freedom to 1e10. Elsewhere it uses a cancellation-free algebraic
inverse in place of the source's truncated
small-ratio polynomial. Like the source, it restricts each degree of freedom to
[0.001, 1e10]. The upper bound approximates infinite degrees of freedom. Lower
clipping is flagged if either degree is clipped; the source combines its two
lower-bound flags with AND and can miss one-sided clipping. p must be
nonnegative; p and abs(q) are limited to 1e150 to keep the conversion finite.

| Model from the source | p | q | sigma |
| --- | --- | --- | --- |
| Lognormal limit | 0 | 0 | Estimated |
| Weibull limit | 0 | 1 | Estimated |
| Exponential limit | 0 | 1 | Fixed at 1 |
| Log-logistic | 1 | 0 | Estimated |
| Reciprocal Weibull limit | 0 | -1 | Estimated |
| Generalized gamma boundary | 0 | User-specified nonzero value | Estimated |

These boundary models use the original finite-df approximations, rather than
silently changing the underlying likelihood to a different exact limit. In
particular, for p=q=0, tau=50000 and the approximately normal standard deviation
of log time is sigma/tau, **not sigma**. This explains the large reported sigma
for the lognormal example. A coefficient has the usual multiplicative AFT effect
exp(beta) on time for a unit covariate change.

## Probabilities and derivatives

`accflf_logf(w,numerator_df,denominator_df)` accepts arbitrary-shaped real arrays
and returns immutable log density, log CDF and log survival, plus the first and
second derivatives of each with respect to w. Degrees of freedom must lie within
the source's supported bounds; up to two million w values are allowed per call.

The internal native `LLDRLF(case=1)` omits log(tau), where
`tau=sqrt(1/(2/dfn+2/dfd))`. ACCFLF restores this factor when constructing its
likelihood. Python's public `log_density` includes it already and integrates to
one in w. First and second w derivatives are unaffected by this constant.

The implementation reuses the package's stable beta-density factors, evaluates
both beta coordinates independently, and keeps tiny probabilities in log space.
It passes the smaller beta coordinate to the lower or complementary beta
function; passing a rounded near-one coordinate previously caused inaccurate
shape-profile likelihoods for highly unequal degrees of freedom. Seven added
native cases (df=8 and 1e10) verify this correction at 1e-12 tolerance.
Underflowed beta tails use a continued fraction; their derivatives use the
fraction ratio directly to avoid subtracting two very large negative logarithms.
Unit-shape tails have closed-form expressions. Curvature near a zero limiting
value can lose relative precision; unrepresentable results and nonconvergent
fractions raise errors rather than returning fabricated finite values.

## Fixed-shape fitting and prediction

```python
import numpy as np
from mdanderson_stats import fit_accflf, accflf_survival

time = [1, 2, 3, 4, 6, 8, 9, 12, 15, 19, 23, 30]
event = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0]
x = np.tile([-1.0, 0, 1], 4)[:, None]
fit = fit_accflf(time, event, covariates=x, p=0.7, q=0.4)
print(fit.sigma, fit.coefficients, fit.log_likelihood)
survival = accflf_survival(
    time,
    p=fit.p,
    q=fit.q,
    sigma=fit.sigma,
    coefficients=fit.coefficients,
    covariates=x,
    log=True,
)
```

`fit_accflf` adds the intercept automatically; covariates have one row per
observation and at most 16 columns. event=1 denotes a failure, event=0 denotes
right censoring. All times must be positive, with at most 20,000 rows. Optional
positive `weights` act as case weights; integer weights equal replicated rows.
Set `fixed_sigma=1` for the exponential submodel. The fit requires at least one
failure, varying log times and a full-rank design.

Optimization uses analytic gradients in log(sigma) and the linear coefficients,
with standardized log times and covariates. Estimates and observed-information
covariance are transformed back to original units. A failed score check or
nonpositive/singular information raises an error. The result includes iteration
count and maximum absolute score per total weight in standardized optimization
coordinates; covariance is conditional on the supplied p and q, not uncertainty
from estimating shape. Its coordinates are **[log(sigma), intercept, beta...]**;
the fixed-sigma row/column are zero. This fit does not estimate p or q.

`log_likelihood` follows the source's printed likelihood for log times.
`time_log_likelihood` additionally subtracts sum(weight*event*log(time)), the
Jacobian for a density on event times. The difference is parameter-independent
for the same data, so either yields the same MLE. `accflf_loglikelihood` evaluates
supplied parameters with this same convention; set `time_density=True` for the
second convention. `accflf_survival` predicts one positive time per covariate row;
`log=True` preserves tiny survival probabilities without underflow to zero.

## Validation

The original Fortran sources compiled with gfortran without source edits. A
small driver called LLDRLF for density, CDF and survival and their first two
derivatives over thirteen degree-of-freedom pairs and seven positions each, including
w=±1000 and the upper df bound. The normalized density adjustment is recorded in
[the fixture](../tests/fixtures/accflf-native.json). For the tested moderate degrees of freedom (up to 40),
errors are approximately machine precision for values and below 1e-12 for the
derivative comparisons. At df=1e10, the largest curvature discrepancy divided by
1+abs(reference) was below 4e-9; density/log-tail discrepancies on that scale were
below 5e-11. Reflection, complementary tails and numerical density normalization
are also checked.

An independent R fit using df/pf and BFGS on synthetic weighted censored data at
p=.7,q=.4 agrees in coefficients/log sigma within 9e-8. Focused checks verify
analytic likelihood gradients and Hessians, frequency-weight replication,
time-unit scaling by 1e150, covariance transformation, the exponential MLE and
survival limit, and rejection of all-censored data.

The original KP example was read locally, without redistributing its data. The
four fixed-shape models with its covariate reproduce the manual's printed
likelihoods:

| Model | Python log likelihood | Manual (rounded) |
| --- | --- | --- |
| Weibull | -20.6177785063 | -20.6178 |
| Exponential | -42.9991516550 | -42.9992 |
| Lognormal | -22.8900863412 | -22.8901 |
| Log-logistic | -21.6134298844 | -21.6134 |

These checks establish the supplied fixed-shape computations; shape search
validation is described below.

A local throughput check evaluated all nine kernel outputs for 100,000 positions
(df=3,8; w from -10 to 10) in 0.041 seconds, excluding package import. This is
a single-machine measurement, not a cross-platform performance guarantee.


## Shape searches, grids and six-model comparisons

The original `psftdo`/`pqnll` performs an outer search over p,q, refitting
sigma, intercept and covariates at each evaluation. `search_accflf` follows this
profile-likelihood structure, using bounded Nelder-Mead instead of the bundled
David Gay optimizer. By default it starts at (.5,.5), (.5,-.5) and (5,0), using
log1p(p) and signed-log1p(q) coordinates. Native bounds are p in [1e-10,1e10]
and q in [-1e10,1e10]; the finite-df restrictions above also apply.
`fixed_p=0` estimates the generalized-gamma boundary, with starting q values
+.5 and -.5. Any nonnegative fixed p up to 1e10 is supported. Custom `starts`
contain p,q pairs, with their p coordinate equal to fixed_p when supplied.

Every profile evaluation performs a complete fixed-shape fit. Exact repeated
shape pairs are cached within a search. Both degrees of freedom clipped below
the native lower bound constitute an inadmissible shape-search point, as in the
source. Numerical failures likewise cannot improve the objective; their shape
coordinates and error messages remain in `failed_shapes`. This is explicit
optimizer-domain handling, not a substituted finite likelihood. One-sided
clipping is allowed as in the source and is visible in `best.shape`.

`best` is the highest-likelihood endpoint among the runs, including any run that
hit its evaluation limit. Always inspect `converged` and `runs`: convergence
means that a run ending at the selected best fit satisfied both simplex
coordinate and likelihood-spread tolerances. It does not establish global
optimality, identifiability or a unique shape estimate. `tolerance` defaults to
1e-6 (absolute tolerances in transformed coordinates and log likelihood), and
`max_evaluations` defaults to 500 per start. A search allows up to 20 starts,
5,000 evaluations per start and 20 million observation/evaluation pairs.
If no run produces a valid endpoint the routine raises an error.

The nested `best.covariance` describes coefficients/log-sigma conditional on the
selected p,q. It must **not** be interpreted as covariance incorporating shape
estimation. No shape standard errors or automatic likelihood-ratio p-values
are supplied by this interface.

`scan_accflf` refits every pair in a rectangular grid, matching
`scan_over_several_values`. Its likelihood matrix has p rows and q columns;
`fits` and `errors` use the corresponding row-major order. Numerical failures
remain NaN likelihoods/None fits with an error message. `best` is the best
successful point, or None if all fits failed. Grids allow 1–100 values on each
axis and at most 20 million observation/grid-point pairs. Grid success alone
is not an optimization certificate.

```python
import numpy as np
from mdanderson_stats import scan_accflf, search_accflf

rng = np.random.default_rng(1601)
raw = np.exp(2 + 0.7 * np.log(rng.gamma(1.5, 1 / 1.5, size=60)))
censor = np.exp(rng.normal(2.8, 0.4, size=60))
event = (raw <= censor).astype(int)
time = np.minimum(raw, censor)
grid = scan_accflf(time, event, p=[0, 1], q=[-0.5, 0.5, 1])
search = search_accflf(time, event, fixed_p=0)
print(search.best.q, search.best.log_likelihood, search.converged)
```

`compare_accflf(time,event,covariates=...,weights=...)` covers the source's
`fit_all_models`, returning records in order: generalized F, generalized gamma,
Weibull, exponential, lognormal and log-logistic. Each record has a model name,
fit, optional complete search diagnostics, and optional numerical error.
A numerical failure stays in the comparison rather than silently removing a
model. Generalized F/gamma records require the same convergence assessment as a
direct search. This convenience function accepts the same search evaluation
limit and tolerance.

### Search validation and limitations

The 60-subject synthetic example above gives q=0.80943648 and profile log
likelihood -62.1520594155. An independent R implementation using the exact
limiting generalized-gamma distribution, nested BFGS and scalar minimization
on positive/negative q intervals [.1,3] and [-3,-.1] agrees in q/log-sigma/intercept
within 3e-7 and log likelihood within 2e-10. The comparison is against that
limiting model; Python retains the native finite-df boundary convention.
The focused test also checks grid orientation/best-fit selection and explicit
nonconvergence when a run reaches its ten-evaluation limit.

All 25 p,q points in the original KP example grid reproduce the manual's printed
likelihoods (maximum difference below 4.6e-5, within its rounding). The six-model
comparison reproduces all six printed likelihoods, including generalized gamma
at q=0.8691031, LL=-20.5652717849 and general log-F at LL=-19.4510556545.
The original source uses different starting points/optimizer trajectories; no
iteration-by-iteration optimizer parity is claimed.

The general log-F maximum has a flat ridge at the lower df boundary, as the
manual warns. The Python example returns p=1955.65,q=1.8150, while the manual
shows p=1952.23,q=1.70091 for one detailed fit. The effective degrees of freedom,
sigma=0.0001503863, intercept=4.7537411 and covariate coefficient=0.1368605 match
the detailed native result to its printed precision. Different p,q coordinates
can map to the same clipped df pair. The 550 evaluated shapes included 28
inadmissible/failed points; these are recorded, not hidden. This is a fit on a
numerical boundary, not evidence of uniquely identified unconstrained shapes.
The complete six-model run took about 43 seconds locally.

The synthetic data, independent R reference, KP grid values and comparison
summaries are recorded in [accflf-search.json](../tests/fixtures/accflf-search.json).
Original KP patient data remain excluded. File, covariate and report adapters
are described below.


## Original table, covariate and reporting workflow

`read_accflf_data(path, covariates=(...))` implements the numeric table reader and
column-role assignment. It reuses the existing QLEX lexer, accepts optional
identifier headers, ignores blank and full-line comment records (default `#`),
and rejects mixed text/numeric or ragged records. Names are uppercase and
truncated to eight characters, matching the native reader; collisions after
truncation raise an error. Headerless columns receive X1, X2, ... . Named roles
are case-insensitive, or can be supplied as **zero-based** integer indices.
Use explicit `time` and `event` roles when TIME and STATUS names are absent.

A column named MULTI is detected automatically; `multiplicity=None` disables
that role, and an explicit name/index selects another column. File multiplicity
must contain positive integers, matching the native input check. Time must be
positive, status must be zero or one, and all values must be finite. Numeric
lexer overflow/underflow is rejected instead of silently importing a truncated
value. At most 20,000 rows and 200 columns are supported. Inline comments are
not part of the source table grammar.

The returned `AccflfData` retains the full immutable table, including unselected
covariates. `.select((...))` replaces the covariate set, `.add((...))` adds unused
columns, and `.select(())` gives an intercept-only model. They return new data
views without mutating earlier selections or fit results. `.available_covariates`
identifies unused eligible columns. Duplicate selections, reserved time/status/
multiplicity columns and more than sixteen covariates raise errors. Selecting
columns is an explicit modeling choice, not automatic statistical selection.
This replaces the source's covariate menus without its special-case allocation
and label-indexing problems when only one covariate remains.

```python
from pathlib import Path
from mdanderson_stats import (
    read_accflf_data,
    fit_accflf,
    accflf_marginal_survival,
    accflf_report,
)

# Supply a source-format file containing TIME, STATUS and COV columns.
data = read_accflf_data("KP.data").add(("COV",))
fit = fit_accflf(
    data.time,
    data.event,
    covariates=data.covariates,
    weights=data.weights,
    p=0,
    q=1,
)
report = accflf_report(fit, covariate_names=data.covariate_names)
Path("fit-report.txt").write_text(report)
survival = accflf_marginal_survival(
    [50, 100, 200],
    p=fit.p,
    q=fit.q,
    sigma=fit.sigma,
    coefficients=fit.coefficients,
    covariates=data.covariates,
)
```

The same selected data feed `search_accflf`, `scan_accflf` and `compare_accflf`.
Calling `.select` or `.add` and fitting the resulting design is the native
change/add-covariate workflow. Reading another file replaces the source's
change-dataset menu. Supplied parameter evaluation uses `accflf_loglikelihood`
and the prediction functions without requiring a fit object.

`fit_accflf` exposes `tolerance` (default 1e-6) and `max_iterations` (default 500).
The tolerance is the maximum normalized score error in standardized coordinates,
with a tighter internal optimizer target. This is an explicit Python convergence
criterion, not the native optimizer's relative-function criterion. Supported
ranges are 1e-10..1e-2 and 1..10,000 iterations. Search tolerances/limits remain
separate as described above; fixed-shape failures do not return fake converged fits.

### Covariate-averaged survival

`accflf_marginal_survival` ports the source's `srvprb`: for **each** requested
time, it averages survival over **all** supplied covariate rows. This differs
from `accflf_survival`, which pairs one time with one covariate row. The native
routine uses equal row weights and **ignores MULTI**, even when multiplicity
weighted the fit; Python preserves that convention explicitly. To average over
individuals represented by frequency counts, supply the expanded covariate rows.
That is a different averaging population from the default native report.

Repeated linear predictor values are grouped with their counts, preserving
row frequencies while reducing work. Time batches limit temporary arrays;
up to 20 million distinct predictor/time pairs are supported. `log=True` returns
the logarithm of the averaged survival, calculated with log-sum-exp, not the
average of log survival. Supply `data.time` for predictions at every observed
time, or any positive-time vector. Nonpositive times raise an error instead of
being silently discarded as in the interactive native time-list handler.

`accflf_report` accepts a fixed fit, shape search, grid or six-model comparison.
Fit reports include p,q,sigma,mu,covariate coefficients, df/tau and clipping flags,
both likelihood conventions and conditional covariance with its coordinates.
Search reports include every start and failed shape plus local-convergence
status; grid reports retain numerical errors at their original points; model
comparisons retain failed-model records. Reports are returned as text for
printing or explicit file writing. This preserves numerical report content,
not the native terminal prompt layout or optimizer iteration trajectory.

The original 40-row KP table matches the unchanged compiled `read_table` output
elementwise. A synthetic eight-time/four-covariate-row average using unchanged
native `tailf` and the `srvprb` loop agrees within 1.2e-16; its fixture is
[accflf-marginal.json](../tests/fixtures/accflf-marginal.json). Integration checks
cover covariate changes, frequency input validation, report writing, duplicate-row
averaging, extreme log survival and an explicit iteration-limit failure. A
separate native-data run completed all six fits, observed-time marginal
predictions and every report variant. Original patient data are not redistributed.

## Source coverage reconciliation

| Original role | Python coverage |
| --- | --- |
| pqtodf and df bounds | accflf_shape; near-origin limit and explicit clipping |
| lldrlf, ltlf, tailf and auxiliary series | accflf_logf; stable beta factors/tails and analytic derivatives |
| compute_f_and_derivatives, find_ll | accflf_loglikelihood and analytic weighted score/information |
| pqnll, iniest, mvlogf | fit_accflf; fixed-shape optimization and moment initialization |
| psftdo, perform_optimization | search_accflf; profile shape search and explicit local termination |
| single_fit and fit_all_models | Fixed/custom shape fitting, generalized-gamma/F searches and compare_accflf |
| scan_over_several_values | scan_accflf rectangular grid |
| read_table, tokenizer, get_parameters_columns, read_in_data | read_accflf_data and existing QLEX lexer; named roles, validation and frequency counts |
| add_covariates, get_new_covariates, covariate displays | AccflfData.add/select and available/selected names |
| srvprb and find_survival_probabilities | accflf_marginal_survival over observed or supplied time lists |
| enter_parameters | Explicit model parameters in likelihood and prediction APIs |
| change_convergence_criterion | Fixed-fit score tolerance and shape-search tolerances/limits |
| Header, constraints, parameter, df/tau, grid/model reports | accflf_report and standard Python text/file output |
| Bundled math, sorting, strings, interface and David Gay optimizer | Existing package numerics/QLEX, NumPy/SciPy, typed arguments and result objects |

The source's interactive menus, memory management, stale binary-cache comments,
and platform build scripts introduce no additional statistical methods. Coverage
is complete for the supplied workflow with the documented Python semantics;
local optimization, finite-df bounds and shape identifiability limitations remain.
