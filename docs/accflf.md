# ACCFLF: accelerated failure-time log-F models

MD Anderson entry 16, by Barry W. Brown, fits log-F accelerated failure-time
models with right censoring and covariates. The supplied Fortran 95 source,
LaTeX manual and example data are in `ACCFLF_V1.tar.gz` from the
[software page](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/16).
The archive contains a source directory labelled 1.0 and a Linux binary directory
labelled 1.1. The README incorrectly describes a binomial-design program; this
port follows the actual ACCFLF source and manual. File hashes are recorded in
[accflf-sources.json](accflf-sources.json).

**Coverage is partial.** The log-F probability/derivative engine, fixed-(p,q)
likelihood, regression fitting, survival prediction, profile shape searches,
rectangular grids and all six named-model comparisons are implemented.
Covariate selection/update workflows, source-format readers and reports remain
pending.
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
validation is described below. The complete file/report workflow remains pending.

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
Original KP patient data remain excluded. Covariate manipulation and original
file/report adapters are still pending.
