# CUMINC cumulative-incidence curves

Catalog entry 39 is **implemented**. Curve estimation, Aalen variance,
multi-group/stratified Gray tests, pointwise confidence intervals, combined
numerical summaries and plots are covered by the [source audit](cuminc-coverage.md).

Source: [CUMINC](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/39),
contact Ken Hess, distributed as `CUMINC_V1.tar.gz`. Its S-PLUS interface calls
Fortran CINC for the curves and CRSTM/CRST for group comparisons. Original source
and data remain local research inputs; no original source or binary is bundled.

```python
from mdanderson_stats import cumulative_incidence

curve = cumulative_incidence([1, 2, 3, 4], [1, 2, 0, 1])
estimate, variance = curve.at([0, 1, 2, 3, 4])
print(estimate)  # [0, 0.25, 0.25, 0.25, 0.75]
```

`time` and `event` are nonempty matching vectors. Times must be finite and
nonnegative. Event codes are nonnegative integers; the default censoring code is
0 and the event of interest is 1. All other noncensoring codes are competing
events. Input order need not be sorted. Missing values are rejected explicitly;
this core API does not silently apply the S-PLUS interface's complete-case deletion.
`cuminc` supports string cause, group and stratum labels; `gray_test` supports
string group and stratum labels.

At each distinct time, the risk set includes all observations whose follow-up
ends at that time, including tied censored observations. Overall survival falls
according to all failure causes. The incidence increment for the selected cause
is pre-event survival times its failure count divided by the risk set. Treating
competing events as censored would compute a different estimand.

The result's `time`, `estimate` and `variance` arrays are read-only step-function
corners, matching CINC's layout: initial zero, paired left/right limits at each
selected-cause event time, and extension to maximum follow-up. `.at(times)`
returns right-continuous values, including events at time zero, and extends the
last value beyond observed follow-up. This extension is a step-function
convention, not extrapolated risk estimation. An absent event of interest gives
a zero curve. `n_observations` and `n_events` count people; the original S-PLUS
interface labels the number of distinct selected-event times as an event count,
which understates individuals when failures tie.

Variance follows the original Aalen-style finite-risk-set correction. It is an
estimated asymptotic variance, not the empirical Bernoulli variance or a bootstrap
variance. In particular, CINC's terminal risk-set-of-one convention contributes
the squared preceding survival; it is retained, including the singleton case.
A centered recurrence updates the variance and its covariance terms directly,
avoiding repeated subtraction of large raw-moment terms. Negligible negative
roundoff is clipped to zero; materially negative or nonfinite results raise.

Sorting and vectorized time aggregation take O(N log N) work and O(N) storage;
variance accumulation takes O(U) work over U distinct times. Repeated curve
queries use array binary search. There is no subject-by-subject covariance matrix.

## Validation and remaining work

`tools/reference_cuminc.py` extracts unchanged CINC and invokes it with an
independent driver. Thirteen fixtures include censoring, competing risks, ties,
time zero, absent events, singleton observations, and both causes within each
group of the supplied `test.data`. Source/data/extraction/driver/compiler hashes
identify the oracle. Independent tests cover empirical incidence without
censoring, risk-set treatment of tied censoring, the first-jump variance formula,
competing-event handling, permutation, time scaling, code remapping and invalid
inputs. A 10,000-observation uncensored curve was also evaluated successfully.

## Group comparisons

```python
from mdanderson_stats import gray_test

result = gray_test(
    [1, 1, 2, 2, 2, 2, 2, 2],
    [1, 1, 0, 0, 0, 0, 0, 0],
    ["treatment"] * 4 + ["control"] * 4,
)
print(result.statistic, result.pvalue)  # approximately 2.33333, 0.12663
```

`gray_test` compares one selected cause across two or more groups. Optional
`strata` defines separate risk sets; scores and covariance matrices are summed
across strata before testing. A stratum may omit some groups. Event/censor codes
follow `cumulative_incidence`; group and stratum labels may each be all strings
or all finite numbers. Labels are sorted, and the final group is the reference
for the exposed score vector and covariance matrix. Changing that reference
preserves the test statistic. Missing labels or observations are rejected.

`rho` is a finite real number controlling the weight `(1 - pooled_incidence)**rho`;
the default is zero. The test uses Gray's pooled incidence estimator, which need
not equal the incidence curve obtained by merging all groups. Risk-set and
influence-moment updates are vectorized over groups. For U distinct times and G
groups, dense covariance accumulation is O(U G³) work and O(G²) working storage,
in addition to O(N) input/sorting storage; strata are processed separately.

The statistic is computed by a linear solve as `score @ solve(covariance, score)`.
The p-value uses the chi-square upper tail with G−1 degrees of freedom, as
explicitly specified in CRSTM and in the method's reference,
[Gray (1988)](https://doi.org/10.1214/aos/1176350951). This corrects the archived
S wrapper's use of a normal tail of the squared statistic. This is an asymptotic
test, not an exact finite-sample test. If covariance lacks full numerical rank,
`statistic` and `pvalue` are `None`; `rank`, scores and covariance remain available.
The implementation does not silently discard a group or change degrees of freedom.
Materially indefinite covariance or nonfinite arithmetic raises a numerical error.
Extreme rho values can exceed floating-point range.

`tools/reference_gray.py` builds unchanged CRSTM/CRST with an independent driver.
Fifty-seven native cases cover 2–4 groups, rho −0.5/0/1/2, ties, censoring,
stratification, absent groups within strata, degenerate events, and the supplied
sample's two causes with and without strata. Group and stratum labels are recoded
to consecutive integers for the Fortran contract. Fixtures record source,
extraction, driver, supplied-data hashes and compiler version. Independent tests
cover a hypergeometric first-event variance, chi-square tail identities, stratum
addition, reference-group/row/time/code transformations, fully and partially
singular comparisons, and input validation.

## Combined analyses and reports

```python
from mdanderson_stats import cuminc

study = cuminc(
    [0, 1, 2, 3],
    ["relapse", "relapse", "death", "censored"],
    ["A", "A", "B", "B"],
    censor="censored",
    confidence=0.9,
)
print(study.report(times=[0, 1, 2, 3]))
selected = study.summaries([0, 1, 3], causes="relapse", groups="A")
# study.write_report("cuminc.txt", times=[0, 1, 2, 3])
```

`cuminc` fits every observed noncensoring cause within every group and computes
one Gray test per cause when multiple groups exist. Optional `strata` only changes
the tests; descriptive curves still use the full group. The result exposes
immutable `curves[(cause, group)]`, `tests[cause]` and `group_counts` mappings.
Curve objects use internal consecutive event codes; mapping keys preserve the
original labels. Each label vector must contain all strings or all finite numbers.
An omitted group represents one group. All-censored input has group counts and
empty cause, curve and test collections.

Missing observations are rejected by default. `missing="drop"` explicitly enables
the archived interface's joint complete-case deletion over time, cause, group and
stratum. `None` and numeric NaN are missing; infinity remains invalid. `n_dropped`
records the excluded row count. Removing every row raises an error.

`curve.summary(times=None, confidence=0.95)` returns an `IncidenceSummary` containing
five read-only columns: time, incidence, standard error, lower limit, upper limit.
Intervals follow the archive's normal approximation, clipped to [0, 1]. They are
pointwise asymptotic intervals, not simultaneous confidence bands. The quantile
uses the small tail to avoid rounding the upper probability to one at confidence
levels near 100%. Zero variance stays finite at these levels.

Omitting summary times preserves all stored step corners, including left/right
limits. Explicit scalar or vector times use right-continuous evaluation, preserve
input order and duplicates, and extend the last observed value past follow-up.
The archive delegates time selection to the unavailable S-PLUS `survindex2` routine;
these Python selection semantics are explicit and tested. `study.summaries`
selects exact cause/group labels; ambiguous partial matching is not used.

`summary.report` and `study.report` return text; `write_report` writes UTF-8 files
and propagates filesystem errors. Significant-digit precision is configurable
from 1 to 17. Interval column labels reflect the actual confidence level, fixing
the archive's unconditional “95%” labels. Study reports include individual event
counts, Gray statistics/p-values/rank (NA for singular tests), and curve summaries.
Formatting and interval evaluation do not recompute the fits.

Validation applies the archived interval formula to all 13 native CINC fixtures
at two confidence levels using an independent standard-library normal quantile.
Combined-interface tests compare 48 native Gray cases through string cause/group
labels and native CINC curves through the single-group interface. Additional
checks cover time-zero and duplicate queries, interval nesting, extreme confidence,
selection, all-censored data, missing-data policy, stratification, immutable results,
precision, file replacement and report round trips.

## Plots

With the optional `plot` extra installed:

```python
from mdanderson_stats import plot_cuminc

axes = plot_cuminc(study)  # all cause/group curves on one axis
panels = plot_cuminc(study, overlay=False)  # pointwise confidence limits
selected = plot_cuminc(study, causes="relapse", groups="A", overlay=False)
# axes[0].figure.savefig("cuminc.png", dpi=150)
```

Overlay plots omit confidence limits, as in the original. Individual panels show
the fitted incidence and both pointwise confidence limits using the study's
confidence level. Rows are causes and columns are groups; a single selected group
uses one horizontal row of cause panels. Selection uses exact labels and preserves
requested order. `xlabel` and `ylabel` customize labels. `legend_at=(x, y)` places
the overlay legend's upper-left corner in data coordinates. Empty selections and
all-censored studies raise an explicit no-curves error before creating a figure.

Returned axes allow font, color, size and other Matplotlib customization, replacing
S-PLUS graphical keyword forwarding. Plotting does not show or save automatically,
change the global backend/style, or recompute the statistical fits. Curves retain
the native step corners, including time-zero jumps. Constant-zero curves and
zero-time datasets use nondegenerate axis limits. Overlay legends use automatic
placement by default; colors, line styles and constrained layout replace the
original device settings. These are functional plots, not pixel reproductions.

Plot tests check exact curve and confidence-limit coordinates, panel ordering,
selection, legend placement, labels, unchanged global style and PNG/SVG export.
Both layouts were also rendered and visually inspected on the supplied dataset.
Study reports retain censor label, rho, confidence and stratum labels, including
single-group analyses with no Gray test.
