# CUMINC cumulative-incidence curves

Catalog entry 39 is **partial**. One-cause curve estimation and the original
Aalen variance convention and multi-group/stratified Gray tests are implemented.
Confidence intervals, full multi-curve summaries and plots remain pending.

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
String cause labels will be addressed in the pending multi-curve interface;
`gray_test` already supports string group and stratum labels.

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
