# CUMINC cumulative-incidence curves

Catalog entry 39 is **partial**. One-cause curve estimation and the original
Aalen variance convention are implemented. Multi-group/stratified Gray tests,
confidence intervals, full summaries and plots remain pending.

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
String cause/group labels will be addressed in the pending multi-curve interface.

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

The archived S wrapper computes group-test p-values using a normal tail of a
quadratic statistic and counts distinct event times in some summaries. Those
interface behaviors need explicit review during the remaining group-test and
reporting conversion; they are not implemented or implicitly endorsed by this
curve-only increment.
