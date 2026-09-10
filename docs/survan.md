# SURVAN survival comparisons

`survan_group_test` implements the multi-group log-rank and Gehan–Breslow
comparisons from SURVAN, optionally stratified. These are independent of the
existing two-arm Efron Cox score test, which has different tied-event variance.

```python
from mdanderson_stats import survan_group_test

result = survan_group_test(
    time=[1, 2, 1, 2],
    event=[1, 1, 0, 1],
    group=["treatment", "treatment", "control", "control"],
)
assert abs(result.statistic - 1) < 1e-12
assert result.degrees_of_freedom == 1
```

Use `method="gehan_breslow"` for risk-count weighting. Pass a matching `strata=`
vector for within-stratum risk sets and summed scores/covariances. Time must be
finite and nonnegative, event must be 1 for a death or 0 for right censoring,
and group/stratum labels must be homogeneous integer or string vectors. Labels
are sorted in the result. No missing values, delayed entry or individual case
weights are supported. Exact time ties are grouped; tied censors remain at risk
for failures at the same time. Zero times are supported with explicit event
indicators, avoiding the native signed-time encoding's zero ambiguity.

For a risk set with n individuals, d deaths, group proportions p and group death
counts d_g, the score increment is w(d_g − d p). Its covariance increment is
w² d(n−d)/(n−1) [diag(p) − p pᵀ], with zero covariance when n=1.
The log-rank weight is 1; the Gehan–Breslow weight is n **within the stratum**.
This is the archive's hypergeometric variance convention.

The result contains patient counts, unweighted observed/expected deaths, weighted
scores, covariance, test statistic, chi-square upper-tail p-value and degrees of
freedom, plus scores/covariances for each stratum. Numerical arrays are immutable.
The statistic uses the estimable covariance eigenspace, with df equal to its
resolved rank. This handles disconnected groups without inverting a singular
matrix. No comparison information raises an error; negative covariance eigenvalues
beyond rounding tolerance or unresolved score components also raise errors.
Small eigenvalues below 64 × machine epsilon × group count × largest eigenvalue
are treated as unresolved. The p-value is asymptotic, not an exact small-sample test.

Input limits are 1,000,000 records, 2..100 groups, records × groups ≤20,000,000
and strata × groups² ≤1,000,000. Time/group risk tables use vectorized cumulative
counts, and covariance accumulation uses matrix multiplication. Strata are
partitioned after a single sort, avoiding repeated full-data filtering.

## Validation and coverage

The original `lrc` and `gbc` Fortran routines were compiled unchanged using
gfortran. Only their observation iterator, initialization helpers and packed
matrix index helper were supplied by a small reference harness. Records were
ordered by decreasing time, censors before events at an equal time. A common
+1 shift allowed zero-time censors in generated data to use native signed times
without altering ordering or ties.

Fixtures compare scores and packed covariance values on the 40-patient manual
example and a generated 120-patient, three-group, three-stratum tied dataset.
Each stratum was evaluated by the original single-stratum routines and combined
using the native stratified summation convention. Native single precision is
allowed relative tolerances of 2e-5 for scores and 2e-6 for covariance.
The manual example reproduces the displayed statistics 3.123 and 2.651.
The manual prints Gehan–Breslow scores −114/+114, whereas the executable returns
+114/−114 in the example's group order; this port follows the executable. The
sign reversal does not change the omnibus statistic.

Additional checks cover an analytic tied-censor example, independent-stratum
addition, group relabeling, absence of information and disconnected groups.
No claim is made that the entire SURVAN application is implemented: its native
multivariable proportional-hazards fits, descriptive summaries and
interactive reporting remain to be ported. Kaplan–Meier and Simon–Lee confidence
calculations are now available through [survan_km](survan-km.md), and
[logistic regression](survan-logistic.md) provides multivariable binary fits.

Sources: [MD Anderson SURVAN](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/59)
and [source archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/SURVAN/SURVAN_V1.tar.gz).
Hashes are in `survan-sources.json`; original programs remain in ignored research
storage. Original terms are preserved in the project's third-party notices.
