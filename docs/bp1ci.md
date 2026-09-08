# BP1CI compatibility and generalized intervals

```python
from mdanderson_stats import bp1ci, bp1ci_binomial_interval

result = bp1ci(12, 18)  # successes and failures, 95 percent confidence
print(result.report())
result = bp1ci(12, 30, entry="trials", confidence_percent=99)
result = bp1ci(10000, distribution="poisson", confidence_percent=99, exposure=100)
low, high = bp1ci_binomial_interval([0.5, 1.25], [2.5, 4])
```

Catalog entry 74 is implemented, with the differences below. Standard integer
Clopper–Pearson and corrected Garwood intervals remain available separately as
binomial_interval and poisson_interval. The BP1CI-named Poisson APIs retain the
original lower-bound defect for reproducibility and label it in the report.

## Input and output conventions

`bp1ci` defaults to binomial input, with successes followed by failures and a
95% confidence level. entry="trials" interprets the second input as the total.
Each entered binomial number must lie in [0,1e10], following do_binomial_ci; the
successes-plus-failures total can therefore reach 2e10. Poisson accepts events
at least 1e-6, including fractional events. A second count is not accepted for
Poisson. The confidence_percent range is [0.1,99.9999], inclusive, matching setup.
The lower-level interval APIs use confidence fractions in (0,1).

Arguments broadcast across independent cases. The result contains distribution,
confidence_percent, events, failures/trials where applicable, exposure, estimate,
lower and upper. For binomial data the estimate is successes/trials. Poisson
exposure must be positive and scales bounds and the event-rate estimate, implementing
the guide's manual scaling step. Nonunit exposure is rejected for binomial data.

report(digits=6) returns a tab-separated table with one row per broadcast case,
flattened in C order. digits selects 1–17 significant digits. It includes the
observations, bounds and estimate, and explicitly labels the original Poisson
formula. No global print settings, interactive prompts or output files are changed.
Repeated calls with explicit settings replace the original setup/menu loop.

## Fractional binomial formula and source defects

For successes k, total n and tail probability t=(1-confidence)/2, the lower
bound is BetaQuantile(t; k,n-k+1) for k>0 and zero for k=0. The upper bound is
BetaSurvivalQuantile(t; k+1,n-k) for k<n and one for k=n. These extend the
source's real-valued beta-tail inversion. They do not establish frequentist
coverage for fractional observations. Ordinary binomial coverage tests apply
to integer counts.

`bp1ci_binomial_interval` requires finite 0<=k<=n<2**53. It returns [0,1] for
n=0, the no-information case. The higher-level bp1ci wrapper rejects zero totals,
because a reported success-rate estimate would be undefined. It also rejects
successes above total trials, which the original input screen fails to check.

For 0<k<1 the original calls cdf_binomial with s=k-1<0, which fails input
validation and leaves the lower output undefined; one native run prints zero.
The port evaluates the well-defined beta quantile. Closed-form tests use
Beta(k,1), whose quantile is t**(1/k), including fractional k below one.

A second discrepancy occurs for 1.25 successes, 2.75 failures and 0.1% confidence:
the native program prints lower=0.2151. The beta-tail solution is approximately
0.214956143932777. An independent 60-digit Decimal integration of the binomial
series for the beta density confirms its CDF is .4995 within 1e-15. This test
uses B(5/4,15/4)=77*pi*sqrt(2)/2048 from gamma recurrence/reflection, not SciPy's
incomplete beta implementation. The discrepant source value is recorded as such,
not used as a valid numerical oracle or hidden by a wider tolerance.

## Validation, performance and source audit

The original 24 native reference intervals remain checked. The additional
`tools/bp1ci_extended_reference.py` records 36 unmodified-executable cases spanning
fractional counts, both binomial entry modes, .1/95/99.9999 percentage levels,
and counts up to 1e10. Thirty-five match at printed precision; the discrepancy
above has a separate independent test. Four-decimal binomial printing cannot
establish tail accuracy for very narrow/extreme intervals. Tests additionally
check forward fractional beta tails, reflection symmetry, broadcasting, integer
agreement, closed-form fractional boundaries, input limits, exposure scaling and
report fields. No claim is made that matching rounded native output proves
accuracy at arbitrary larger magnitudes.

A local workload of 5,000 fractional intervals (successes .25–99.75, total 100.5)
produced matching vector and scalar results. The vector call took about .0091s
versus .138s for a Python loop, approximately 15x faster in this environment.
This is a workload-specific measurement, not a portable timing guarantee.

| Original component | Python coverage |
| --- | --- |
| setup / menu | bp1ci distribution, percentage and entry arguments; repeated calls |
| do_binomial_ci | Both entry modes, bounds on entered numbers, estimates and report |
| binomialci | Fractional beta inversion; integer exact alternative; corrected undefined cases |
| do_poisson_ci / poissonci | Original tail formula, minimum positive input and report; corrected exact alternative |
| Guide's unit conversion | Poisson exposure scaling |
| banner / terminal helpers / distribution libraries | Attribution, explicit Python validation/I/O, NumPy/SciPy inverse kernels |

The package does not reproduce terminal pauses, malformed-input retry loops,
undefined outputs, original numerical-library failure fallbacks, or byte-identical
screen formatting. These are documented substitutions, not unimplemented
statistical features. Supporting distribution libraries are used through their
BP1CI formulas rather than exported as a generic port of every Fortran routine.
