# SURVAN Kaplan–Meier and Simon–Lee confidence calculations

`survan_km` returns the native life-table quantities: time, survival, risk count,
death count, censor count, Greenwood standard error, and Simon–Lee pointwise
confidence limits. It reuses the package's pooled-tie Kaplan–Meier estimator.

```python
from mdanderson_stats import survan_km

curve = survan_km([1, 2, 3, 4, 5], [1, 0, 1, 1, 0], confidence=0.95)
median = curve.quantiles(0.5)
print(float(median.time), int(median.flag))
```

Time is nonnegative; events are 1 and right censors 0. Censors remain at risk for
events at the same time. All distinct observation times, including censor-only
times, appear in the table. An initial (time=0, survival=1) row is inserted when
the first observation is later than zero. Arrays are immutable. Confidence levels
range from 1e-6 through 1−1e-12. Input limits follow `exploratory_survival`.

## Pointwise confidence limits

For survival estimate S, current risk count n, and standard-normal two-sided
critical value z, SURVAN solves

(n + z² S) p² − (2n + z²) S p + n S² = 0.

The two roots are the confidence limits. This is the original Simon–Lee
risk-based interval, **not** the existing IPDfromKM Greenwood log interval.
It can change at censor-only times even when S is unchanged. At S=0 or 1,
both bounds equal S; Greenwood standard error is zero at terminal S=0,
following SURVAN. These are pointwise intervals, not a simultaneous band.

The Python calculation factors the discriminant as S² z²[z² + 4n(1−S)] and
obtains the lower root using the product of roots. This avoids subtracting
nearly equal quadratic terms and the original generic solver's small-coefficient
cutoffs. The shared normal quantile replaces the native approximation.

## Quantiles and interval flags

`curve.quantiles(probabilities)` uses **survival probabilities**, so 0.25 denotes
a time with 25% surviving. The source's point estimate linearly interpolates
between adjacent table rows bracketing a strict crossing below the requested
probability. A censor time can therefore supply the left interpolation endpoint.
An exact plateau uses its last row before the next drop. If survival never falls
strictly below the probability, the result is missing (including equality at the
terminal plateau). These conventions differ from first-crossing and midpoint
Kaplan–Meier quantiles elsewhere in the package.

At each table row except the last, the native time-interval statistic is

(n − deaths − censored) (S − q)² / [S q(1−q)].

The routine records the first row whose statistic is at most z² as its lower
limit, then the first subsequent row above z² as its upper limit. A later
re-entry below z² marks the candidate interval unreliable. The Python comparison
uses logarithms to avoid overflow for probabilities near zero or one.

| Flag | Meaning |
|---|---|
| 0 | Quantile exists, but no interval was found |
| 1 | Lower limit found; upper limit undetermined |
| 2 | Both limits found |
| 3 | Re-entry makes the candidate interval unreliable |
| 4 | No strict survival crossing |

Unavailable estimates or limits are NaN rather than native zero sentinels.
Flag 3 retains the candidate bounds for inspection; they should not be treated
as a reliable connected interval. Up to 1,000 probabilities and 20 million
probability × table-row calculations are supported, preserving input shape.

Two zero-survival edge cases have explicit handling: a quantile crossed by events
at time zero is returned as zero, whereas native CALQV lacks its left bracket;
and S=0 is outside the time-interval region for q>0, avoiding native 0/0.

## Validation

Original KMC, FSROUT, KMCI, CALQV and KMCIQT routines were compiled unchanged,
along with the original normal approximation, polynomial evaluator and quadratic
solver. A small harness supplied observation iteration, file-unit allocation and
in-memory reversal; the original FSROUT performed survival/variance accumulation.
The complete 33-row manual-example table and nine quantiles were compared against
Python. The largest absolute survival-limit difference was below 7.2e-7, consistent
with native single precision. All nine interval flags and available bounds match.

Focused checks also cover censor plateaus, all-censored data, terminal zero
survival, zero-time events, probabilities as small as 1e-300, and interval re-entry.
Source hashes and original terms are linked from [SURVAN coverage](survan.md).
All advertised calculation families are mapped in the SURVAN coverage audit.
