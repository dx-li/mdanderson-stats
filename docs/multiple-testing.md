# MULTI: implemented procedures and remaining scope

Catalog entry 50 includes a desktop program and an S library. This port currently
covers the desktop's nine adjustment/threshold procedures, two sharpened procedures,
and Schweder line fitting, plus the S library's Schweder bootstrap. It remains
**partial**: beta mixtures, nonparametric modeling, order-statistic diagnostics,
clustered p-value generation, and plotting helpers still need implementations and
validation. The catalog does not count this entry as complete.

## Adjustment procedures

```python
from mdanderson_stats import multiple_testing

result = multiple_testing([0.04, 0.001, 0.2, 0.03], "holm", alpha=0.05)
print(result.adjusted_pvalues)
print(result.reject)

rom = multiple_testing([0.02, 0.021, 0.8], "rom", alpha=0.05)
print(rom.critical_values)  # Rom returns thresholds, not adjusted p-values
```

For sorted p-values p(i), m tests and i=1,...,m, these are the source's algorithms:

| Source option | Python method | Rankwise quantity and accumulation |
|---|---|---|
| One-step Bonferroni | `bonferroni` | m*p(i) |
| One-step Sidak | `sidak` | 1-(1-p(i))^m |
| Step-down Holm | `holm` | Running maximum of (m-i+1)*p(i) |
| Step-down Sidak | `holm-sidak` | Running maximum of 1-(1-p(i))^(m-i+1) |
| Step-down Finner | `finner` | Running maximum of 1-(1-p(i))^(m/i) |
| Step-up Hochberg | `hochberg` | Reverse running minimum of (m-i+1)*p(i) |
| Step-up “Hommel” | `multi-hommel` | Reverse running minimum of m*H(m)*p(i)/i, H(m)=sum(1/j) |
| Step-up Simes | `simes` | Reverse running minimum of m*p(i)/i |
| Step-up Rom | `rom` | Alpha-dependent recursive critical values, followed by step-up rejection |

Adjusted p-values are capped at one. The last axis is one testing family;
leading axes are independent batches. Results retain the original input order,
and `order` records the sorted-rank indices. Input p-values must be finite and
in [0,1]; invalid values are not discarded. Alpha is in [0,1].

The S library's function named `bonferroni` actually runs sequential Holm;
its equivalent here is `method="holm"`. The Python `bonferroni` option follows
the desktop's one-step option. The S `sidak` function maps to `method="sidak"`.

The source's “Hommel” formula is the harmonic step-up adjustment, mathematically
the BY formula. It is not the usual Hommel closed-testing method. The ambiguous
`hommel` string is therefore rejected; use `multi-hommel` to request the archived
formula. Similarly, the source's step-up Simes adjusted values use the BH formula,
not a single global Simes p-value. Holm controls familywise error without a
dependence assumption; BH and BY target false discovery rate, with BY allowing
arbitrary dependence. These are different statistical guarantees.
See the [R statistics documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)
for these distinctions. The other procedures have their own dependence conditions;
their presence here does not imply that they can be interchanged in every design.

Rom's critical values are exposed by `rom_critical_values(m, alpha)`. They are in
ascending rank order. The implementation evaluates the original recurrence using
log-domain binomial terms to avoid coefficient overflow; it does not mistake those
cutoffs for adjusted p-values. A qualifying larger rank also rejects smaller ranks,
even when a smaller rank does not meet its own cutoff. The Sidak/Finner transforms
use `log1p` and `expm1` so p-values below machine epsilon do not round to zero.

## Schweder fitting and sharpened decisions

```python
from mdanderson_stats import schweder_fit, schweder_bootstrap, sharpened_testing

p = [0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 0.8]
fit = schweder_fit(p)
decisions = sharpened_testing(p, fit.null_estimate, "hochberg", alpha=0.05)
bootstrap = schweder_bootstrap([i / 41 for i in range(1, 41)], samples=100, rng=123)
```

The line uses unique p-values in descending order, x=1-p, and cumulative counts of
observations **at or above** p. Those counts include ties. The source divides them
by the number of distinct p-values while fitting an origin-constrained line, then
rescales the slope to a null-count estimate. The Python result exposes the plot
coordinates and the number of retained points. It uses the source's prediction
bound to trim the rightmost point until a line is accepted. At least four distinct
values are required; failure raises `SchwederFitError` rather than returning zero.

`prediction_variance` reproduces the S library's BETAVR value on its normalized
ordinate scale. It is not the variance of the null-count estimate. The bootstrap
reports sample variance of fitted null-count estimates instead. Like SWBOOT, it
requires the requested number of successful fits within at most twice that many
attempts. It returns original-data plot coordinates rather than the last resample's
coordinates. An explicit NumPy Generator or seed replaces the old global two-seed
generator; bitwise reproduction of the original random stream is not claimed.

The sharpened methods use the source's integer truncation and capping of the
supplied null-count estimate. A value below one causes all hypotheses to be rejected,
as in the original. A failed line fit must not be substituted with zero. These
estimate-dependent decisions do not automatically inherit a general FWER guarantee.

The original sharpened Holm routine returns after its second rejection threshold;
the subsequent denominator update described in its comments is unreachable.
`sharpened_testing(..., "holm")` preserves the executable's behavior. The Hochberg
variant follows the source's descending-rank thresholds. Neither returns fabricated
adjusted p-values.

## Validation and provenance

`tools/reference_multi.py` compiles the original desktop numerical routines after
removing only the interactive main program. The S SCHWED routine is compiled with
a distinct symbol name to compare its additional variance output. All numerical
routine bodies are retained. The source archive hash, compiler and compatibility
flags are recorded in `tests/fixtures/multi.json`.

The reference fixture contains 32 adjustment outputs, 35 Rom threshold vectors,
96 sharpened-testing outputs, and 30 desktop/S Schweder fits. It includes the
archive's 150-value example, ties, endpoints, fractional null-count estimates,
and different significance levels. An additional 50 bootstrap samples generated
by NumPy are fitted by the archived Fortran routine, allowing deterministic
comparison of bootstrap estimates and their mean/variance.

Independent checks include exhaustive closed Bonferroni testing for Holm,
hand calculations for step-up adjustments and small Rom families, Decimal
arithmetic for tiny Sidak p-values, exact Schweder line recovery, invalid/degenerate
inputs, and bounded bootstrap failure. Fixed-seed simulations check independent
global-null error rates and BH false discovery rate in a mixture of true and false
nulls. These validate the tested conditions; they do not prove all dependence
assumptions or replace a design-specific analysis.

The implementation uses NumPy broadcasting and prefix/suffix accumulations;
Rom's recurrence has quadratic work instead of repeatedly forming coefficients
by inner product loops. Original copyright and redistribution terms are retained
in [the notices](../THIRD_PARTY_NOTICES.md).

`uv run python tools/benchmark_multiplicity.py` reproduces the
[recorded benchmark](multiplicity-benchmark.json). On the recorded machine,
2,000 families of 100 p-values took about 0.0135 seconds in one batched Holm call
versus 0.0470 seconds in separate row calls, about 3.5x faster. A 2,000-test Rom
threshold vector took about 0.105 seconds and stayed finite and monotone. These
timings compare Python call patterns, not original Fortran throughput, and are
not CI performance thresholds.
