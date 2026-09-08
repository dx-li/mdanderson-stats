# ONESAMPLE tests, confidence intervals and reports

Catalog entry 22 is implemented. All four calculations, both binomial entry modes,
input validation and readable results are available as vectorized Python calls.
Source: ONESAMPLE_V1.tar.gz, one_sample_1.0, from the catalog download.

```python
from mdanderson_stats import binomial_test, poisson_test

test = binomial_test(12, 30, null_probability=0.3)
print(test.estimate, test.p_less, test.p_greater)
test = poisson_test(10, null_rate=2, exposure=5)
```

For observed count k, p_less is P(X<=k) under the null and tests against a smaller
probability/rate. p_greater is P(X>=k), for a larger probability/rate. Both include
the observed outcome, so their sum generally exceeds one. No two-sided convention
is inferred. The original internal names p_lo/p_hi and some comments are confusing;
these names follow the hypotheses printed by write_output_mod.

Inputs broadcast over independent tests. Counts must be nonnegative integers;
binomial requires positive trials and successes no greater than trials, and the
null probability lies in [0,1]. Its estimate is successes/trials. Poisson allows
zero events, requires a nonnegative null rate and positive exposure, and uses the
null mean rate*exposure. Its estimate is events/exposure. Nonfinite inputs and
unrepresentable means/rates raise. These are known-exposure tests, not models for
uncertain exposure or estimated offsets.

The result contains estimate, null_value (probability or rate), p_less, p_greater,
and legacy_cutoffs. Incomplete beta and gamma functions evaluate complementary
tails directly; the implementation avoids subtracting a nearly-one CDF to recover
a small tail. Boundary cases are explicit and produce valid probabilities at null
probability 0/1 or null mean zero.

## Extreme binomial tails

SciPy's Boost incomplete-beta backend can lose relative accuracy or return zero
prematurely for extremely small binomial tails. The package now reevaluates tails
below 1e-250 with SciPy's independent Cephes `bdtr`/`bdtrc` routines when trials
fit a signed 32-bit integer, for greater tails at p<=0.5 and less tails at p>=0.5.
The opposite orientations retain the original evaluation: Cephes would form 1-p
and can lose precision amplified by large trial counts. The ordinary vectorized
path and larger-count domain remain available; this fallback makes no additional accuracy guarantee beyond the
Cephes count limit. Legacy forced-zero cutoffs are applied afterwards.

For example, P(Binomial(200,0.02)>=190) is about 2.881774153016122e-307, whereas
the local SciPy 1.18.1 incomplete-beta call returned about 2.218453989e-307.
At 191 events it returned zero for a representable probability near 3.0788e-310.
Regression tests sum exact integer binomial coefficients and probabilities using
100-digit Decimal arithmetic, covering both tails, broadcasting, true underflow,
subnormal results, compatibility cutoffs and the fallback's integer boundary.
Subnormal comparisons allow two units of the smallest representable float because
relative accuracy is limited by their spacing.

## Original cutoffs and validation

Default legacy_cutoffs=False evaluates the mathematical tails. True reproduces
the calculation module's shortcuts: the binomial upper tail becomes zero when
p0<=1e-10 for positive successes, and its lower tail becomes zero when
p0>=1-1e-10 with at least one failure. The Poisson upper tail becomes zero for
positive events when rate*exposure<=1e-10. Trivial all/zero-success boundaries
retain their source values. These cutoffs can discard real nonzero probabilities;
they are compatibility settings, not accuracy improvements.

`tools/reference_onesample.py` builds the archived source and directly calls
one_sample_calc_mod, recording binary64 results for 36 cases at three confidence
levels, including boundaries, tail cutoffs and a count of 10,000. The first build
exposed undefined success-status outputs in cdf_aux_mod under the current compiler.
The private reference build therefore initializes optional status=0 at entry to
add_to_one, check_complements, dbl_in_range, int_in_range and validate_parameters.
The archive is preserved, and this patch changes status initialization only;
statistical algorithms and source cutoffs remain unchanged. The fixture records
the archive/source hashes, compiler and exact patch description. No patched or
original Fortran is included in the Python package.

Tests compare both test tails and the existing confidence intervals against these
reference calls. Independent binomial mass sums and Poisson recurrence sums verify
tail orientation and inclusive boundaries. Further checks cover tiny nonzero
tails, explicit cutoffs, broadcasting, exposure and input errors. The original
Poisson confidence routine uses the correct Garwood lower shape, unlike BP1CI's
historical lower-bound formula; ONESAMPLE validation uses poisson_interval.

## Input and report workflow

```python
from mdanderson_stats import one_sample

result = one_sample("binomial_confidence", [0, 12, 30], 0.95, second=30, entry="trials")
print(result.report())
result.write_report("intervals.tsv", digits=12)

result = one_sample("binomial_test", 12, 0.3, second=18)  # second is failures
result = one_sample("poisson_confidence", 10, 0.95, exposure=5)
result = one_sample("poisson_test", 10, 2, exposure=5)  # null rate, not null mean
```

The second positional value after events is a confidence fraction for confidence
calculations, or a null probability/rate for tests. Named calculations replace
menu choices 1–4. Calls replace the original prompt/retry loop; ending the caller's
session replaces menu choice 0. Results contain the inputs, estimate, interval or
both inclusive tails, and the compatibility setting.

The wrapper preserves the interface's entered-count limit of 1e9, confidence and
null-probability range [1e-10, 1-1e-10], and exposure/null-rate range [1e-10, 1e9].
Failures entry can yield a total of 2e9. Fractional counts, empty binomial totals
and successes exceeding trials raise rather than undergoing the source's silent
integer conversion or reaching undefined calculations. The lower-level APIs
remain available for their broader mathematical domains. Legacy tail cutoffs are
opt-in for tests and rejected as irrelevant for confidence calculations.

Reports echo counts, trials or exposure, confidence or null value, estimates and
results. Broadcast cases appear in C order as TSV rows. Precision is configurable
from 1 to 17 significant digits, avoiding the original fixed-width fields' overflow
asterisks. write_report explicitly replaces a UTF-8 file and propagates I/O errors.
The original main program prints to the terminal; optional unit routing exists in
a general printing helper but is not exposed by its menu. Python file output is
an explicit convenience rather than a reproduction of an additional source menu.

The coverage audit follows one_sample.f90, get_numbers_mod, write_output_mod and
the four calculation routines. Tests exercise all operations, both entry modes,
broadcast reports, file replacement, invalid precision/I/O, interface limits and
agreement with the same native reference fixture. The terminal prompt syntax and
fixed-width whitespace are intentionally replaced by the API and TSV format.
