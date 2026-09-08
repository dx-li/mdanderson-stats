# ONESAMPLE one-sided tests

Catalog entry 22 is partial. Implemented: binomial and Poisson inclusive one-sided
tests; the source confidence formulas are also verified against the existing
binomial_interval and poisson_interval APIs. Input-mode and report workflows remain
pending. Source: ONESAMPLE_V1.tar.gz, one_sample_1.0, from the catalog download.

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

Numerical calculation coverage does not establish completion of the interactive
entry modes, output-file routing and formatted results in one_sample.f90,
get_numbers_mod and write_output_mod. These remain tracked as pending for this entry.
