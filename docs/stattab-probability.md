# STATTAB discrete probability terms

Three vectorized Python functions implement STATTAB's individual binomial,
negative-binomial and Poisson probabilities. These are discrete masses, while
the existing distribution CDFs retain their continuous-count extensions.

```python
from mdanderson_stats import (
    stattab_binomial_term,
    stattab_negative_binomial_term,
    stattab_poisson_term,
)

stattab_binomial_term([0.5, 2.5], 4.75, p=0.5)  # masses at 0 and 2 of 4 trials
stattab_negative_binomial_term(2, 3, p=0.5)  # 0.1875
stattab_poisson_term(0.5, 3)  # exp(-3)
stattab_poisson_term([0, 1], 0)  # [1, 0]
```

## Contract

All arguments broadcast through NumPy. Counts must be finite and nonnegative;
each count is truncated toward zero before the discrete calculation. Binomial
successes must not exceed trials even before truncation. The Poisson mean is
finite/nonnegative and is not truncated. Results are independently owned,
immutable float64 arrays; empty batches are supported.

Binomial and negative-binomial calls accept `p`, the complementary chance `q`,
or both. At least one is required. Values must lie in [0,1], and an explicit pair
must sum to one within eight machine epsilons. The smaller supplied chance is
retained and the larger reconstructed, matching the existing CDFLIB pair rule.
This preserves probabilities whose complements round to one.

Zero binomial trials give unit mass at zero successes. Zero required
negative-binomial successes give unit mass at zero failures, including p=0.
For a positive required-success count and p=0, all finite failure counts have
zero mass. A zero Poisson mean gives unit mass at zero events. These boundary
conventions are explicit extensions or repairs of the native application.

These standalone terms accept finite counts beyond the archived application
search bounds. Counts above 2**53 are interpreted as the exact integers represented
by their float64 input, without claiming that arbitrary adjacent integers remain
distinguishable. Invalid inputs raise `ValueError`; a failed numerical invariant
raises `ArithmeticError`. True probability underflow returns zero. Correct rounding
at every float64 input is not promised.

## Numerical method

Let B(a,b,p,q) = p**a q**b / beta(a,b), the existing beta scaling factor.
For positive interior integer counts:

- Binomial mass is B(s,n-s,p,q) * (1/s + 1/(n-s)).
- Negative-binomial mass is B(s,f,p,q) / f.
- Poisson mass is exp(-mean) * mean**n / Gamma(n+1).

The beta implementation reuses the existing stable factor decomposition, applying
the probability normalization before final exponentiation. The binomial divisor
is computed as min(s,n-s)/(1 + min/max), avoiding overflowing products and sums.
Endpoint masses use logarithmic powers and the smaller complementary chance.
Poisson calculations use a log probability; large counts share the gamma factor's
Stirling/deviance calculation, preserving the displacement near the mean instead
of subtracting large log-factorials. Extracting that internal calculation leaves
the public `rcomp` arithmetic unchanged.

There is no subtraction of adjacent CDFs, per-element Python numerical loop or
new dependency. Small probabilities therefore do not disappear merely because
two rounded CDFs are equal.

## Native repairs and validation

The [STATTAB audit](stattab-research.md) preserves native sessions. For counts
between zero and one, the source prints the continuous CDF as the binomial/Poisson
individual term. Python consistently uses the truncated count. At count 0.5 and
mean 3, the correct Poisson mass is exp(-3), not the native value near 0.111610.
For negative-binomial F=2, S=3, P=0.5, Python returns 0.1875 instead of the zero
printed after a native stale-status failure. Ordinary native terms above one
remain reference checks where they agree with their defining probabilities.

Tests use exact integer combinatorics, independent Decimal probabilities through
800-digit arithmetic, probability sums and recurrences. They cover fractional
counts, exact endpoints, explicit tiny complements, huge count centers, the
smallest subnormal probability bins, mixed broadcasting, ownership, empty batches
and input errors. Existing gamma-factor tests check the shared internal extraction.
The high-precision tests allow the larger of 3e-13 relative error and one ULP.
The smallest-probability-bin grid additionally requires one-ULP agreement. These
sampled checks are not a uniform error proof.

[Benchmark results](stattab-probability-benchmark.json) compare a broadcast call
with repeated calls to the same Python API, verify identical arrays and report
three-run medians. They measure Python batching, not speed relative to Fortran.

The [result](stattab-results.md), [session](stattab-sessions.md) and
[console](stattab-console.md) layers implement the application workflow around
these terms. STATTAB remains **partial** pending its final source/version
reconciliation, tracked in the [application audit](stattab-research.md).
