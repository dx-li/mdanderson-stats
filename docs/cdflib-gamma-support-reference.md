# CDFLIB gamma and digamma support audit

Seven additional public F95 mathematical support procedures have native reference
evidence. Their [Python interfaces](cdflib-gamma-support.md) are now implemented.
This audit preserves executable
behavior separately from mathematical expectations and does not change CDFLIB90's
partial completion status.

| Procedure | Source contract |
|---|---|
| `alngam(x)` | Real log Γ(x); header does not explicitly restrict x to positive values |
| `gamln(a)` | log Γ(a), positive a; returns −1 for nonpositive a |
| `log_gamma(a)` | log Γ(a), positive a; no equivalent domain guard |
| `gamln1(a)` | log Γ(1+a), −0.2≤a≤1.25 |
| `gam1(a)` | 1/Γ(1+a)−1, −0.5≤a≤1.5 |
| `gamma(a)` | Real gamma, including negative nonintegers; zero on computation failure |
| `psi(x)` | Real digamma, including negative nonintegers; zero on computation failure |

These contracts must not be collapsed into a single positive-domain wrapper.
For example, `alngam(-1.5)` computes a valid real logarithm because Γ(-1.5)>0,
whereas `log_gamma(-1.5)` produces NaN outside its documented domain. Negative
intervals where gamma is negative have no real logarithm. Nonpositive integer
arguments are poles of gamma and digamma; their native zero sentinels are not
mathematical values.

## Native provenance and independent validation

`tools/reference_cdflib_gamma_support.py` compiles the unchanged constants and
mathematical modules using bounds checking and disabled contraction. The fixture
contains archive/source hashes, compiler identity, command and complete driver.
There are no source adaptations. Of 237 calls, 236 completed and one reached its
three-second timeout. This records F95 contracts; similarly named C/F77 procedures
have not been independently validated by this audit.

The grid covers minimum subnormal inputs, tiny arguments, neighborhoods of gamma's
unit values, approximation switches, factorial and half-integer values, finite
output limits, negative nonintegers, poles and the native integer cutoff.

Independent oracles use 100-digit Decimal arithmetic, exact rational Bernoulli
numbers and recurrence to arguments of at least 50 before 32-term Stirling
expansions. The first omitted absolute term is below 1e-65 for both log gamma and
digamma in that region. Reflection at half-integers avoids trigonometric error.
Near-zero `gamln1` and `gam1` use their first-order Euler-constant terms when
|a|<1e-30; the omitted relative error is O(|a|), far below float64 resolution.
This also avoids losing a when forming 1+a in the oracle. Factorial, harmonic-sum
and half-integer identities check the oracle independently.

## Findings that affect a port

* `alngam(5e-324)` and `alngam(1e-309)` return infinity even though their log gamma
  values are finite. The source forms a reciprocal/product before taking its log.
* Near x=1, `alngam`'s approximation produces positive values around 4e-15,
  losing the zero at one and the negative sign immediately above it. It also
  misses the exact zero at two. The other two log-gamma procedures preserve
  these roots much better.
* `alngam(-175.5)` has about 6.3e-6 absolute error from taking the logarithm of a
  rounded subnormal product. At -177.5 it returns negative infinity, although
  the real log gamma is approximately -744.1314465173804.
* `alngam(-1e100)` does not terminate within three seconds: its recurrence adds
  one to a floating-point value for which that increment cannot change the value.
  This input is a pole; a future implementation must reject it before iteration.
* `gamma(-172.5)`, `gamma(-175.5)` and `gamma(-177.5)` return zero despite
  representable, nonzero mathematical results. Their positive-gamma intermediate
  exceeds the source's exponential limit. Other zero results represent genuine
  output underflow or the source's pole/overflow sentinels; tests distinguish them.
* `psi(-2147483647.5)` returns zero because of the default-integer machine bound,
  even though the digamma value is finite and greater than 20. At positive 1e10,
  the same bound switches early to log(x), dropping the leading −1/(2x) correction.
* Tiny signed digamma inputs can genuinely overflow. Likewise log gamma at 1e306
  or 1e308 exceeds finite float64; these are not intermediate-overflow defects.

The tests isolate these discrepancies explicitly rather than broadening ordinary
reference tolerances. The Python implementation retains valid negative-domain
behavior, preserves small remainders and tails, and distinguishes invalid arguments
from output overflow. Other gamma/beta ratios, combinatorial logarithms, constants
and remaining support interfaces are still open.
