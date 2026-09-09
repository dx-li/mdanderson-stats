# CDFLIB error-function and exponential reference audit

This audit covers four public F95 `biomath_mathlib_mod` procedures. Their
[Python interfaces](cdflib-error-exponential.md) are now implemented. It records
the original executable behavior and
independent mathematical checks so that a port can preserve valid behavior and
repair numerical failures deliberately.

| Native procedure | Defining operation |
|---|---|
| `erf(x)` | 2/√π times the integral of exp(-t²) from 0 to x |
| `erfc1(ind,x)` | erfc(x) when ind=0; exp(x²)erfc(x) for every nonzero integer ind |
| `esum(mu,x)` | exp(mu+x), with integer mu |
| `exparg(l)` | log(HUGE(double)) when l=0; log(TINY(double)) otherwise |

`tools/reference_cdflib_error_exponential.py` compiles the unchanged constants
and mathematical modules with gfortran, contraction disabled and bounds checking
enabled. The fixture retains the compiler, command, driver, source hashes and
archive hash. No source adaptation is made. All 134 recorded calls completed;
nonfinite answers are represented by explicit strings in strict JSON. This is
F95 evidence, not a verification of the distinct C/F77 helper implementations.

The error-function grid includes negative and positive coordinates, the native
piecewise boundaries 0.5, 4, 5.6 and 5.8, minimum subnormal arguments, tail
underflow, and positive coordinates up to 1e308. Both positive and negative
nonzero scaling flags are exercised. Exponential sums cover all sign combinations,
large cancellation, signed int32 endpoints, overflow and subnormal results.

## Independent checks and findings

The tests evaluate the defining error-function power series with 1,100-digit
Decimal arithmetic, computing π independently by Gauss–Legendre iteration. This
precision absorbs cancellation in the complement near x=28. For x≥30, a 40-term
alternating asymptotic expansion checks the scaled complement; its omitted
relative term is below 1e-70 on this interval. The unscaled complement there is
already below the representable float range. Negative arguments use the exact
reflection identities. Exponential sums use Decimal exp on the combined exponent.
Machine-range results are checked against the host double limits.

* `erfc1(0,26.7)`, `erfc1(0,27)` and `erfc1(0,27.2)` return zero although their
  mathematical values round to positive subnormals. The source tests x² against
  -log(TINY(double)) before evaluating the tail.
* `esum(1000,-999.5)` and `esum(-1000,999.5)` return NaN. Their answers are exp(0.5)
  and exp(-0.5). A sign-dependent branch multiplies separately overflowed and
  underflowed exponentials. The opposite cancellation directions work.
* `esum(710,-1)` returns infinity, although exp(709) is finite.
* `esum(-745,1)` returns 1.5e-323, while exp(-744) rounds to 1e-323. Separately
  rounding exp(-745) before multiplication loses substantial relative accuracy.
* Nonzero `exparg` returns approximately -708.3964185322641, the logarithm of the
  smallest **normal** double. Its header describes the most negative exponent
  with a nonzero result, but subnormal results remain possible below this value.
  A port must document this executable contract rather than silently changing
  the threshold used by other algorithms.

Scaled erfc at sufficiently negative arguments and exp(710) genuinely exceed the
finite double range; these are distinguished from intermediate overflow with a
finite mathematical answer. Ordinary reference agreement uses tolerances that
accommodate the original rational approximations. The identified failures have
specific assertions, not a broad tolerance that hides them.

See the [CDFLIB90 completion audit](cdflib90-completion.md) for the
complete library scope. This reference audit makes no performance claim.
