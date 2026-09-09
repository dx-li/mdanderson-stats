# Legacy DCDFLIB support primitives

This page covers ten legacy public contracts in `dcdflib_support`: two
machine-parameter lookups, polynomial evaluation and seven C translation helpers.
These mappings are audited separately from the F95 support modules.

```python
from mdanderson_stats import dcdflib_support as legacy

assert legacy.spmpar(1) == 2.0**-52
assert legacy.devlpl([1, -2, 3, 999], 3, [0, 2]).tolist() == [1, 9]
assert legacy.fifmod([-5, 5], [3, -3]).tolist() == [-2, 2]
```

## Machine parameters

`ipmpar(i)` accepts a Python integer index 1–10. It describes the archived int32,
IEEE binary32 and IEEE binary64 model, independent of the host's C `long` width
or Python's arbitrary-precision integers.

| Index | Meaning | Value |
|---|---|---|
| 1–3 | Integer radix, magnitude digits, largest magnitude | 2, 31, 2147483647 |
| 4 | Floating-point radix | 2 |
| 5–7 | Binary32 significand digits, minimum/maximum exponent | 24, -125, 128 |
| 8–10 | Binary64 significand digits, minimum/maximum exponent | 53, -1021, 1024 |

Exponent bounds use Fortran's significand interval `[1/radix, 1)`; the minimum is
one above NumPy's `minexp`. `spmpar(i)` accepts 1–3 and returns binary64 epsilon,
smallest **normal** positive value and largest finite value. Despite its name and
old header, both archived implementations were modified to return double-precision
constants. The smallest normal is not the smallest subnormal. Invalid or boolean
indices raise `ValueError` instead of reading outside the table or selecting a
fallback branch. These scalar lookups return Python `int`/`float` values.

## Polynomial and arithmetic helpers

`devlpl(a, n, x)` evaluates `a[0] + a[1]*x + ... + a[n-1]*x**(n-1)` using the
existing batched Horner implementation. Coefficients must be a finite 1D sequence;
`n` is a positive Python integer at most its length. Unused trailing coefficients
are validated but do not enter the polynomial. `x` may have any shape. An
unrepresentable result or Horner intermediate raises `ArithmeticError`.

The following operations broadcast their inputs and return owned immutable
NumPy arrays, including a zero-dimensional array for scalar input:

| Function | Contract |
|---|---|
| `fifdint(a)` | Truncate toward zero, return binary64; zero is positive as in the native integer-cast path |
| `fifidint(a)` | Truncate binary64 input toward zero, return checked int64 |
| `fifdmax1(a, b)` | Maximum, choosing `a` on equality |
| `fifdmin1(a, b)` | Minimum, choosing `b` on equality |
| `fifdsign(mag, sign)` | Negate `mag` if it is negative, then negate if `sign < 0` |
| `fifmod(a, b)` | Exact integer remainder with the dividend's sign |

The float operations require finite binary64 inputs. Min/max preserve the chosen
operand's zero sign. `fifdsign` deliberately uses the source's comparisons: a
negative-zero sign argument does not make a positive magnitude negative, and
negative-zero magnitude survives until a strictly negative sign argument negates
it. This differs from `copysign`.

`fifdint` supports every finite binary64 value; it avoids the source's undefined
out-of-range cast through C `long`. `fifidint` explicitly uses int64 on all hosts
and rejects values outside `[-2**63, 2**63)`. Its input remains binary64, so an
integer supplied above exact binary64 precision is rounded by conversion first.
Use integer inputs directly with `fifmod` when exact large integers are required.

`fifmod` accepts integer arrays/scalars representable in int64. It rejects floats,
booleans, out-of-range unsigned/object values and zero divisors. It never converts
integers through floating point. For example, `fifmod(-5, 3)` is -2 whereas Python
`-5 % 3` is 1. The mathematical remainder for `(-2**63, -1)` is returned as zero
without the overflowing native quotient. Empty typed integer arrays are supported.

`ftnstop(message=None)` raises `RuntimeError`, preserving a supplied string as its
message or using `"Fortran STOP"`. It does not write to stderr or terminate the
Python process. Invalid message types raise `ValueError`.

## Evidence and remaining scope

The [reference generator](../tools/reference_dcdflib_support.py) verifies the
archive SHA256 and compiles unchanged sources: 71 C cases and 25 F77 cases cover
all ten mappings. The C driver requires 64-bit `long` and probes only defined
integer conversions/remainders. Compiler options, source/driver hashes, outputs
and exit codes are retained in the [fixture](../tests/fixtures/dcdflib_support.json).

Tests compare native results bit-for-bit, including signed zero, and independently
check the machine model, polynomial prefix, broadcasting, ownership and validation.
More than 1,000 integer pairs satisfy exact quotient/remainder identities across
the full int64 range. Undefined native cases are tested against mathematical
invariants rather than treated as numerical references.

The [benchmark](dcdflib-support-benchmark.json) compares one vectorized remainder
call with repeated scalar calls to the same Python API. It is not a speedup claim
over native C or Fortran. Integer arithmetic uses NumPy batch operations, with no
per-element Python loop; polynomial evaluation loops over coefficients while
processing all evaluation points together.

These ten primitives and the [31 mathematical helpers](dcdflib-math.md) reconcile
41 legacy C support names. The [normal/t quantile helpers](dcdflib-quantile-helpers.md)
and [incomplete-gamma inverse](dcdflib-gamma-inverse.md) bring the total to 45 of 49.
The [legacy root finders](dcdflib-root.md) complete all 49 mappings.
CDFLIB90 stays partial and the full catalog conversion remains in progress.
