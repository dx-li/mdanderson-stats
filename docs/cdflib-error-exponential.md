# CDFLIB error functions and exponential helpers

Four more F95 mathematical support procedures have vectorized Python interfaces:

| Function | Result |
|---|---|
| `erf(x)` | Real error function |
| `erfc1(ind,x)` | erfc(x) for ind=0; exp(x²)erfc(x) for any nonzero ind |
| `esum(mu,x)` | exp(mu+x) |
| `exparg(l)` | log(max float64) for l=0; log(min **normal** float64) otherwise |

Coordinates must be finite. Integer parameters accept integral values in the
native signed int32 range, including array inputs. Arguments broadcast using
NumPy rules. Results are owned immutable float64 arrays, including scalar-shaped
results and empty batches. Invalid domains raise `ValueError`; mathematically
unrepresentable finite-range results raise `ArithmeticError` for the whole call.

```python
from mdanderson_stats import erfc1, esum, exparg

small_tails = erfc1(0, [26.7, 27.0, 27.2])
scaled_tails = erfc1(1, [30.0, 100.0, 1e308])
finite_cancellation = esum([1000, -1000], [-999.5, 999.5])
normal_range = exparg([0, 1])
```

`erf` and scaled `erfc1` use SciPy's compiled array kernels. Unscaled erfc uses
the compiled kernel through x=26. Above that point, both the original F95 routine
and the installed SciPy unscaled kernel can discard subnormal tails. The port
instead computes h=exp(-x²/2) and `(h*erfcx(x))*h`. This avoids premature cutoff
and keeps intermediates normal until the final multiplication in the subnormal
output region. The scaled function remains useful even at x=1e308, where forming
exp(x²) directly would overflow. Sufficiently negative scaled arguments genuinely
overflow and raise an error.

`esum` combines mu+x before exponentiation. This repairs the original NaNs for
exp(1000-999.5) and exp(-1000+999.5), intermediate overflow for exp(710-1), and
double-rounding for exp(-745+1). Addition and exponentiation use float64;
this is not an arbitrary-precision or correctly-rounded transcendental API.

`exparg` deliberately retains the **executable** normal-range threshold. The
original comment incorrectly describes the most negative argument yielding a
nonzero exponential. Subnormals remain possible below the returned threshold;
the helper does not attempt to report the precise exp rounding-to-zero boundary.

## Evidence and performance

The [reference audit](cdflib-error-exponential-reference.md) records 134 unchanged
F95 executions and their source/archive provenance. The implementation tests use
independent 1,100-digit defining-function calculations for error functions and
Decimal exponentials, rather than accepting the identified native failures.
Additional checks cover the tail switch, final subnormal rounding, sign/flag
semantics, integer limits, overflow, broadcasting, empty arrays and immutability.
No existing distribution kernel is changed.

[Benchmarks](cdflib-error-exponential-benchmark.json) compare batch calls with
repeated scalar calls to the same API at 64 and 256 coordinates. Each timing
checks identical outputs. They include regular, scaled and subnormal erfc and
large-cancellation exponential sums. These measure batching benefits, not a
speedup over the original Fortran implementation. No dependency was added.

CDFLIB90 remains partial. The later [gamma/digamma](cdflib-gamma-support.md) and
[gamma-ratio](cdflib-gamma-ratios.md) ports extend mathematical support. The rest
of the mathematical module, imported constants and other support contracts remain
open.
