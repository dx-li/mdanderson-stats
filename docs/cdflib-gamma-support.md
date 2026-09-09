# CDFLIB gamma and digamma helpers

Seven F95 mathematical support procedures now have vectorized Python interfaces.
Their distinct domains are preserved:

| Function | Result | Domain |
|---|---|---|
| `alngam(x)` | Real log Γ(x) | Finite x with Γ(x)>0, including valid negative intervals |
| `gamln(a)` | log Γ(a) | Positive finite a |
| `log_gamma(a)` | log Γ(a) | Positive finite a |
| `gamln1(a)` | log Γ(1+a) | −0.2≤a≤1.25 |
| `gam1(a)` | 1/Γ(1+a)−1 | −0.5≤a≤1.5 |
| `gamma(a)` | Γ(a) | Finite real a excluding nonpositive integer poles |
| `psi(x)` | Digamma, Γ′(x)/Γ(x) | Finite real x excluding nonpositive integer poles |

Scalar or array inputs preserve their shape and return owned immutable float64
arrays. Empty arrays are supported. Invalid domains raise `ValueError`, replacing
native sentinels and invalid evaluations. Results outside the finite float64
range raise `ArithmeticError` for the whole call. True gamma output underflow is
allowed and retains its sign. This is a real-valued API, not complex log gamma.

```python
from mdanderson_stats import alngam, gamln, gamln1, gam1, gamma, psi

finite_logs = gamln([1e-309, 0.5, 1, 2])
negative_log = alngam(-177.5)
small_remainders = gamln1([1e-100, 1e-10])
reciprocal_remainders = gam1([1e-100, 1 + 2**-52])
signed_tails = gamma([-172.5, -175.5, -177.5])
negative_digamma = psi(-2147483647.5)
```

## Numerical implementation

The local log-gamma calculation vectorizes Morris's two original rational
approximations, retaining their coefficients and the existing CDFLIB source
notice. It evaluates directly in a near a=0 and in a−1 near a=1, preserving
remainders that would disappear if 1+a were formed first. It reuses the package's
vectorized Horner evaluator.

For positive log gamma, arguments through 0.8 use log Γ(1+x)−log(x); those through
2.25 use the local approximation at x−1. This avoids reciprocal overflow for
subnormal x and preserves the zeros at 1 and 2. Larger arguments use SciPy's
compiled log-absolute-gamma kernel. `alngam` checks the gamma sign before using
that kernel on negative nonintegers, so negative intervals with positive gamma
remain supported without forming gamma or performing unbounded recurrence.

`gam1` uses expm1 of the negative log-gamma remainder. Outside the local
approximation's interval, its domain stays sufficiently far from the remainder's
zeros to use the compiled log-gamma kernel safely.

`gamma` uses the compiled direct kernel on ordinary arguments. Below −170 it
combines the separately computed sign and exponential of log-absolute-gamma.
This avoids an overflowing positive-gamma intermediate and recovers representable
subnormal results. `psi` uses SciPy's compiled digamma evaluation after pole
validation, eliminating the native default-integer cutoff. No existing
distribution kernel was changed and no dependency was added.

## Validation and performance

The [native audit](cdflib-gamma-support-reference.md) contains 237 unchanged F95
calls. Implementation tests compare against independent Decimal recurrence and
Stirling calculations, explicitly repairing the identified native failures.
Additional checks cover exact roots, rational switches, factorial/recurrence and
reflection identities, signed subnormal/zero outputs, negative arguments beside
poles, domain boundaries, overflow, ownership and empty arrays. Negative values
beside poles also use independent high-precision product recurrence.

The API does not promise correctly rounded transcendental results at every input.
Relative accuracy can deteriorate near nontrivial digamma zeros and real log-gamma
zeros outside the explicitly stabilized neighborhoods of 1 and 2.

[Benchmarks](cdflib-gamma-support-benchmark.json) compare batches of 64 and 256
coordinates with repeated scalar calls to the same public API. Every comparison
checks identical results; negative subnormal gamma is included. These measure
batching benefits, not speedup against Fortran.

The later [paired beta integral](cdflib-bratio.md) completes all 35 F95
mathematical procedures. Imported constants and other CDFLIB support interfaces
remain open, along with the rest of the software catalog. CDFLIB90 remains partial.
