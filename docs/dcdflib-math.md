# Legacy DCDFLIB mathematical support

The `dcdflib_support` namespace now maps all 31 mathematical helpers shared by
the archived C/F77 libraries. It reuses the validated vectorized kernels already
available for F95 support, with a separate implementation where the legacy
contract differs. Together with the ten [support primitives](dcdflib-support.md),
these account for 41 legacy support contracts. The later
[normal/t quantile helpers](dcdflib-quantile-helpers.md) and
[incomplete-gamma inverse](dcdflib-gamma-inverse.md) bring the total to 45 of 49.

```python
from mdanderson_stats import dcdflib_support as legacy
from mdanderson_stats import exparg as f95_exparg

assert float(legacy.Xgamm(4)) == 6
p, q = legacy.bratio(2, 3, [0.25, 0.5])
assert p.shape == q.shape == (2,)
assert float(legacy.exparg(0)) < float(f95_exparg(0))
```

## Complete mapping

Names in this table are available under `dcdflib_support`. C pointers and Fortran
output arguments become returned immutable float64 arrays; numerical arguments
broadcast. Scalar inputs return zero-dimensional arrays. Domains below refer to
finite real inputs. The linked kernel pages specify numerical methods, tolerance
policies, endpoints, extended domains and independent validation in greater detail.

| Legacy name | Python contract and domain |
|---|---|
| `alnrel(x)` | log(1+x), x>-1; [elementary helpers](cdflib-elementary.md) |
| `rexp(x)` | exp(x)-1, representable finite result |
| `rlog(x)` | x-1-log(x), x>0 |
| `rlog1(x)` | x-log(1+x), x>-1 |
| `alngam(x)` | Real log Γ(x), where Γ(x)>0; [gamma helpers](cdflib-gamma-support.md) |
| `gamln(x)` | log Γ(x), x>0 |
| `gamln1(x)` | log Γ(1+x), -0.2≤x≤1.25 |
| `gam1(x)` | 1/Γ(1+x)-1, -0.5≤x≤1.5 |
| `Xgamm(x)` / F77 `gamma` | Γ(x), excluding nonpositive integer poles; Python legacy name is `Xgamm` |
| `psi(x)` | Digamma, excluding nonpositive integer poles |
| `erf1(x)` / F77 `erf` | Error function; Python legacy name is `erf1`; [error/exponential helpers](cdflib-error-exponential.md) |
| `erfc1(ind,x)` | erfc(x) for ind=0, exp(x²)erfc(x) otherwise |
| `esum(mu,x)` | exp(mu+x), with a signed int32 scale |
| `exparg(l)` | Conservative legacy exponential limits described below |
| `algdiv(a,b)` | log Γ(b)-log Γ(a+b), b≥8 and a+b>0; [gamma ratios](cdflib-gamma-ratios.md) |
| `bcorr(a,b)` | δ(a)+δ(b)-δ(a+b), a,b≥8; δ is the Stirling correction |
| `gsumln(a,b)` | log Γ(a+b), a,b in [1,2] |
| `betaln(a,b)` | log B(a,b), positive shapes; [beta logarithms](cdflib-beta-support.md) |
| `rcomp(a,x)` | exp(-x)x^a/Γ(a); [gamma factor](cdflib-gamma-factor.md), including documented signed continuation and zero rules |
| `grat1(a,x,r,eps)` | (P,Q), 0≤a≤1, x≥0, r≥0, eps>0; [gamma tails](cdflib-incomplete-gamma.md) |
| `gratio(a,x,ind)` | (P,Q), a,x≥0 excluding (0,0); all accuracy modes use full Python precision |
| `brcomp(a,b,x,y)` | x^a y^b/B(a,b); [beta factors](cdflib-beta-factors.md) |
| `brcmp1(mu,a,b,x,y)` | exp(mu) times the beta factor, including documented signed continuation |
| `bup(a,b,x,y,n,eps)` | I_x(a,b)-I_x(a+n,b), positive shapes, positive int32 n, eps>0; [shape shift](cdflib-beta-shift.md) |
| `fpser(a,b,x,eps)` | I_x(a,b), positive shapes, x in [0,0.5], b<min(eps,eps*a); [tiny companion series](cdflib-fpser.md) |
| `apser(a,b,x,eps)` | I_(1-x)(b,a), positive shapes, x in [0,0.5], a≤min(eps,eps*b), b*x≤1; [small first-shape series](cdflib-apser.md) |
| `bpser(a,b,x,eps)` | I_x(a,b), positive shapes, x in [0,1], b≤1 or b*x≤0.7; [power series](cdflib-bpser.md) |
| `bgrat(a,b,x,y,w,eps)` | Returns w+I_x(a,b), a≥15, 0<b≤1, eps>0; [accumulated integral](cdflib-bgrat.md) |
| `basym(a,b,lambda_,eps)` | I_x(a,b), x=(a-lambda_)/(a+b), a,b≥15, 0≤lambda_≤a; [asymptotic helper](cdflib-basym.md) |
| `bfrac(a,b,x,y,eps)` | I_x(a,b), a,b>1; displacement is computed internally; [continued fraction](cdflib-bfrac.md) |
| `bratio(a,b,x,y)` | Paired beta tails with nonnegative shapes and nonsingular zero-shape limits; [paired integral](cdflib-bratio.md) |

Explicit integer selectors/scales must be integral signed int32 values. Series
and fraction tolerances must be positive. Optional coordinate/accuracy defaults
follow the linked Python APIs. Paired coordinates preserve the smaller supplied
value; their validation/reconstruction policy is documented by each kernel.

## Cross-version differences and Python semantics

**Legacy `exparg` is distinct from the F95/root-level function.** C and F77 return
`0.99999 * (m * 0.69314718055995)`, using m=1024 for l=0 and m=-1022 otherwise.
These are approximately 709.7756150662599 and -708.3893345680835. Python's legacy
namespace preserves both the safety margin and the source's rounded log(radix).
F95/root-level `exparg` continues to return log(maximum) and log(minimum normal).
Neither lower limit is the smallest exponent with a nonzero subnormal result.

The native `bfrac` displacement argument must equal `(a+b)*y-b`. Python computes
it from a,b,x,y using the validated compensated kernel rather than accepting an
independently inconsistent value. Unlike `bfrac`, `basym` retains its displacement
argument because it defines a coordinate without rounding it first.

`bratio`, `grat1` and `gratio` return both tails instead of modifying caller-owned
output arguments. `bgrat` returns a new accumulator. Native `IERR` failures and
invalid inputs become exceptions, not partially initialized results. The audit
exercises all seven `bratio` input-error statuses; the corresponding Python
`ValueError` retains the IERR number. `gratio`'s invalid-domain sentinel P=2 and
zero gamma/digamma pole sentinels likewise become explicit errors.

The `grat1` scaling argument remains meaningful: its continued-fraction branch
consumes the supplied r. Reduced-accuracy `gratio` modes are accepted but compute
full precision. Valid tolerances control bounded Python methods as described by
the kernel pages; they do not promise to reproduce native iteration counts.

Domain comparisons use the exact binary64 inputs when rounded products conceal
a boundary crossing. In the recorded apser case b=10, x=0.1, binary64 multiplication
rounds to one although the exact product exceeds one. Native code admits it;
Python rejects it according to the documented b*x≤1 condition. This difference
is tested explicitly using rational arithmetic.

The existing numerical repairs also apply through the legacy namespace:
subnormal log remainders, error-function tails and signed gamma results are
preserved; tiny log-gamma arguments avoid reciprocal overflow; digamma does not
inherit the native default-integer cutoff; and very small gamma-ratio shifts
retain their leading term. Extended signed/huge domains use the linked validated
Python kernels. Invalid domains raise `ValueError`, numerical failures raise
`ArithmeticError`, and genuine output underflow is permitted.

## Validation and performance

The [reference generator](../tools/reference_dcdflib_math.py) extracts unchanged
source bytes from the SHA256-verified archive and compiles C and F77 independently.
The [fixture](../tests/fixtures/dcdflib_math.json) records **718 completed calls per
language**, source and driver hashes, compiler options, output pairs and status.
Every call has a three-second limit; none timed out or failed to execute.

The [tests](../tests/test_dcdflib_math.py) compare ordinary cases across the public
helper domains and approximation switches. High-accuracy native values are
compared at 5e-13 relative tolerance, with a small absolute allowance only at exact
logarithmic zeros. Legacy `gratio` reduced-accuracy modes use their looser
reference tolerances; the Python kernels remain covered by independent
full-precision tests. Source losses and extreme values are checked against reused
independent Decimal series, recurrence, Stirling and beta-integral oracles, with
explicit nonzero and sign assertions. The full existing numerical suite continues
to exercise broad domains, subnormal tails, overflow, output ownership and errors.

These are the same batched kernels described and benchmarked in the linked
F95 pages, not duplicate numerical implementations. `exparg` evaluates selectors
in one NumPy batch. No new dependency or native-language speedup claim is added.

The [normal/t quantile helpers](dcdflib-quantile-helpers.md) are also implemented.
Four legacy contracts remain: `dinvr`, `dstinv`, `dzror` and `dstzr`. CDFLIB90 stays partial until these and the final
archive/documentation audit are complete; the full catalog conversion is ongoing.
