# CDFLIB beta and gamma-ratio support audit

Six further public F95 mathematical procedures now have unchanged native
reference evidence. Their Python interfaces remain pending. CDFLIB90 remains
partial; this audit adds validation requirements rather than claiming a port.

| Procedure | Mathematical operation and source constraints |
|---|---|
| `algdiv(a,b)` | log Γ(b)−log Γ(a+b); header specifies b≥8 |
| `bcorr(a,b)` | δ(a)+δ(b)−δ(a+b), with a,b≥8 |
| `betaln(a,b)` | log B(a,b), using positive beta shapes |
| `log_beta(a,b)` | Renamed copy of `betaln` |
| `gsumln(a,b)` | log Γ(a+b), with 1≤a,b≤2 |
| `log_bicoef(k,n)` | −log B(k+1,n−k+1)−log(n+1), with real k,n |

Here δ(x) is the Stirling correction defined by
log Γ(x)=(x−1/2)log(x)−x+log(2π)/2+δ(x).

The combinatorial helper is a continuous gamma expression. Valid examples include
fractional counts, k>n and negative k or n: the positive-shape formula is defined
when n>−1 and −1<k<n+1. Restricting it to nonnegative integer counts would discard
valid native behavior. At the formula's poles, the native routine can instead
produce a misleading finite answer from an internal sentinel.

The `algdiv` header restricts b but does not explicitly require a≥0. Negative a
can produce a valid finite gamma ratio when a+b>0. The audit records both a
successful case and failures where the approximation is inaccurate at small a+b;
a port must distinguish mathematical validity from approximation applicability.

## Native evidence and independent checks

`tools/reference_cdflib_beta_support.py` compiles the unchanged constants and
mathematical modules with bounds checking and contraction disabled. All 192 calls
completed. The strict JSON fixture contains the compiler, command, driver,
archive/source hashes and no source adaptations. This validates F95 procedures,
not similarly named C/F77 implementations.

The cases cover unit and half shapes, source branch boundaries, symmetry,
subnormal shapes and ratios, arguments whose sum exceeds float64, fractional
combinatorial inputs, close sums and invalid boundaries. Independent 800-digit
Decimal calculations use log-gamma identities, recurrence and Stirling expansions.
Local expansions around gamma's unit values preserve offsets that a fixed
absolute-error approximation would hide. Exact binomial coefficients, B(1,b)=1/b,
B(1/2,1/2)=π and integer gamma-ratio identities independently check the oracle.
The two renamed beta entry points agree on every recorded case.

## Findings that affect the port

* `algdiv` forms a/b directly. When that quotient rounds to zero, a leading
  contribution is lost. For example, a=1e-100 and b=1e308 produce an error exceeding
  0.01% despite a finite, nonzero answer. A nonzero but subnormal quotient can also
  lose accuracy, as at a=1e-10, b=1e308.
* `algdiv(-1,10)` correctly gives log(9). At (-8,10), the absolute error is about
  2.7e-7; at (-9.5,10), it exceeds 3. Both mathematical ratios are finite, but the
  asymptotic correction no longer has its intended accuracy at small a+b.
* `gsumln(1,1+2**-52)` returns zero because a+b rounds to two before subtraction.
  The mathematical log gamma of the exact input sum is about 9.39e-17. The same
  issue affects the symmetric case and propagates into `betaln(1,1+2**-52)` and
  `log_beta`, producing more than 40% relative error in the tiny result.
* `log_bicoef(1e-100,1)` and `log_bicoef(5e-324,1)` return zero although the correct
  results round to 1e-100 and the minimum positive float, respectively.
* Invalid beta shapes can return −1 or zero as if they were ordinary answers.
  Invalid combinatorial boundary arguments can likewise produce finite results.
  A port needs explicit domain handling.

The source also preserves useful extreme behavior: `bcorr` can return a positive
subnormal correction for enormous arguments, and beta logarithms remain finite
for two shapes of 1e308 despite their overflowing float64 sum. A naive difference
of three float64 log-gamma values would lose these properties. Genuine log-ratio
output overflow is distinguished from avoidable intermediate loss.

The implementation should retain these successes, repair the identified failures,
and keep fractional domains explicit. Other incomplete-beta/gamma ratios,
constants and remaining support interfaces are still open.
