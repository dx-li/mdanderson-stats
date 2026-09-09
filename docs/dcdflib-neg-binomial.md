# Legacy DCDFLIB negative binomial

`cdfnbn` and `cumnbn` implement the archived C/F77 negative-binomial interfaces.
They supplement the more tightly bounded F95 `cdf_neg_binomial` interface.

```python
from mdanderson_stats import cdfnbn, cumnbn

r = cdfnbn(f=2.5, s=3, pr=0.4)
f = cdfnbn(2, p=r.p, q=r.q, s=3, pr=0.4).f
s = cdfnbn(3, p=r.p, q=r.q, f=2.5, pr=0.4).s
chance = cdfnbn(4, p=r.p, q=r.q, f=2.5, s=3)
p, q = cumnbn(2.5, 3, chance.pr, cpr=chance.cpr)
```

## Contract

Modes 1–4 compute p/q, failures f, successes s, and pr/cpr respectively.
Omit the computed group. Source S/XN map to f/s, and PR/OMPR map to pr/cpr.
All input counts are nonnegative finite with no upper cap. Computed counts
are bounded by 1e100. Counts remain real; inverses do not round to integers.
Results are owned, immutable broadcast arrays in `DCDFLIBNegativeBinomial`.

Both probability pairs accept either coordinate or both, retaining the smaller
coordinate. Supplied pair sums must be within three machine epsilons of one.
Every inversion requires q>0, matching the source. Count inversions additionally
require p>0 and interior pr/cpr; failure inversion needs positive successes.
These conditions reject unidentifiable or unattainable boundary targets.
Chance inversion requires positive successes and maps p=0 to pr=0 exactly.

Zero required successes give P=1, Q=0 at every chance, including pr=0. This adopts
the established F95 Python convention: no trials are needed for zero successes.
The native zero-success/pr=0 result differs, as retained in the
[reference audit](dcdflib-neg-binomial-reference.md).

Invalid input, unattainable bounds, unidentified parameters and unrepresentable
chance coordinates raise `ValueError`. Numerical-kernel failures and inverse
answers that fail their forward check raise `ArithmeticError`. No native integer
status is returned. General extreme beta shapes can still exceed the numerical
kernel's capabilities; the implementation reports failure instead of fabricating
a probability or claiming that every finite input is numerically resolvable.

## Numerical methods

The distribution is I_pr(s,f+1). General cases use direct complementary beta
ratios. Exact power identities handle f=0 and s=1 using logarithms and `expm1`,
so small complementary coordinates survive rounding. Symmetric shapes at pr=cpr
return exactly 1/2 before calling a kernel that may fail on enormous shapes.
Zero counts/chance endpoints are handled before beta evaluation.

For disparate shapes, beta(a,b) approaches gamma(a,rate=b), or its reflected
counterpart. The forward gamma path requires b>=1e20, a*a/b<1e-14 and
(b*x)**2/b<1e-14, checked in logarithms; the reflected path swaps shapes and
coordinates. These constrain the leading normalization and density corrections.
Inverse chance candidates use the corresponding gamma inverse when the shape
condition holds, then undergo the same forward verification as every inverse.
Independent finite sums test both orientations at large shapes 1e20 and 1e200.
The shared gamma code supplies logarithmic scaling and subnormal-tail recovery.

For success shape below 1e-300 outside the exact/gamma paths, the upper tail
is evaluated by its linear-in-shape limit. A normal reference shape
1e-14/(1+abs(log(pr))+log(f+1)) bounds the leading relative correction near 1e-14;
scaling its directly evaluated upper tail avoids premature underflow. Tiny-shape
chance inversion searches the smaller chance coordinate explicitly, retaining
both complements. Independent 400-digit integer-shape series validate these
tails and both inversions down to success counts of 1e-320.

Count inversion uses the shared batched positive-parameter search. It searches
f+1 over [1,1e100] for failures and positive representable s through 1e100 for
successes. As in the native f+1 reduction, failure resolution is limited near
zero. Mean-based initial values preserve large central roots; a small upper
target adjusts the success-count anchor. Logarithmic fallback avoids overflow,
and exact power identities supply initial values at the special boundaries.
Original row indices remain aligned when endpoint rows resolve early.

Chance inverses use direct beta coordinates, exact power formulas, or the gamma
limit. A positive target requiring an unrepresentable chance/complement is
rejected. Every inverse is checked against the smaller forward tail with
1e-7 relative tolerance plus 32 minimum subnormal floats. Subnormal probability
quantization can limit parameter accuracy even when tail probabilities agree.

## Validation and performance

The unchanged C/F77 [reference audit](dcdflib-neg-binomial-reference.md) records
180 ordinary cases per language, plus invalid, boundary and wide cases. Its
fixtures retain source/archive hashes, compiler commands, drivers, native
false-success results and explicit timeout records. No native code is bundled.

The 452 implementation tests compare all ordinary references and independently
check integer-count PMF sums, 800-digit power identities and disparate-shape
finite sums. They cover all four modes, complement preservation, zero-success
semantics, empty/mixed batches, ownership, bounds through 1e100 and symmetric
inputs through 1e308. The new inverse returns pr=1e-100 for the native false-success
case that returned 5e-51, and about 1.4426950408889634e-100 successes where the
native routine returned zero. The symmetric 1e308 case returns 1/2 without the
native timeout.

[Batch measurements](dcdflib-neg-binomial-benchmark.json) compare array calls
with repeated scalar calls to the same Python API, with answer agreement checked.
They measure Python batching rather than native C/Fortran execution speed.

For 64/256 rows, recorded speedups were about 34/81 for tails,
45/98 for failures, 37/68 for successes and 29/49 for chance inversion.
