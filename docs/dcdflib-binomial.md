# Legacy DCDFLIB binomial

`cdfbin` and `cumbin` implement the archived C/F77 binomial interfaces,
supplementing the separately bounded F95 `cdf_binomial` API.

```python
from mdanderson_stats import cdfbin, cumbin

r = cdfbin(s=2.5, n=10, pr=0.4)
s = cdfbin(2, p=r.p, q=r.q, n=10, pr=0.4).s
n = cdfbin(3, p=r.p, q=r.q, s=2.5, pr=0.4).n
chance = cdfbin(4, p=r.p, q=r.q, s=2.5, n=10)
p, q = cumbin(2.5, 10, chance.pr, cpr=chance.cpr)
```

## Contract and endpoints

Modes 1–4 compute p/q, successes s, trials n, and pr/cpr respectively.
Omit the computed group. Source XN maps to n and PR/OMPR map to pr/cpr.
Input n is positive finite with no upper cap; input s is nonnegative and,
when n is supplied, cannot exceed n. Computed s spans [0,n], even for n>1e100.
Computed n lies in [max(s,1e-100),1e100]. Counts are continuous real numbers;
count inverses do not round to integer quantiles. Results are owned, immutable
broadcast arrays in `DCDFLIBBinomial`.

Either coordinate of each probability pair may be supplied, or both. Pair sums
must agree with one within three machine epsilons, and the smaller coordinate
is retained. Unlike legacy negative binomial, binomial inversion accepts q=0.
For count inversion this returns the boundary s=n, provided the trial search
bounds permit it and pr>0. Count inversion rejects p=0, pr=0, and pr=1 with
positive q as unattainable or unidentified. At pr=1,q=0 the boundary s=n is
chosen. Forward evaluation at s=n gives P=1,Q=0, including chance endpoints.

Chance inversion requires s<n: s=n cannot identify a chance. It maps p=0 to
pr=1,cpr=0 and q=0 to pr=0,cpr=1 exactly. Unsupported modes are rejected,
repairing the source's impossible invalid-mode guard. Python raises `ValueError`
for invalid, unattainable, unidentified or unrepresentable inputs/results, and
`ArithmeticError` for numerical-kernel or inverse-forward-check failures.

## Numerical methods

The lower tail is I_cpr(n-s,s+1). The implementation reuses the independently
validated legacy negative-binomial kernel with failures=s, successes=n-s and
chance=cpr. Its exact power/geometric identities, symmetry shortcut, bounded
gamma-limit approximations, tiny-shape scaling and direct complementary beta
ratios therefore apply here; see the [kernel notes](dcdflib-neg-binomial.md).
The zero-success identity P=cpr**n uses logarithms and `expm1` to preserve tiny
upper tails. The zero beta shape at s=n is handled before kernel evaluation.

Success inversion searches s directly, with zero handled as an explicit
endpoint and the remaining search extending from the minimum positive float to
n. Searching s rather than 1+s retains fractional success resolution for tiny
n. A mean-based initial value n*pr preserves symmetric roots beyond 1e100 and
stays within the search interval, avoiding the native C initial-value process
exit for n<5. Trial inversion searches n directly over its distinct bounds;
for s=0 the exact logarithmic power inverse supplies its initial value.
Both searches use batched bracketing and logarithmic refinement with original
row indices retained when endpoint rows resolve early.

Chance inversion uses the negative-binomial inverse with exchanged chance
coordinates. This computes tiny pr directly instead of subtracting a rounded
complement from one. Every inverse is checked against the smaller forward tail
with relative tolerance 1e-7 plus 32 minimum subnormal floats. Extreme general
beta shapes can still exceed the kernel's capabilities, and representable count
spacing can prevent an inverse from attaining its target. Such failures are
reported; the API does not guarantee a resolvable inverse for every finite input.
Subnormal probability quantization can also limit parameter accuracy.

## Validation and performance

The unchanged C/F77 [reference audit](dcdflib-binomial-reference.md) records 156
ordinary cases per language plus invalid, boundary and wide cases, preserving
source/archive hashes, compiler commands, drivers and explicit process failures.
No archived source or executable is bundled.

The 394 implementation tests cover all ordinary references, including 18 C
process exits, and independent 120-digit integer PMF sums. Additional 800-digit
power and 400-digit tiny-shape checks validate small tails and inversions.
Tests cover all four modes, mixed/empty batches, immutable ownership, endpoints,
trial search bounds, success searches beyond 1e100 and symmetric inputs through
1e308. Native false-success examples now recover pr=1e-100, trials about
1.4426950408889634e-100, and the symmetric success root 5e199.

[Batch measurements](dcdflib-binomial-benchmark.json) compare one array call
with repeated scalar calls to the same Python API, checking answer agreement.
For 64/256 rows, recorded speedups were about 29/66 for tails, 37/70 for
successes, 37/68 for trials and 31/55 for chance inversion. These measure Python
batching, not speed relative to native C or Fortran.

CDFLIB90 remains partial: six legacy distribution entry points and the public
numerical/support interfaces still require reconciliation.
