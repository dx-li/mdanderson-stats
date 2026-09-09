# Legacy DCDFLIB beta

`cdfbet` and `cumbet` implement the archived C/F77 beta contracts, supplementing
the separately bounded F95 `cdf_beta` API.

```python
from mdanderson_stats import cdfbet, cumbet

r = cdfbet(x=0.2, a=0.5, b=0.25)
quantile = cdfbet(2, p=r.p, q=r.q, a=0.5, b=0.25)
a = cdfbet(3, p=r.p, q=r.q, x=0.2, b=0.25).a
b = cdfbet(4, p=r.p, q=r.q, x=0.2, a=0.5).b
p, q = cumbet(None, 0.5, 0.25, cx=0.8)
```

## Contract

Modes 1–4 compute p/q, x/cx, a and b respectively; omit the computed group.
Source Y maps to cx. Input shapes are positive finite with no upper cap.
Computed shapes are bounded by [1e-100,1e100]. Coordinates and probabilities
lie in [0,1]. Either member of a complementary pair may be supplied, or both;
provided sums must agree with one within three machine epsilons. The smaller
coordinate is retained. Results are owned, immutable broadcast arrays in
`DCDFLIBBeta`, including scalar and empty batches.

Quantile endpoints are exact: p=0 gives x=0,cx=1, and q=0 gives x=1,cx=0.
Shape inversion requires interior coordinates and positive probabilities:
endpoint distributions do not identify a shape, and an exact zero tail at an
interior coordinate is unattainable for finite positive shapes. These rules
reject native arbitrary initial-shape and underflow-based false successes.

Invalid inputs, unattainable search bounds, unidentified shapes and
unrepresentable quantiles raise `ValueError`. Numerical-kernel failures and
inverse answers that fail forward verification raise `ArithmeticError`.
No native integer status is returned. Extreme general beta parameters can still
exceed the numerical kernels' capabilities; accepting a finite input domain
is not a guarantee that every inverse is resolvable in floating-point arithmetic.

## Numerical methods

For b>=1, beta(a,b) maps to the validated negative-binomial beta kernel with
failures=b-1 and successes=a. For a>=1 with b<1, reflection supplies the same
mapping. The shared [negative-binomial methods](dcdflib-neg-binomial.md) include
exact power/geometric identities, direct complementary beta ratios, gamma limits
for sufficiently disparate shapes, tiny-shape repairs and symmetry shortcuts.
The legacy beta implementation does not change those existing kernels.

For a,b<1, each tail is expressed as a sum of positive terms:

```text
P = I_x(a+1,b) + x**a * cx**b / (a*B(a,b))
Q = I_cx(b+1,a) + x**a * cx**b / (b*B(a,b))
```

The raised-shape tails use the shared kernel. The boundary terms use
b/(a+b) and a/(a+b), respectively, multiplied by
exp(a*log(x)+b*log(cx)+logGamma(1+a+b)-logGamma(1+a)-logGamma(1+b)).
All gamma arguments lie between one and three. This avoids division by an
underflowed beta normalization or subtraction of nearly equal probabilities,
and preserves the relative tail when either or both shapes are subnormal.
The smaller evaluated tail determines the final complementary pair. Exact
symmetry returns 1/2. SciPy 1.18.1's installed `gammaln` documentation was checked
for the real log-gamma normalization, alongside its current source documentation.

Quantiles use the corresponding negative-binomial chance inverse when a raised
shape is unnecessary, otherwise direct complementary beta inversion. Exact
symmetric medians are handled before the compiled inverse, which fails for
symmetric 1e308 shapes. With both shapes below one and either below 1e-300,
a bounded search explicitly inverts the smaller coordinate over positive
representable floats through 1/2. A logarithmic power-law starting estimate
keeps tiny representable roots within the bracketing iteration budget. If the midpoint already matches within
32 machine epsilons of the smaller target, it is returned. When both shapes
are tiny, the CDF can be indistinguishable across a wide coordinate interval;
this numerical convention does not assert unique coordinate recovery.

Shape inversion uses batched bracketing and logarithmic refinement over the
legacy bounds. Mean-based starting values preserve enormous symmetric roots;
exact power identities supply starting values when the other shape is one.
Original row indices remain aligned as endpoint rows resolve early.
Every inverse is verified against the smaller forward tail with relative
tolerance 1e-7 plus 32 minimum subnormal floats. Subnormal probability quantization
and count/shape spacing can limit parameter accuracy even when tails agree.

## Validation and performance

The unchanged C/F77 [reference audit](dcdflib-beta-reference.md) retains 256
ordinary calls per language and additional invalid, boundary and wide cases,
including explicit timeout records. Source hashes, compilers and drivers are
recorded; no original source or executable is bundled.

The 650 implementation tests cover all ordinary native references, independent
120-digit integer-shape binomial sums, 150-digit arcsine identities, 400-digit
tiny-shape integrals and 800-digit power identities and tiny-shape ratios.
They exercise all modes, immutable broadcasting, empty/mixed batches,
shape-search endpoints, unrepresentable targets and symmetric shapes through
1e308. The audited false successes now return quantiles of 1e-100, shapes near
1.4426950408889634e-100, and the exact symmetric shape 1e90. Exact quantile
endpoints replace native spurious coordinates of 5e-51.

[Batch measurements](dcdflib-beta-benchmark.json) compare array calls with
repeated scalar calls to this Python API, checking answer agreement. These
measure Python batching rather than speed relative to native C or Fortran.
For 64/256 rows, recorded speedups were approximately 41/107 for tails,
34/65 for quantiles, 43/93 for a and 44/101 for b.

See the [CDFLIB90 completion audit](cdflib90-completion.md) for the
complete library scope.
