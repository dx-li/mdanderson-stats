# Legacy incomplete-gamma inverse

`dcdflib_support.gaminv(a, p, q=None, x0=0)` computes the unit-rate gamma
coordinate with lower/upper probabilities p/q. Inputs broadcast and the result
is an owned immutable float64 array, including a zero-dimensional scalar result.

```python
import math
from mdanderson_stats import dcdflib_support as legacy

assert legacy.gaminv(1, None, math.exp(-1), x0=1) == 1
quantiles = legacy.gaminv([0.5, 2, 100], 0.5)
extreme = legacy.gaminv(0.5, None, 5e-324)
```

## Native contract and Python mapping

The archived C and F77 routine takes `(a,x,x0,p,q,ierr)`, writes x and reports
the status or iteration count in ierr. The Python call returns x and uses
exceptions for errors; it does not fabricate native iteration counts.

| Native input or result | Python behavior |
|---|---|
| Positive shape a | Positive finite input, with no arbitrary search ceiling |
| Complementary p/q | Finite values in [0,1]; either can be omitted using `None` |
| Explicit p+q | Three-epsilon tolerance, consistent with the package's other legacy inverses; native uses one epsilon |
| p=0 | Returns exactly zero |
| q=0 | Retains the native maximum-finite sentinel, **not** a finite solution to Q(a,x)=0 |
| x0<=0 | Automatic inverse |
| x0>0 | Check the supplied candidate; retain it if sufficiently accurate, otherwise solve automatically |
| ierr>=0 | Successful returned coordinate; no native iteration-count claim |
| ierr=-2/-4 | Invalid shape/probabilities raise `ValueError` |
| ierr=-3/-6/-7/-8 | Repaired when a validated solution is available; unrepresentable coordinates raise `ValueError`, failed numerical verification raises `ArithmeticError` |

The smaller supplied probability is preserved while the larger is reconstructed.
This permits tiny tails whose complement rounds to one. All inputs, including
x0 at endpoints, must be finite and broadcastable. One invalid batch member
fails the whole call. Inputs are not mutated.

Positive hints are optional acceleration information, not a request to replay
the source's 20-step Schroder iteration. Candidate acceptance requires both
relative tail residual <=1e-12 and estimated relative coordinate correction
<=1e-12. The latter uses `rcomp(a,x)=x*dP/dx`; a zero density factor does not
qualify. Hints with smaller-tail probabilities below 1e-280 use automatic solving
to avoid acceptance based solely on rounded subnormal bins. A poor finite guess
therefore does not prevent recovery of a valid solution.

## Numerical implementation and limits

Ordinary automatic cases reuse the existing [legacy gamma inverse](dcdflib-gamma.md),
its small-shape/logarithmic repairs and forward verification. A positive
probability whose coordinate underflows to zero is rejected. A representable
subnormal coordinate can also fail verification when its spacing is too coarse
to reproduce the requested probability accurately.

Two additional paths address defects exposed by independent checks during this
port. They apply to `gaminv`; the existing distribution interfaces are unchanged.

* For 1e-12<=a<1 and subnormal upper q, a compiled initial coordinate is refined
  in the log domain. An integration-by-parts expansion of the upper gamma integral
  has a first-omitted-term relative bound below 1e-50 in this x>600 regime.
  Three Newton corrections use 32 terms; final log-tail residual must be <=2e-12.
  This avoids both quantized false success and premature failure from rounded
  probability checks. For example, a=0.5, q=5e-324 needs a coordinate around
  740.5633, while the archived output is around 740.4426. Both tails round to the
  same minimum subnormal, concealing the coordinate error in ordinary checks.
* For a>=1e20, use the normal deviate z for the preserved tail pair and the
  Cornish-Fisher coordinate `a + sqrt(a)*z + (z*z-1)/3`. Every positive float64
  tail pair has |z|<39. The next expansion term has magnitude below 2e-7 at
  a=1e20, already far below the coordinate's float64 spacing. At enormous a,
  all positive-probability quantiles round to a. Returning that rounded coordinate
  is intentional even though evaluating its CDF cannot recover the original
  tail. The endpoint sentinel remains a separate contract.

These calculations do not promise a uniform relative probability error at every
finite float64 input. Tail probabilities and coordinates both have resolution
limits. Numerical failures remain explicit rather than returning native failed
iterates, negative coordinates or an unreported zero.

## Validation and performance

The [reference generator](../tools/reference_dcdflib_gamma_inverse.py) verifies
the archive SHA256 and compiles unchanged C and F77 sources. The
[fixture](../tests/fixtures/dcdflib_gamma_inverse.json) records 271 calls per
language, compiler settings, source/driver hashes, native statuses and outputs.
Each native call has a three-second execution limit. Cases include ordinary
shapes/probabilities, optional and poor starting values, endpoints, invalid
inputs, tiny shapes, subnormal tails and enormous shapes.

The [tests](../tests/test_dcdflib_gamma_inverse.py) independently evaluate gamma
integrals using 800-digit Decimal series and logarithmic tails, without rounding
the oracle into float64 probability bins. Large-shape checks use an independent
normal integral and explicit coordinate-spacing bounds. Successful ordinary
native results are compared; failed native outputs are retained as evidence of
repairs rather than used as numerical truth. Broadcasting, ownership, hint
acceptance, invalid inputs and strict floating-point error settings are covered.

The [benchmark](dcdflib-gamma-inverse-benchmark.json) compares one NumPy batch
with repeated scalar calls to this same API, requiring identical outputs. It
does not claim a speedup over native C or Fortran. There is no per-element Python
loop in the production inverse; special paths operate on selected array batches.

This inverse reconciles 45 of 49 legacy support names. The subsequent
[legacy root finders](dcdflib-root.md) complete all 49 mappings.
See the [CDFLIB90 completion audit](cdflib90-completion.md) for the
complete library scope.
