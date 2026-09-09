# CDFLIB beta tail from a displacement

`basym(a,b,lambda_,eps=5e-15)` computes I_x(a,b), where
x=(a-lambda_)/(a+b). It retains the source domain a,b>=15 and nonnegative
lambda, with lambda_<=a to define a probability coordinate. All inputs must be
finite and eps must be positive. Inputs broadcast to independently owned,
immutable float64 outputs.

```python
from mdanderson_stats import basym

ordinary = basym(50, 100, 5)
endpoint = basym(15, 50, 15)  # zero
small_displacement = basym(1e308, 1e308, 1e146)
far_tail = basym(50, 15, 25)
```

## Preserving the displacement

The expansion's exponent is
`f = a*rlog1(-lambda/a) + b*rlog1(lambda/b)`.
Computing the tiny remainders before multiplying by their shapes can underflow.
For |lambda/shape|<=0.5, the implementation factors the logarithmic remainder
and evaluates `lambda*(lambda/shape)*H`, using a 64-term polynomial for H.
Its omitted relative series tail is below 1e-20; floating-point rounding dominates.
Outside this interval, stable logarithmic expressions preserve a-lambda near the
endpoint. The exponent never requires forming a+b.

This repairs the [audited native failure](cdflib-beta-remaining-reference.md) at
a=b=1e308,lambda=1e146: the result is approximately 0.49999999435810416, while the
native routine returns 0.5. Reconstructing a float64 x first would also lose the
displacement. The x=0 endpoint and symmetric lambda=0 midpoint are handled exactly.

The inverse square-root scale also avoids an overflowing shape product. A gamma
moment-generating-function bound gives I_x(a,b)<=exp(-f) for nonnegative lambda.
Values with f>800 therefore provably underflow in float64, with ample margin for
exponent rounding.

## Expansion and alternate evaluations

The source's coefficient recurrences are evaluated in NumPy batches. Each batch
has at most 2048 elements, limiting the four coefficient matrices to about
1.4 MiB. Successive coefficient pairs use the source convergence test; the work
limit remains 20 orders. Requested tolerance is capped at 5e-15 and floored at
four machine epsilons. Already converged results retain their first accepted sum.
Complete exponential scaling preserves representable subnormal tails.

If the convergence test is not met, a positive beta-integral series handles
x<=0.5. It sums `(a+b)_n*x**n/(a+1)_n`, including the n=0 term, with successive
ratios formed as `(a-lambda)/(a+n) * (1+(n-1)/(a+b))`. The complete normalization
comes from f and the stable Stirling correction bcorr. Neither the normalization
nor the recurrence rounds x first. Ratios decrease because b>1, giving a geometric
remainder bound; this alternate series has a 4096-term limit. A ratio rounded to
one cannot incorrectly certify convergence.

For x>0.5 after failed asymptotic convergence, the existing legacy beta adapter
uses both coordinates, formed with a scaled denominator. These alternate paths
repair the native far-tail error at (a,b,lambda)=(50,15,25), whose relative error
was about 1.06e-8 despite eps=5e-15. The positive series also retains the last
representable displacement before the endpoint with b=1e308, where a reconstructed
subnormal x materially changes the integral. Independent positive integrals check
very small large-companion tails. The implementation uses the beta integral
directly.

Invalid input raises `ValueError`. Invalid numerical output or failure of the
alternate evaluator raises `ArithmeticError`. The convergence test is not a
universal bound on asymptotic or kernel error, and eps is not a correct-rounding
guarantee. No partially converged expansion is returned after the work limit.

## Validation and performance

[Tests](../tests/test_cdflib_basym.py) compare native audit cases and broader grids
against independent 800-digit beta integrals, positive binomial sums and a separate
midpoint-density expansion. They cover large integer shapes, huge symmetric and
unequal shapes, displacements lost by float64 coordinates, representable subnormal tails, the last representable
displacement before an endpoint, tolerance and algorithm transitions, broadcasting,
immutable ownership and batches exceeding the coefficient-work limit.

[Benchmarks](cdflib-basym-benchmark.json) require identical batched and repeated
scalar outputs. They measure batching benefits, not speed relative to Fortran.

The later [bfrac port](cdflib-bfrac.md) brings coverage to 34 of the 35 F95
mathematical procedures. `bratio`, other CDFLIB support interfaces and the
remaining software catalog still require work. CDFLIB90 remains partial.
