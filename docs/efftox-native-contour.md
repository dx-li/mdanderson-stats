# Recovered legacy EffTox contour fitting

`fit_efftox_native_contour` implements the inverse-quadratic fitting routine
exported by the official EffTox 5.2.3 calculations library. Current GUI contours
use Lp utility; this is the preserved legacy fitting service.

```python
from mdanderson_stats import fit_efftox_native_contour

fit = fit_efftox_native_contour([[0.5, 0], [0.7, 0.25], [1, 0.65]])
print(fit.coefficients)  # approximately (2.2916667, -2.1375, 0.4958333)
print(fit.fallback, [(r.iterations, r.converged) for r in fit.runs])
contour = fit.as_contour()
print(contour.utility([0.5, 0.7, 1], [0, 0.25, 0.65]))  # approximately zero
```

Pass `contour` to `efftox_decision` or `simulate_efftox`. This bridge evaluates
published radial desirability with the fitted curve. The native legacy utility
implementation has not been recovered, so this is mathematical compatibility,
not native utility parity. Nonphysical curves and native fallback results are
rejected by `as_contour`; fitting itself still reports those results.

The curve is g(e)=a+b/e+c/e². Its objective weights the first and last squared
residual by four, other residuals by one. At the **third** efficacy coordinate,
penalties of 1000 multiply negative slope, negative fitted toxicity and toxicity
above one. A further penalty 1000(c−D) applies when c>0 and discriminant
D=b²−4ac<0. The solver compares a fixed start, three-point interpolation and,
when third efficacy equals one, interpolation of the first two points with
zero slope at one. Each uses a 0.05 simplex and at most 200 steps.

The original expansion, second-highest bookkeeping and strict stopping tests
are preserved; ordinary SciPy Nelder–Mead is not equivalent. The returned runs
expose objective, iterations, evaluations and convergence. If the best fit fails
the original validity test, native fallback (.5,.5,−.5) is reported explicitly.
When every objective is at least 100, original outputs are uninitialized:
Python raises `ArithmeticError`. Python also rejects nonfinite, ill-conditioned,
extremely small-efficacy and over-1000-row inputs. These are explicit bounds,
not claims about native exception behavior.

## Verification and provenance

[Official archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/EffTox/EffTox_V5.2.3.0.zip)
SHA-256: `84fc159c00bb825d0e28c18d095a77a98a8fd97e17c8b2451f9e03e3484b5c47`.
Calculations DLL: `7570fa52e5c03c4818ae8c4f17913bd75a2f036ebac844aa6c86e2ce3a183fd0`.

`tools/reference_efftox_native_contour.py` executes original managed instructions
for the solver, objective, simplex operations, stopping, validity, and LU solve.
It requires separately obtained original DLL plus research dependencies `dnfile`
and `dncil`; no original binary or instruction bytes ship in the package.
Host services provide containers and elementary vector arithmetic. Fifteen
fixtures cover fallback, exhausted starts, additional targets, narrow coordinates
and a third efficacy below one. This is bounded instruction execution, not a
Windows/.NET process test. All iteration/evaluation counts match; coefficient
comparisons allow rounding amplification in ill-conditioned and exhausted runs.
One hundred independently calculated rational objective cases also matched the
original interpreter within relative 1e-14 / absolute 1e-10.
