# CUMNOR, GOFCHI, and INVMF

These implementations cover the numerical functionality of the three catalog
programs through Python APIs. Interactive menus, S interpreter bindings, and
platform-specific output files are replaced by function arguments and structured
results. No upstream Fortran implementation is bundled.

| Catalog entry | Python API | Functionality |
|---|---|---|
| CUMNOR (40) | `normal_tails(x, log=False)` | Normal CDF and survival function on scalars or arrays; natural-log output preserves extreme tails |
| GOFCHI (44) | `chi_square_gof(observed, weights, degrees_of_freedom=...)` | Normalize relative null frequencies, calculate expected frequencies, contributions, Pearson statistic and upper-tail p-value; supports batches |
| INVMF (46) | `invert_monotone(function, initial, target, ...)` | Invert increasing or decreasing continuous functions with domain bounds, named parameters, bracket expansion and error controls |

## Examples

```python
from mdanderson_stats import chi_square_gof, invert_monotone, normal_tails

logcdf, logsf = normal_tails([-100, 0, 100], log=True)
result = chi_square_gof([6, 9, 5], [2, 3, 5])
assert abs(float(result.statistic) - 5) < 1e-12
root = invert_monotone(lambda x: x**3, 0, 27, lower=-100, upper=100)
```

CUMNOR's printed mantissa/exponent representation is replaced by natural logs for
probabilities below floating-point range. For instance, `normal_tails(-100,
log=True)[0]` is about -5005.5242; the ordinary CDF underflows to zero. NumPy can
construct input grids using `linspace` or `geomspace` and perform the original
list editing, sorting, and deduplication operations.

GOFCHI uses a chi-square asymptotic null distribution. Choose degrees of freedom
appropriate to the model; the default is the number of positive null weights minus
one, with no parameter fitting. The result preserves category order, so callers
can attach their own names to expected frequencies and contributions. Repeated
tests can be stacked along leading array dimensions. Null weights do not need to
sum to one and are scaled before normalization to avoid overflow.

The original GOFCHI ignores contributions whose expected count is zero, even
when the observed count is positive. The default Python behavior returns an infinite
statistic and p=0 for this impossible observation. `legacy_zero_expected=True`
reproduces the original omission explicitly. A category with both zero observed
and zero expected count contributes zero. SciPy's gamma survival function replaces
the original single-precision chi-square approximation and subtraction from one;
small p-values remain representable.

INVMF evaluates domain endpoints, the initial guess, and geometrically expanding
steps. It checks monotonicity at those sampled points; callers must supply a
continuous monotone function over the entire interval. It refines the bracket
with SciPy's Brent solver, replacing the original Algorithm R. Accuracy is
controlled by `absolute_tolerance + relative_tolerance * abs(root)`. Unlike the
original S implementation, calls can be nested. Functions must return finite
scalar values, including at the endpoints. Unbracketed targets, invalid values,
and exhausted iteration budgets raise errors. Python keyword names for the
original S controls are:

| Original | Python |
|---|---|
| `f`, `init`, `fequals` | `function`, `initial`, `target` |
| `small`, `big` | `lower`, `upper` |
| `name`, `...` | `parameter`, `function_kwargs` |
| `absstp`, `relstp`, `stpmul` | `absolute_step`, `relative_step`, `step_multiplier` |
| `abstol`, `reltol` | `absolute_tolerance`, `relative_tolerance` |

## Reference validation

`tools/reference_numerics.py` compiles the downloaded originals with gfortran.
The fixture records archive URLs, SHA-256 hashes, compiler and flags. Original
archives remain local research inputs under `research/raw/`.

- CUMNOR: 21 arguments spanning both signs, the ordinary/asymptotic transition,
  and both documented extremes of +/-67,861,400. An independent harness calls
  the original `cumnor`/`dlanor` numerical routines without changing their code.
  Additional checks use the standard-library complementary error function and
  Mills-ratio bounds; logs are tested separately from underflowed probabilities.
- GOFCHI: eight executions of the original interactive program, including
  unequal weights, fractional observations, user-specified degrees of freedom,
  perfect fit and zero null frequencies. Tests use the executable's printed
  precision. Separate hand calculations and the exact chi-square(df=2) survival
  formula check the Python implementation independently.
- INVMF: eight cases from the original `dinvr` and `dzror` routines, with a
  Fortran callback bridge replacing only the S interpreter bridge. Cases include
  square roots, exponentials, decreasing linear functions and signed cubics,
  with starts on both sides of the solution. Analytic roots, named parameters,
  nested inversion, invalid evaluations and nonconvergence are also tested.

Sources: [CUMNOR](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/40),
[GOFCHI](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/44),
[INVMF](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/46).
CUMNOR and INVMF are by Barry W. Brown; GOFCHI lists Dennis A. Johnston as contact.

## Batch performance

`uv run python tools/benchmark_numerics.py` measures three repetitions of two
deterministic workloads and checks that batched and scalar results agree. The
[recorded run](numerical-benchmark.json) measured about 128x speedup for 10,000
pairs of normal log tails and 89x for 1,000 twenty-category goodness-of-fit tests.
These compare one batched call with repeated scalar calls to the same Python
API, not with original Fortran throughput. Both use compiled NumPy/SciPy kernels;
the batched version also amortizes Python dispatch and input validation.
Timing results are machine-specific and are not enforced as CI thresholds.
