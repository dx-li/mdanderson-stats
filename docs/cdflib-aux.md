# CDFLIB distribution metadata and validation adapters

`mdanderson_stats.cdflib_aux` exposes the historical F95 `cdf_aux_mod` support
interfaces. The namespace contains all 13 distribution descriptors (including
`the_dummy_binomial`), immutable `CDFDistribution`/`CDFParameter` records, range
and complement helpers, batch parameter validation and root-search adapters.

These are archive descriptors. Their bounds and `max_which` values are preserved,
including `the_f.max_which == 2`, `the_non_central_f.max_which == 3`, and the dummy
binomial's zero maximum. They do not describe every inverse or extended numerical
domain supported by the package's distribution functions. Existing distribution
APIs continue to use their own documented contracts.

```python
from mdanderson_stats import cdflib_aux as aux

status = aux.validate_parameters(
    aux.the_beta,
    which=2,
    params=[[0.5, 0.5, 0, 0, 2, 3], [0.5, 0.5, 0, 0, -1, 3]],
)
assert status.tolist() == [0, -6]
```

Parameter rows have shape `(..., 6)`, including unused padding. `which` is one
scalar inverse selector. Validation returns an immutable int64 array over the
leading dimensions; a single row returns a zero-dimensional array. The first
error wins, in source order:

1. A selector outside `[1, max_which]` returns `-1`.
2. For selectors other than 1, cumulative and complementary probabilities must
   sum to one within three binary64 epsilons; failure returns `+3`.
3. Each checked parameter must be finite and within its inclusive bounds. Failure
   at one-based slot `i` returns `-(i+1)`.
4. Success returns `0` explicitly. The native code leaves success status undefined.

Slots whose `no_check` matches `which`, and slots beyond `nparam`, are ignored;
they may contain NaN placeholders for unknown values. Nonfinite checked values
fail validation. Batch operations use NumPy over all rows, with at most six
parameter passes. Invalid array shape or a noninteger selector raises `ValueError`.

`add_to_one` performs only the three-epsilon sum check. For example, `(-1,2)`
passes; callers must separately check probability ranges. `dbl_in_range` supports
broadcasting and inclusive bounds. Both helpers reject nonfinite inputs and
return immutable boolean arrays. `int_in_range` compares scalar Python integers
without rounding through floating point; generic `in_range` selects that path
when all three arguments are Python integers, and the floating array path otherwise.
Inverted bounds are rejected.

`check_complements(x=None, y=None, set_values=True)` requires at least one value.
With one coordinate it computes the other as `1-value`; with both it preserves
both supplied coordinates. It does not check range or sum, matching the source's
presence/completion role. Returned arrays broadcast and own immutable copies.
With `set_values=False`, only presence is checked and the result is `None`.
Missing values raise `ValueError`, replacing native optional-status/STOP handling.

`cdf_set_zero_finder(distrib, which)` uses a one-based **parameter slot**, not the
inverse selector. It configures that slot's bounds with source settings: absolute
and relative steps 0.5, multiplier 5, absolute tolerance 1e-50 and relative tolerance
1e-8. It returns an independent [root state](cdflib-root.md), with the root API's
4096-evaluation default cap. Inactive or out-of-range slots are rejected.

```python
from mdanderson_stats import interval_zf

state = aux.cdf_set_zero_finder(aux.the_beta, which=3)  # x, bounded by [0,1]
result = interval_zf(lambda x: x, y=0.25, local=state)
assert result.x == 0.25
assert aux.cdf_finalize_status(state) == 0
```

`cdf_finalize_status` returns 0 on success, -50 for a left-bound failure and +50
for a right-bound failure. Unstarted/pending searches raise `ValueError`;
evaluation exhaustion raises `ArithmeticError`. They are not mislabeled as bound
failures. `which_miss` raises a missing-argument `ValueError` with routine, selector
and argument context instead of terminating the process.

The source types `one_parameter` and `the_distribution` map to `CDFParameter` and
`CDFDistribution`. Descriptor names are available directly in this namespace and
through read-only `DISTRIBUTIONS`. String labels are trimmed of Fortran padding;
all six parameter slots, numeric bounds and selectors are retained.

The [native fixture](../tests/fixtures/cdflib_aux.json) was produced by compiling
six unchanged source modules from the verified archive, including dependencies.
Its [generator](../tools/reference_cdflib_aux.py) records source/driver hashes and
compiler flags. It captures 13 complete descriptors, 26 defined validation failures,
six floating-range outcomes and five sum-check outcomes. Undefined success status
is represented by null, never treated as meaningful. Native `dbl_in_range` accepts
NaN; Python deliberately rejects it. Tests compare every descriptor and recorded
outcome, then check inclusive boundaries, failure precedence, ignored unknowns,
batch ownership, root adapters and protocol errors independently.

The [benchmark](cdflib-aux-benchmark.json) compares batched validation against
repeated scalar calls to the same Python API, verifying identical statuses. On
the recorded machine, 256 rows were about 127× faster and 10,000 rows about 1,066×
faster. These are not comparisons with native Fortran timing. Reproduce with
`uv run python tools/benchmark_cdflib_aux.py`.

CDFLIB90 remains partial: console/message support and remaining legacy interfaces
still need implementation and validation.
