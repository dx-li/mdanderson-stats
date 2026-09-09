# CDFLIB generic sorting

`sort_list` replaces the public `biomath_sort_mod.sort_list` generic from the
pinned CDFLIB90 F95 source. All four native overloads—double, single, integer
and character—are covered, including the optional comparison function.
The overload implementations and their nested swap/comparison helpers are
private; they introduce no additional public entry points.

```python
from mdanderson_stats import sort_list

ascending = sort_list([3, 1, 2])
prefix = sort_list([3, 1, 2, 99], ncol=3)
descending = sort_list([3, 1, 2], a_gt_b=lambda a, b: a < b)
words = sort_list(["beta", "Alpha", "alpha"])
```

The first `ncol` values are sorted; the remaining suffix is preserved. Omitting
`ncol` sorts the whole input, and zero is valid. Python returns an owned immutable
copy instead of modifying the caller's array. Numeric output retains its NumPy
integer or floating dtype, including float32. Integer values are never converted
to float for sorting. A sequence of strings returns an immutable tuple, preserving
full string contents and lengths, including trailing spaces and NUL characters.

Inputs must be one-dimensional and contain either finite real numbers or
exclusively strings. Booleans, complex numbers, mixed numeric/string sequences,
nonfinite numbers and out-of-range prefix lengths are rejected. Python integer
and floating NumPy dtypes extend the native default-integer/single/double choices;
non-numeric object arrays are supported only when all their elements are strings.

## Comparison and stability

The default numeric order is ascending. Default string comparisons pad shorter
strings with spaces, matching Fortran fixed-width character comparisons. This
matters for control characters: `"a\t"` sorts before `"a"`, while `"a"` and
`"a "` compare equal. Unicode strings extend the native character domain using
Python code-point order. No case conversion or whitespace stripping occurs.

An optional `a_gt_b(a,b)` returns a scalar boolean indicating that `a` belongs
after `b`. The callback must define a consistent total ordering of equivalence
classes. Both strict (`>`) and inclusive (`>=`) styles work: comparing in both
directions determines ties, which remain stable. Transitivity is the caller's
responsibility; the implementation does not attempt a quadratic global proof.
Comparator exceptions propagate. Callbacks receive strings padded to the maximum
input width, while returned strings retain their original contents and lengths.

Stability is a deliberate Python guarantee. The archived implementation does not
promise stable ties, and its default non-strict comparator creates a more serious
problem in quicksort partition scans.

## Native defects and evidence

`tools/reference_cdflib_sort.py` compiles the unchanged constants and sorting
modules with bounds checking, using no numerical/source patches. The fixture
retains source and archive hashes, compiler identity, command and driver. Its
64 calls cover all four overloads, ascending and custom descending comparisons,
empty and singleton lists, insertion/partition sizes, duplicates, prefix sorting,
fractional and extreme floating values, blank padding and long strings.

Seven default-comparison calls terminate with an array-bound failure for duplicate
values. Source inspection confirms that the partition scans use `>=` without a
duplicate-safe stopping condition. The corresponding custom strict descending
comparators complete. Python sorts those inputs successfully and preserves their
multiplicities.

Both long-string calls complete but fail permutation preservation. The character
sorter declares a 256-character temporary for swapping and insertion, so longer
input strings are truncated and blank-padded when moved. Python retains all
300 characters in these tests and imposes no such temporary-buffer limit.

The tests compare native successful results after fixed-width padding, explicitly
retain the native failures, and separately verify ordering, permutation, stable
ties, dtype precision, prefix preservation, immutability and callback behavior.

## Performance and remaining scope

The numeric default uses NumPy's compiled stable sort. String sorting uses
Python's stable sorting with blank-padded keys. Custom comparators use Python's
comparison adapter and therefore incur callback overhead; they are not a
vectorized numeric fast path.

[Recorded benchmarks](cdflib-sort-benchmark.json) compare the numeric implementation
with `np.asarray(sorted(array.tolist()), dtype=array.dtype)` at 10,000 and 100,000
values, verifying equal outputs on each repetition. These measurements do not
compare against the archived native executable.

This closes the sorting module's public interface. CDFLIB90 remains partial:
its mathematical helpers, constants, solver state, console interfaces and string
module still require public-contract coverage. In particular, the string module's
`qlex` is a stateful command-language lexer, not merely a lexical comparator.
