# CDFLIB named constants

`mdanderson_stats.cdflib_constants` exposes all 27 parameters from the archived
F95 `biomath_constants_mod`. The 23 real constants are ordinary Python binary64
floats, including the correctly rounded fractions used by the source.

```python
from mdanderson_stats import cdflib_constants as constants

third = constants.third
scale = constants.hundred
```

The namespace contains `zero`, `one`, `two`, `three`, `four`, `five`, `six`,
`seven`, `eight`, `nine`, `ten`, `twelve`, `hundred`, `thousand`, `half`, `third`,
`fourth`, `fifth`, `sixth`, `eighth`, `tenth`, `hundredth` and `thousandth`.

## Legacy identifiers

The four integer values are retained for source compatibility:

| Name | Value | Meaning in the audited source build |
|---|---:|---|
| `dpkind` | 8 | GNU Fortran kind for the 64-bit real model |
| `spkind` | 4 | GNU Fortran kind for the 32-bit real model |
| `stdin` | 5 | Fortran standard-input logical unit |
| `stdout` | 6 | Fortran standard-output logical unit |

Kind numbers are compiler-specific opaque identifiers. They are not NumPy dtype
arguments, and the package does not use them to select numerical precision.
Numerical APIs continue to use float64. Likewise, the logical units are not
operating-system file descriptors or Python stream objects. Importing this module
does not open, replace or write to any stream. The constants are namespaced rather
than adding generic names such as `one` or `stdin` to the package root.

## Validation and source coverage

The [reference generator](../tools/reference_cdflib_constants.py) verifies the
pinned archive hash, extracts the original module bytes and compiles them without
editing the source. It records every named parameter and the storage width,
radix, significand digits, exponent limits, epsilon, smallest normal and largest
finite value for both real kinds. Compiler flags and source/driver hashes are
included in the [fixture](../tests/fixtures/cdflib_constants.json).

[Tests](../tests/test_cdflib_constants.py) compare all public values with those
native records, including exact binary64 encodings, and verify the recorded
floating models against NumPy's representations. Constants need no numerical
iteration or performance benchmark.

Together with the 35 mathematical procedures, this covers the constants and
mathematical support modules with documented Python semantics. The root-finder,
console and distribution-adapter support interfaces remain open, as do remaining
legacy helper contracts. CDFLIB90 and the full catalog conversion remain partial.
