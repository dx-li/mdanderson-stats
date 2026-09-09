# CDFLIB numeric array formatting

`format_cdflib_array(values, format_spec)` renders a finite, one-dimensional
numeric array. `CDFConsole.write_array` writes the same result to its output and
optional report stream, returning the record without its terminating newline.

```python
from io import StringIO
from mdanderson_stats import CDFConsole, format_cdflib_array

text = format_cdflib_array([1, 2, 3], "2F11.6 3X E11.3")
assert len(text) == 36
output, report = StringIO(), StringIO()
console = CDFConsole(StringIO(), output, report_stream=report)
assert console.write_array([1, 2, 3], "2F11.6 3X E11.3") == text
assert output.getvalue() == report.getvalue() == text + "\n"
```

Formats contain `rFw.d`, `rEw.d` and `nX` fields, separated by whitespace or commas.
Repetition defaults to one; widths and repetition/space counts must be positive.
Letters are case-insensitive, and one surrounding pair of parentheses is allowed.
Numeric field counts must sum to the input length. `nX` adds exactly `n` spaces.
Nested groups and other Fortran descriptors are rejected explicitly.

`F` preserves the source's adaptive fixed/scientific display. Absolute values
below `0.0010000000474974513` or above 1000 use scientific notation. The lower
threshold is the archived default-REAL `1e-3` literal rounded to binary32, then
promoted for comparison with the binary64 input; it is slightly larger than the
Python binary64 literal `0.001`. Otherwise fixed-point formatting is used.
When an F field switches to scientific notation, precision greater than four is
reduced by three, exactly as in the source.

`E` always uses scientific notation with the requested precision. This implements
the source's advertised E-field support, which its actual parser rejects. Scientific
precision must be at least one. An F field with precision zero is valid for fixed
output, but raises `ValueError` if its value requires the scientific path, replacing
the corresponding native runtime error. Fixed-point zero precision retains the
Fortran decimal point.

Scientific output uses the native `0.dddE±ee` convention. With a three-digit
exponent, the `E` is omitted, as in the source's default E descriptor; for example,
`0.100+309`. Signed zero is retained. Numeric rounding uses Python's binary64
formatting, checked against native examples and independent Decimal error bounds.
If a field is too narrow, its leading zero before the decimal point may be omitted;
if it still does not fit, it contains exactly `w` asterisks, matching Fortran overflow
markers. Extremely large precision requests that cannot fit do not allocate giant
intermediate strings.

The result has exactly the requested total field width, including explicit trailing
spaces. Python does not add the native list-directed leading blank or pad to the
source's fixed 79-character buffer. Longer records are supported without truncation
or out-of-bounds writes. `max_output=1000000` bounds the total record length before
rendering; callers may supply a different positive integer limit.

All input and format validation completes before `write_array` writes either stream.
Inputs are not modified. Output/report streams remain caller-owned, and using the
same stream for both writes only once. Stream failures propagate; no cross-stream
transaction is promised. Nonfinite values, incorrect field counts and unsupported
formats raise `ValueError`.

The [reference generator](../tools/reference_cdflib_array_format.py) compiles four
unchanged modules from the verified archive. The [20 native cases](../tests/fixtures/cdflib_array_format.json)
record source/driver hashes, compiler settings, return codes, output and report
routing. They cover ordinary fields, rounding ties, spacing, threshold boundaries,
signed zero, extreme exponents, narrow fields and known native failures. Runtime
addresses and temporary build paths are normalized.

Tests compare every successful native array record, classify and repair the native
E/lowercase-field rejection and fixed-buffer failure, and verify decimal rounding
independently. They also exercise output ownership, explicit spacing, overflow
markers, output limits and rejection before writing. Formatting uses Python's
compiled numeric conversion and a single final string join; no speedup over native
Fortran I/O is claimed.

The same fixture records a repaired message-control defect: `unit_only=False`
suppresses native console output merely because the argument is present. The
[message controls](cdflib-message-format.md) now honor the boolean value. CDFLIB90
and the overall catalog conversion remain partial.
