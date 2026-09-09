# CDFLIB strings and lexical analysis

The five public operations of `biomath_strings_mod` are implemented:
`lower_case_char`, `upper_case_char`, `lower_case_string`, `upper_case_string`
and `qlex`. The [unchanged native audit](cdflib-strings-reference.md) records
applicable behavior and the defects repaired by this port.

```python
from mdanderson_stats import lower_case_string, qlex

assert lower_case_string("ABC ß") == "abc ß"
tokens = list(qlex("x=-1.25; 'don''t'"))
assert [(t.kind, t.text) for t in tokens] == [
    ("ID", "x"),
    ("OP", "="),
    ("RL", "-1.25"),
    ("DL", ";"),
    ("QS", "don't"),
]
```

## Case conversion

Case conversion changes only ASCII A–Z/a–z. Other Unicode characters, NUL,
controls, whitespace and string length are preserved. These functions return
new immutable strings, replacing the source's in-place whole-string procedures.
The character functions require exactly one Unicode character. Inputs must be
Python strings; bytes and automatic conversion from other objects are rejected.
Translation uses Python's compiled `str.translate`, without per-character Python
case conversion or a NumPy object-array conversion.

## Token streams

`qlex(text)` returns an iterator of immutable `QlexToken` objects. Each iterator
owns its scanning state: streams can interleave, and another call restarts from
the beginning. There is no global saved position or `qstart` flag. End-of-input
exhausts the iterator, with no extra token or synthetic trailing blank.

Tokens contain `kind`, decoded `text`, zero-based `start`/`stop` source offsets,
`integer`, `real`, `overflow` and `underflow`. `length` is the decoded text length.
The source span includes quotes and spaces between a unary sign and its number,
while decoded text removes those syntactic characters. No user-sized token buffer
is needed. All scanning/storage work depends on input length, not numeric exponent
magnitude. Python's finite memory remains the practical input-size limit.

The eight kinds remain IN (integer), RL (real), ID (identifier), QS (quoted
string), DL (delimiter), OP (operator), OS (malformed recognized text) and UC
(unrecognized characters). Non-numeric tokens have both numeric fields `None`.
Only ASCII space separates ordinary tokens; tabs, newlines and NUL outside quotes
are UC characters. Outside quotes, non-ASCII characters also form UC tokens;
inside quotes, all Unicode characters are preserved. Identifiers start with an
ASCII letter and continue with ASCII letters, digits, period, underscore or dollar.
Delimiters are `()[]{},:;`, and operators use `+-*/<>=`.

Quoted strings use either quote style. A doubled matching delimiter produces one
literal quote. Unterminated strings produce OS with the content read so far,
without inventing a trailing blank. Missing exponent digits and invalid numeric
forms produce OS with no numeric value; `1e` no longer becomes numeric zero.
Numbers accept decimal points and E/e/D/d exponents, including `1e3` and `1D-3`.
Recognized but malformed unquoted strings, such as `123abc`, remain OS tokens.

## Explicit sign policy

The native header and executable disagree about signs. Python uses a consistent
policy that retains common valid forms and repairs source grouping defects:

- A sign starts a numeric token at the beginning, after a space, or after an
  operator/delimiter, provided a numeric mantissa follows. Spaces after that sign
  are allowed and removed from decoded text.
- After an identifier/number without an intervening space, a sign is an operator.
  Thus `a+2` gives ID `a`, OP `+`, IN `2`; `a + 2` gives ID `a`, IN `+2`.
- Operator sequences end before a following numeric sign. `x=-4` gives ID `x`,
  OP `=`, IN `-4`, instead of grouping `=-` as one operator.
- Operator sequences never cross spaces. `+ -` yields two OP tokens.

This is a documented Python policy, not a claim of token-for-token compatibility
with the source's stale previous-character state. The native fixtures retain
those differences explicitly.

## Numeric results

`real` is correctly rounded Python binary64 conversion of the numeric text,
with D/d exponent spelling normalized. `integer` truncates that finite real
toward zero when the result fits signed int32. The signed minimum -2147483648
is accepted. Integer tokens outside int32 retain their original decimal text,
but their real field can lose integer precision beyond binary64's exact range.
For real tokens near an integer boundary, truncation follows the rounded real,
not an unrounded arbitrary-precision decimal value.

Overflow codes have defined Python meanings: 0 means available numeric results
fit; 1 means the integer is unavailable but the real is finite; 2 means the real
is unavailable, and both numeric fields are `None`. The native source's code 3
and zero/infinity placeholders are replaced by code 2. Non-numeric tokens use
code 0 and have no numeric fields. `underflow` is true when a nonzero decimal
mantissa rounds to real zero; representable subnormals remain nonzero and are
not marked as underflow. Signed zero is retained in `real`.

Numeric conversion does not repeatedly multiply by powers of ten. Inputs such
as `1e-1000000000` resolve to zero with `underflow=True`; huge positive exponents
produce explicit overflow. No arbitrary-size integer is constructed from the
input text, so very long digit strings do not invoke Python's integer-string
conversion limit or allocate storage proportional to an exponent value.

## Validation and performance

Tests cover all 256 native byte case mappings, whole-string conversion, all 170
native lexical inputs with explicit repair expectations, independent intended
tokens, interleaved iterators, source spans, immutability, long inputs, malformed
numbers, Unicode, buffer-boundary repairs and numeric extremes. The separate
native reference tests continue to preserve the original failures.

[Throughput measurements](cdflib-strings-benchmark.json) record validated complete
lexing and case conversion at several sizes. Regex matching and numeric/string
conversion use compiled standard-library operations; stateful scanning remains
Python code. Measurements include token construction/materialization and make
no claim of a speedup over the archived native implementation.

The existing MULTI lexer has a different grammar and historical numerical
semantics; it is unchanged. This closes the CDFLIB string module's public scope.
See the [CDFLIB90 completion audit](cdflib90-completion.md) for the
complete library scope.
