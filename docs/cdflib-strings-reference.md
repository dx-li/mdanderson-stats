# CDFLIB string and lexer reference audit

This audit covers the five public names of `biomath_strings_mod`:
`lower_case_char`, `upper_case_char`, `lower_case_string`, `upper_case_string`
and `qlex`. The module is private by default. Its character tables and the
nested lexer helpers are implementation details, not additional public APIs.
The Python port remains pending; no lexer functionality is claimed by this audit.

## Evidence

`tools/reference_cdflib_strings.py` compiles unchanged constants and string
modules with bounds checking and no source adaptations. The fixture records the
archive and source hashes, compiler identity, command, driver, inputs, outcomes
and outputs. Each lexical input runs in its own process with a three-second
timeout. Driver inputs and token strings use integer character codes, preserving
NUL, control and high-byte characters without text-encoding ambiguity.

The case-conversion evidence covers all 256 byte values individually and five
whole strings, including empty, long, control-character and trailing-space inputs.
The 170 lexical calls comprise 23 ordinary command strings, nine numeric edge
cases, six exact-buffer cases, four byte/output-buffer cases and all 128 single
ASCII values. Of these lexical calls, 160 complete, nine fail with checked bounds
errors and one times out. The 413 tests check independent ASCII mappings,
character classes, token grammar, decimal numeric values and explicit native
failure behavior. No archived numerical source or executable is bundled.

Ordinary lexer inputs have two extra trailing blanks and an output buffer at
least as long as the input. These are caller-owned buffers, not source patches.
Separate exact-buffer and undersized-output cases expose what happens without
that extra storage. The native procedure's output arguments are undefined when
it returns false, so only its return and reset flag are recorded at termination.

## Case conversion

All four conversion procedures change only ASCII A–Z or a–z. Other bytes are
preserved, including high bytes, NUL and controls. Whole-string conversion
preserves length and trailing spaces; it modifies the caller's string. Character
functions accept one character. Python's Unicode-wide `lower()`/`upper()` alone
would not preserve this contract: an eventual port needs ASCII-only translation
and an explicit return-value replacement for Fortran's in-place strings.

## Stateful lexer contract

`qlex` returns one token at a time. A true `qstart` resets the saved position and
previous-character state; every call sets it false. The routine saves `posn`,
`lenstr`, `prvchr` and `prvtyp` globally, so independent streams cannot safely
interleave without resetting. A Python replacement should isolate stream state.

The eight token types are:

| Type | Meaning |
|---|---|
| IN | Integer |
| RL | Real number |
| ID | Identifier beginning with an ASCII letter; later letters/digits/period/underscore/dollar |
| QS | Quoted string |
| DL | One of `()[]{},:;` |
| OP | Operator sequence from `+-*/<>=` |
| OS | Recognized characters that do not form a valid token |
| UC | Unrecognized character sequence |

Only the ASCII space is the ordinary token separator. Tabs, newlines, NUL and
other controls form UC tokens outside quotes. Names, numbers and delimiters can
abut. Numeric signs may be separated from digits by spaces, which are omitted
from the returned numeric text. Both E/e and D/d exponent notation occur in
successful fixtures, including integer mantissas such as `1e3`.

The documented sign rule is not the executable rule. The header describes unary
signs based on the preceding token or a following digit. The executable instead
uses the saved preceding character and its state machine. For example, `a+2`
produces ID `a`, OP `+`, IN `2`, whereas `a + 2` produces ID `a`, IN `+2`.
`x=-4` groups `=-` as an operator, and the input `+ -` can combine the two signs
into one operator token despite the intervening blank. The port must explicitly
choose and document its corrected token policy rather than infer it from the
header alone.

## Independently established defects

**Doubled quotes lose data.** The documented rule says `'don''t'` should yield
`don't`; native output is `don''`. Likewise, `"a""b"` becomes `a""`. Source
inspection shows that skipping a doubled delimiter advances the saved position
without refreshing the already-read lookahead character.

**End-of-input can emit a spurious token.** After a final quoted token, the
recorded build emits an additional empty OS token. The procedure does not
initialize `curtyp` at the start of each call, yet inspects it after exhausting
the input. This involves undefined/stale output state and must not become a
portable Python contract. An unmatched quote also appends a synthetic trailing
blank to its returned OS text.

**Malformed exponents become numeric zero.** Inputs `1e`, `1e+` and `1e-` are
reported as RL tokens with value zero instead of malformed numeric text. The
tests independently reject these strings as floating-point numbers.

**Overflow flags differ from the header.** The observed default integer is
signed 32-bit. Although -2147483648 fits, the native magnitude-before-sign check
flags integer overflow and returns an integer placeholder of zero. Decimal
integer accumulation uses double precision, so 9007199254740993 loses a unit
in its returned real value. The source's `rmax` is also double-kind, so 1e39
receives flag 1, not a single-precision overflow flag. Exponent overflow at
1e309 returns an undocumented flag 3 and real placeholder zero. A 310-digit
integer instead returns infinity with flag 2. Defined values and placeholders
must be distinguished in a Python result model.

**Exponent work is unbounded in the exponent value.** `1e-1000000000` times out:
the source keeps multiplying by 0.1 after the value has underflowed. Independent
decimal conversion establishes a float result of zero without a billion-step
loop. Even ordinary repeated scaling accumulates rounding error, visible at
1e-300; representable subnormals and rounded-zero values are retained in evidence.

**Buffer and byte bounds are unsafe.** Exact-length inputs such as `abc`, `123`
and `'x'`, and the zero-length input, read past the input buffer. A one-character
output buffer fails while reading `abc`. The classifier's table covers only
0–127; bytes 128 and 255 fail, including a high byte inside a quoted string,
despite documentation allowing arbitrary quoted characters. Python must validate
or define the extended character domain and allocate token storage safely.

## Remaining work

Implement the four ASCII conversion operations and a reentrant lexer with
explicit choices for sign grouping, malformed tokens, quoted-string decoding,
integer/real representation and overflow reporting. Validate those choices
against the applicable native cases and independent intended-token examples;
retain the native defects as regression evidence rather than reproducing silent
data loss or unsafe buffer access. CDFLIB90 remains partial.
