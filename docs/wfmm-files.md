# WFMM numerical file interchange

`wfmm_encode_draws(values)` returns headerless little-endian double bytes from
samples × components × coordinates. `wfmm_decode_draws(payload, draws=...,
components=..., coordinates=...)` requires the exact dimensions and byte count.
Coordinates vary fastest; all components for one sample stay together. This
layout is independently checked against pinned Linux x86_64 WFMM3 output with
one/two fixed effects and one/two concatenated chains. The guide's `wbeta`,
`beta`, variance and random-effect blocks use this ordering, but the codec does
not infer a variable's meaning, units or basis from bytes.

The little-endian default is the verified platform profile. Explicit big-endian
conversion is available but does not establish another native-platform match.
Files contain no dimensions, labels, chain boundaries, burn-in or thinning
metadata. Wrong dimensions of the same product, or wrong endianness yielding
finite numbers, cannot be detected. Keep authoritative study metadata separately.
Each block is limited to two million cells; nonfinite, truncated and trailing
values are rejected. Outputs are owned read-only arrays. Encode individual
chains or concatenate retained samples explicitly. The functions return bytes
and leave path selection and writing to the caller.

`wfmm_summary_mat(summary, prefix="Lbeta")` returns a MATLAB-v5 report containing
selected native-style names. Prefixes `LbetaLt` and `U` are also supported.
Means and sample SDs have effects × coordinates shape. Quantile/effect-size
rows pack effects with coordinates fastest; simultaneous bounds have one row
for the summary's confidence level. Metadata includes quantiles, effect sizes,
alpha and an explicit scientific-summary profile.

The report uses the Python scientific summaries. It does not reproduce the
audited native quantile, integer-absolute-value or zero-SD defects. It is not a
native initialization file, full native report clone, or evidence of clinical
validity. Saved Python studies remain the reproducible complete Python input
contract. No original binaries are needed by these APIs.
