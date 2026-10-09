# WFMM native energy compression

Five synthetic uncompressed Haar transforms and 46 native compression/bypass/
error runs were recovered from the checksum-verified WFMM 3.1 Linux bundle.
`tools/reference_wfmm_compression.py` generates MATLAB inputs, runs the original
WFMM1 executable with explicit `db1`, `periodic`, `extended_mode=1` and selected
decomposition levels, then reads native D and DIndex without computing expected
compression decisions. The same archive and executable hashes as the
[variance-prior recovery](wfmm-native-prior-audit.md) are recorded in the fixture.
There are 44 successful native runs and two explicit zero-column failures.

## Verified rule

Native `WFMM::CompressD` at VA `0x491460` squares each coefficient, sorts row
energies in descending order and computes cumulative energy **including** the
current coefficient. After normalization, a coefficient qualifies when that
cumulative fraction is **strictly below** `alphawav`. Columns are retained when
the number of qualifying curves is **strictly greater** than `t`. The helper
calls and `WaveSpecs` debug member offsets agree with the executable outputs.
Zero-energy curves receive normalized fractions of one and cast no votes.
At `alphawav=1`, the transform bypasses compression entirely; `t` has no effect.

This excludes the crossing coefficient, unlike common minimum-energy retention
rules. Actual energy can fall below alpha, especially with larger t. Python's
`wfmm_compress_coefficients` reproduces these strict comparisons and reports
the actual retained energy fraction for every curve. Results compose with
existing selection/zero-filled restoration and coefficient-model APIs.

Native sorting does not define stable ties. A structured 16-column fixture
splits equal-energy coefficients at alpha .95; the native insertion-sort result
matches smaller-index precedence in that example. It does not establish the
same precedence for larger C++ introsort problems. Python therefore rejects
split ties by default, and exposes `original_index` as an explicit alternative.
Native empty selections fail in matrix resizing; Python gives a clear error.
Per-curve rescaling avoids energy overflow/underflow and does not claim binary
floating-point equality at every threshold. Input is limited to 2,000,000 cells;
sorting temporaries cover one curve at a time.

## Transform scope

The existing Python periodic Haar packing matches all five native full D
matrices: 16 points at one/two levels, 32 points at two/three levels and 64
points at four levels. References use `extended_mode=1`. This does not resolve
the native truncated `extended_mode=0` convention or establish other wavelet
parity. Additional probes demonstrate that native periodic db2 at 64 points
and two levels retains 69 coordinates (Kj=[18,18,33]); db4 retains 77
(Kj=[21,21,35]). Existing orthogonal Python periodic transforms retain 64.
Those extended/native transforms remain a separate gap, not a coefficient
permutation. No successful PC reference was recovered.

Seventy-nine focused tests pass, covering all five native transforms and all
46 native compression/error cases, exact strict cutoffs, curve rather than
pooled energy, ties, bypass, zero rows, scale invariance at 1e±300, immutable
metadata, restoration and preflight limits.

```bash
uv run --locked --extra plot python tools/reference_wfmm_compression.py \
  --archive research/raw/source-recovery-2026-10-08/wfmm_v3_1_linux_x86_64.tar.gz \
  --check
```

Original native programs/libraries remain ignored local research inputs.
MOM/profile initialization, automatic proposals, extra transforms/extensions,
pass filtering and native file workflows remain open; WFMM stays partial.
