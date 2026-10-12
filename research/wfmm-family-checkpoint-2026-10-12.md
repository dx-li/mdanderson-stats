# WFMM named Symlet and Coiflet families

Base checkpoint 38b1cbf7fdc6af927f81bac8405d19e61a8fca1c. This adds the guide's
sym2–sym8 and coif1–coif5 extended 1D families, alongside existing db1–db10.
WFMM remains partial; 103 implemented / 31 partial / 4 pending are unchanged.

The checksum-pinned original WFMM1/2/3 reference generator captures 72 analysis/
synthesis maps: 12 named families, periodic/zero/endpoint-repeating reflection,
even length 64 at one level and odd length 65 at two levels. Synthetic impulse
beta draws replace native MCMC output before inverse reconstruction. No original
executable, library, example observation or source code enters the artifact.
Source archive provenance remains the boundary checkpoint's official WFMM 3.1
Linux bundle and hashes.

The production filter table consists of factual numerical impulse responses:
for a one-level zero-padded analysis with filter length L, column (L-2)/2 has
unwrapped support beginning at input zero and records its L low-pass taps.
The generator's optional --filters-output independently recreates this table,
including the native fixture hash. High-pass taps follow the quadrature-mirror
identity. The transform loops remain the independently implemented cropped
analysis/synthesis rules, reused for every family. Aliases follow the manual;
in particular d12 maps to db6 despite the manual table's apparent d6 typo.

Independent tests verify unit norm, scaling sum sqrt(2), quadrature-mirror
identity, even-shift orthogonality and vanishing moments (Symlet order n;
Coiflet order 2n), with normalized moment coordinates to avoid power overflow.
Original analysis maps agree exactly in the current environment; original
synthesis differs by at most 2.22e-16. Native coif5 taps are approximate: the
largest two-level reconstruction discrepancy is 6.843e-9. Independent short
signals through 12 levels reach about 2.014e-8. Tests use an explicit 5e-8
reconstruction tolerance, not a claim of machine-precision orthogonality or
exact arithmetic. Other independent filter identities have separately stated
tolerances in the tests. No conditioning or clinical-validity claim follows.

The named constructor returns immutable rectangular matrix pairs and canonical
wavelet_name metadata. Saved studies accept wavelet_name only with the extended
transform and default length control; historical version-1 files reopen with
wavelet_name=None. Legacy orthogonal and extended db constructors are unchanged.
Existing bounded work and matrix-storage rules still apply.

Advertised interval/reduced-boundary and multidimensional transforms remain
open, as do hyperparameter/sampler contracts. Native report defects remain separate from scientific Python summaries.
The conditional-U variance discrepancy is now explained by the native integer
Abs negligibility check: 472 of 640 vectors return their conditional mean;
168 draw normally. The covariance factor and sampled normal innovations were
independently checked. See wfmm-report-edges-audit-2026-10-12.md.
The scientific Python conditional sampler is unchanged. Notices, license restrictions, packaging and permissions are unchanged.
Final acceptance is recorded in the adjacent validation manifest.
