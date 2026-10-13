# WFMM interval-boundary investigation (not accepted implementation)

`tools/reference_wfmm_interval_audit.py` uses the same checksum-pinned external
WFMM1/2/3 executables and synthetic impulse-draw method as the accepted 1D family
references. It records 28 one-level cases: db1–db4, sym2, sym4, coif1; lengths
32/33; extended mode 0/1. The supplied beta draws are unit coefficient vectors,
so multiplying captured analysis A by reconstructed synthesis B directly checks
whether a data impulse survives the workflow. No random output is used as truth.

All 14 reduced-mode cases abort in the original inverse with its assertion that
input-vector and subband element counts match. The 14 extended-mode cases finish
but max(abs(A B - I)) ranges from 0.9882829323838322 to 29.62300044985098. These
are failures of the tested configurations, not passing compatibility references,
and do not establish that every native interval configuration fails. The JSON
retains numerical maps or the final native error for each setting. A second
independent execution reproduces every synthesis map, error outcome and reported
reconstruction error. Two analysis maps (db4/sym4, length 33, extended mode) differ
only in subnormal entries by at most 1.78922783e-315. Byte-identical analysis
regeneration is therefore NOT claimed; the source of those tiny residues has
not been established. The discrepancy is separately retained in the replay audit.

Separate initial analysis-only probes at ANOVA_MOM entry found reduced-mode
coefficient counts 30 for length 32 and 32 for length 33, in these cases. Such
maps cannot recover every input without additional assumptions. This is not
silently repaired by inventing coefficients, and the observed inverse behavior
is not copied into Python. No interval or reduced-mode implementation is added
by the spatial checkpoint. Standard periodic orthogonal and extended
periodic/zero/reflection implementations remain distinct supported contracts.

A candidate independent mathematical source is Cohen, Daubechies and Vial,
*Wavelets on the Interval and Fast Wavelet Transforms*, ACHA 1(1), 54–81 (1993),
DOI 10.1006/acha.1993.1005. The author institution's
[publication record](https://scholars.duke.edu/publication/767482) confirms the
orthonormal interval-basis subject. The HAL full-text endpoint returned an access
denial during this session; the full paper has not been recovered or validated
here. Do not equate an eventual CDV implementation with XWaveLib's advertised
interval option until its conventions and applicable size/level conditions have
been checked. Public-paper recovery remains the next route; no author contact.

A public primary research follow-up is Leland Jameson, NASA CR-191582 / ICASE
93-94, December 1993, [On the Differentiation Matrix for Daubechies-Based
Wavelets on an Interval](https://ntrs.nasa.gov/api/citations/19940019818/downloads/19940019818.pdf).
Its PDF was readable through the web reader; direct workspace download returned
403, so no local PDF/hash is claimed. Section 2 describes orthonormalization of
boundary functions; Sections 4–6 distinguish sampled values from scaling-function
coefficients and discuss quadrature mappings. This distinction must be audited
before equating XWaveLib's finite-vector operations with a CDV construction.
The present impulse discrepancies establish no accepted lossless finite-vector
contract, rather than disproving continuous interval wavelet theory.
