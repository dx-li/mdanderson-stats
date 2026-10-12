# WFMM explicit conditional variance sampling

Recovery verified remote feature head `071f2694c8fd4b3b88409c9fef58b3779d028c4b`.
The predecessor's completed observation-only handoff reports 76,423 full tests
in 455.17 seconds, with all 774 production Python files matching that commit.
This is transferred acceptance evidence, not a rerun in the replacement executor.
Single-writer ownership was then explicitly transferred before implementation.

The private PCA Library ZIP resolved to
`mdanderson-stats-wfmm-pca-transfer-071f2694-2026-10-12.zip`, but the supported
download failed both initially and after one fresh preparation. No ZIP bytes
arrived. Its three untracked files and seven provisional sign-convention
failures are not integrated or represented as passing here.

## Source evidence and implementation

The official Linux archive was downloaded again and matches SHA-256
`9bc5003271eebfac4a90e265d08c8f677d72252eccae5ff65c90bc25d5d309f9`.
Original executables remain under ignored `research/raw/wfmm-recovery/native`.
The existing six initialization/proposal reference files regenerate byte-for-byte.

`tools/reference_wfmm_frozen.py` runs original WFMM1 and WFMM2 on synthetic
periodic Haar studies with proposal multipliers .5, 1, 1.5 and 4. Each saves
32 draws after eight burn-in iterations. Eight random variance components with
zero proposal scales retain their initial 1e-6 values exactly, giving 1,024
recorded frozen values. The fixture deliberately excludes active stochastic
draws. Regeneration is byte-identical; RNG stream identity is not tested.
These cases establish the floor-component behavior only, not all possible
native zero-scale inputs or the complete native Metropolis update.

`fit_wfmm_coefficients(..., zero_proposal_policy="freeze")` explicitly conditions
on the supplied positive starting variances wherever proposal SD is zero.
Negative/nonfinite scales remain errors. The default `"error"` policy still
rejects zeros. Frozen components consume no Python proposal random draws.
Variance priors remain required under `estimate_variances=True`.
Immutable `variance_proposals` counts attempted updates across chains including
warmup. Frozen acceptance rates and counts are zero, not evidence of poor mixing.
The additional result field defaults to None for backwards-compatible direct
construction; fitter-generated results always populate it.

Independent tests compare an active residual variance against its conjugate
inverse-gamma posterior while another component is fixed above the floor.
All-frozen sampling exactly matches the existing fixed-variance conditional
sampler under the same Python seed. Native reference cases verify the explicit
floor freezes. This does not remove the initializer's signed-stop, restricted
optimizer or zero-proposal readiness holds automatically.

## Scope and next gates

Entry70 remains partial; counts remain 103 implemented / 31 partial / 4 pending.
PCA/hybrid and other native transform/boundary contracts, saved whole studies,
postprocessing/prediction/report orchestration and optional native hyperparameter
updates remain open. Scientific model accuracy, source compatibility, RNG
identity, posterior precision and clinical validity remain separate claims.

The pending exact fits, BLESS/ComPAS remaining deployed contracts, mixed-license
distribution decision and prior Forbidden PR creation remain unresolved. No
notices, permissions, packaging split, merge, release, deployment or outreach
were changed. Final validation is recorded separately after completion.

## Independent PCA lead while transfer is blocked

A replacement-executor probe used the `paperless16` synthetic matrix already
defined in `tools/reference_wfmm_compression.py`, alphaPC=.9, periodic db1 and
two levels. Original WFMM1 returned two PC coordinates, eighteen PCw coordinates
and two wPC coordinates. Using the returned native phi/phi_inv, direct PC
multiplication and PCw concatenation of PC scores plus Python-transformed
residuals agree within 1.8e-15; Python wavelet analysis followed by the returned
wPC matrix agrees within 1.1e-14. Flipping the Python Haar detail signs instead
produces a 6.4 discrepancy for PCw. Thus the existing Python Haar convention
matches this source case. The inaccessible provisional test formula is a likely
lead, not a verified diagnosis of its seven failures. Probe inputs/outputs are
local scratch at /tmp/wfmm-independent-pca; no PCA implementation or acceptance
is claimed from this observation.

## Completed validation

The full all-extras warnings-as-errors suite passed **76,435 tests in 453.62s**
with four workers. All774 production Python files stayed unchanged. Focused
WFMM tests303 pass; the fresh installed wheel passes331 tests (303 WFMM plus
28 plotting/TIFF checks), both initializer examples and dependency checks.
Lint, format, mypy774, builds,778 package Python/JSON source-byte comparisons,
29 retained notice/license files and original-binary exclusions pass. Six
validation tests were strengthened with valid draw counts during the run; the
corrected source12-test suite and installed six-case validation subset pass
separately. See the adjacent validation JSON for exact evidence and artifact
hashes. Artifact source equality is asserted; final evidence documents were
written after those artifacts were built.
