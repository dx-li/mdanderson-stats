# WFMM extended image and volume transforms

Base: 369d3f80b0cb004384b9a2b62c339dbc0514002b. WFMM remains partial;
103 implemented / 31 partial / 4 pending are unchanged.

The guide advertises column-stacked 2D/3D transforms and a rectangular 2D mode
with independent axis wavelet, depth and boundary controls. The implementation
now composes the validated extended 1D pairs into joint-approximation pyramids
and complete rectangular axis transforms. The first spatial index varies
fastest. Two-dimensional orientation bits are 0,1,2,3; the native three-dimensional
ordering is 0,2,1,4,3,6,5,7, with axis one represented by the least-significant
bit. Pyramid subbands are coarse approximation, coarse-to-fine detail groups.
Rectangular partition pairs sort by (maximum partition index, first index,
second index), and each tile is column-stacked. Each contiguous subband is an
explicit shrinkage partition; this does not claim native hyperparameter-file
compatibility. Independent Haar block sums/differences check spatial meaning.

## Qualified source evidence

52 synthetic identity-input analysis maps cover 21 rectangular cases and 31
pyramid cases, odd/even dimensions, separate axis depths/families/boundaries,
three boundary modes, and one/two levels including an 8×8×8 Haar volume.
The maximum analysis discrepancy is below 1.6e-15. Synthesis composes validated
1D inverse maps, with independent identity and arbitrary-signal reconstruction
checks (3e-11 tolerance; Coiflet tap precision is inherited). Native spatial
inverse-file parity is NOT established by these checks.

Original WFMM1 can compute these analysis maps before its initialization stage
fails on several synthetic inputs. `tools/reference_wfmm_spatial.py` verifies
WFMM1 SHA-256, launches only its own Linux x86_64 child, stops at ANOVA_MOM entry
0x4870e0, and reads its transformed synthetic identity matrix through the first
argument. The child is terminated after capture. The matrix layout is data
pointer +8, row count +16, column count +20, stored row-major doubles. No fitting
or postprocessing occurs in this capture. The 52-case fixture regenerates
byte-identically. Only numerical observations enter the fixture, not binary or
original code bytes. Official archive provenance is retained in earlier audits.

`wfmm-spatial-native-failures-2026-10-12.json` preserves 22 failed end-to-end
attempts separately: 18 rectangular WFMM3 attempts fail while resizing a matrix
to 1×0; four pyramid WFMM1 attempts flag invalid floating-point initialization.
Those failed workflows are not counted as passed references. No inference that
all original datasets fail is made. Analysis capture and successful Python
reconstruction do not remove the native workflow qualification.

## Saved studies and bounds

Saved `spatial_wavelet` studies store shape/layout and optional axis controls,
reject conflicting scalar settings, and retain historical version-1 defaults.
They compose initialization, sampling, compression, covariance, prediction and
reports on the original flattened grid. An explicit convergence-hold regression
confirms no MCMC runs when the initializer is not ready. The example is numerical,
not a clinical application. Spatial pass filtering is still rejected pending its
source convention audit. Dense storage is bounded to 512 inputs/two million
estimated live cells, with a work cap and at least two samples on each decomposed
axis. This is a bounded implementation, not unrestricted image-size support.

Interval/reduced boundaries, spatial pass filtering and remaining sampler
contracts remain open. The guide's optional pi/tau updates, thinning and imputed
Vstar controls need an explicit scope audit; no claim that an undocumented native
prediction-file interface is required. Native RNG identity, scientific correctness,
file compatibility and clinical validity remain separate. Notices, licensing,
packaging restrictions and permissions are unchanged. Final acceptance is recorded
in the adjacent validation manifest.

A separate [interval audit](wfmm-interval-audit-2026-10-12.md) preserves 28
failed/reconstruction-inconsistent native configurations. These are research
evidence, not accepted interval implementations or native compatibility passes.

## Acceptance

77,064 full tests passed in 485.98s with all extras and warnings as errors.
932 focused checks and 960 fresh-wheel tests (including 28 plot/TIFF checks),
five examples, dependency checks, lint, formatting, types and build passed.
All 778 production Python files, five data files and 1,606 test/fixture files
stayed unchanged. Both artifacts contain 783 matching package files and 29
unchanged retained notices; original binaries are excluded. Artifacts precede
final research/acceptance documentation. See
[validation manifest](wfmm-spatial-validation-2026-10-12.json).
