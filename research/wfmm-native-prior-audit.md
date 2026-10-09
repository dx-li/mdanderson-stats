# WFMM native variance-prior recovery

On October 8, 2026, the official WFMM 3.1 Linux bundle and example archive
became directly retrievable despite the catalog detail/API routes returning
HTTP 500. The Linux bundle contains original executables, supporting shared
libraries, the user guide and release notes. Its executables retain debug
symbols but the archive supplies no C++ source. The binaries and libraries
remain ignored research artifacts and are not redistributed by this project.

## Verified contract

`wfmm1 input.mat` was run on eight wholly synthetic inputs; none used the
pancreatic example's observations. The archived program identifies itself as
WFMM1 version 3.1.0, built September 14, 2015. Native initialization writes
`omega_MLE`, `prior_omega_a` and `prior_omega_b` to `input_Init.mat`.

For each random-effect level h, containing m_h columns of Z, the native
inverse-gamma parameters are:

```text
a[h,k] = delta_omega * m_h
b[h,k] = a[h,k] * omega_MLE[h,k]
```

For residual stratum c, replace m_h with the number of curves in that stratum.
The density convention is `v**(-a-1) * exp(-b/v)`. These are shape and scale,
equivalently Gamma shape/rate parameters for precision. Partition size does
not multiply a. Native outputs verify changes in delta and sample size,
unequal residual strata, two random-effect levels, and 16 coefficients split
into wavelet or identity partitions of sizes 4, 4 and 8.

Consequently `E[1/v] = 1/omega_MLE`. The input is not the inverse-gamma mean
or mode: the mean is infinite when a <= 1 and otherwise b/(a-1); the mode
is b/(a+1). This matters for the very small native default `delta_omega=1e-4`.
The guide's informal descriptions of prior information were insufficient
to infer the mapping without the recovered native outputs.

`wfmm_variance_prior` implements this contract for explicit positive variance
matrices and counts. It returns a `WFMMPrior` usable by the existing sampler,
with bounded dimensions/counts and rejection of unrepresentable parameters.
It does not reproduce the native MOM/profile optimizer or infer proposal SDs.
Using Python REML estimates changes the centering estimates explicitly.

## Reproduction and validation

The primary archive, size 30,728,770 bytes, is
[wfmm_v3_1_linux_x86_64.tar.gz](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/WFMM/wfmm_v3_1_linux_x86_64.tar.gz),
SHA-256 `9bc5003271eebfac4a90e265d08c8f677d72252eccae5ff65c90bc25d5d309f9`.
The companion
[example archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/WFMM/wfmm_v3_1_Example.zip)
is 13,673,524 bytes, SHA-256
`a6d641875897f3339d6cff49b767c7e6914c1c84dd8b726f26e9710b65a326ac`.

`tools/reference_wfmm_variance_prior.py` verifies the Linux archive checksum,
extracts only the executable and libraries into a temporary directory,
generates the synthetic MATLAB input structs, and runs initialization with
single-threaded BLAS and a 20-second limit per case. It reads native outputs
without calculating the expected mapping. `--check` verifies the saved
fixture within `1e-11` relative / `1e-12` absolute tolerance. The executable
checksum and all native shape/scale/centering values are recorded in
`tests/fixtures/wfmm-native-variance-prior.json`.

```bash
uv run --locked --extra plot python tools/reference_wfmm_variance_prior.py \
  --archive research/raw/source-recovery-2026-10-08/wfmm_v3_1_linux_x86_64.tar.gz \
  --check
```

Thirty-four new tests cover eight original-program references, default
precision centering, input ownership, invalid/unrepresentable parameters and
composition with estimated-variance MCMC. All 45 combined prior, initializer,
model and shrinkage tests passed with warnings treated as errors. This small
sampler integration checks API composition, not applied-chain convergence.

## Remaining native gaps

The native MOM/profile variance estimator and automatic proposal calibration
remain separate from the verified prior mapping. Additional transform
families, extension/truncation conventions, automatic energy compression and
native output-file workflows also remain partial. Energy compression was
subsequently verified in [the compression audit](wfmm-native-compression-audit.md). In particular, a small
periodic Haar run with `extended_mode=0` retained only two of four possible
coordinates; it must not be treated as the existing full Python orthogonal
transform. A synthetic PC run terminated unsuccessfully and supplies no
valid PCA reference. No parity is claimed from those exploratory runs.

The original pancreatic example archive is recovered and its dimensions and
model settings were inspected. Its fitted workflow has not been rerun; it
does not provide independent evidence for the Python sampler's convergence.
